#!/usr/bin/env python
"""Avatar 3D rigged controlado por Kinect v1 (Xbox 360).

Pipeline en tiempo real:
1. libfreenect entrega RGB y profundidad *registrada* (alineada al RGB, mm).
2. El RGB se realza (CLAHE + gamma) para que la pose funcione con poca luz.
3. MediaPipe Pose estima 33 landmarks: 2D en imagen y 3D metricos
   (``pose_world_landmarks``, en metros y centrados en la cadera).
4. La profundidad de la Kinect situa a la persona en la sala (desplazamiento
   lateral y de profundidad del avatar).
5. Un filtro One Euro elimina el temblor sin anadir latencia apreciable.
6. Retargeting con cinematica directa sobre el bind pose real del rig: cada
   hueso gira desde su direccion heredada hacia la direccion medida, y la
   rotacion local se obtiene como ``R_local = R_padre^T * R_mundo``.
7. Panda3D renderiza el modelo; los pies se pegan al suelo automaticamente.

Uso:
    .venv/bin/python avatar.py [--no-mirror] [--no-translate] [--tilt 0]

Teclas: M espejo | C camara | T pose de prueba | R recentrar | Esc salir
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import freenect
except ImportError:
    freenect = None

try:
    import cv2
except ImportError:
    cv2 = None

HERE = Path(__file__).resolve().parent
W, H = 640, 480
# Intrinsecos aproximados de la camara RGB de Kinect v1 (la profundidad
# registrada esta en el mismo sistema de pixeles).
FX, FY, CX, CY = 525.0, 525.0, 319.5, 239.5

# Indices de MediaPipe Pose (33 landmarks).
LM = {
    "nose": 0, "L_ear": 7, "R_ear": 8,
    "L_shoulder": 11, "R_shoulder": 12,
    "L_elbow": 13, "R_elbow": 14,
    "L_wrist": 15, "R_wrist": 16,
    "L_index": 19, "R_index": 20,
    "L_hip": 23, "R_hip": 24,
    "L_knee": 25, "R_knee": 26,
    "L_ankle": 27, "R_ankle": 28,
    "L_heel": 29, "R_heel": 30,
    "L_foot": 31, "R_foot": 32,
}

SKELETON_2D = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
               (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32), (7, 8)]

# Correspondencia de roles -> huesos del rig RobotExpressive. Para otro modelo
# humanoide basta con cambiar estos nombres.
RIG = {
    "root": "Bone",
    "body": "Body",
    "hips": "Hips",
    "abdomen": "Abdomen",
    "spine": "Torso",
    "neck": "Neck",
    "head": "Head",
    "upper_arm_L": "UpperArm.L", "lower_arm_L": "LowerArm.L", "hand_L": "Palm2.L",
    "upper_arm_R": "UpperArm.R", "lower_arm_R": "LowerArm.R", "hand_R": "Palm2.R",
    "upper_leg_L": "UpperLeg.L", "lower_leg_L": "LowerLeg.L",
    "upper_leg_R": "UpperLeg.R", "lower_leg_R": "LowerLeg.R",
    # En este rig los pies cuelgan de la raiz (rig de IK), no de la pierna; la
    # posicion de bind del pie marca el tobillo.
    "foot_L": "Foot.L", "foot_R": "Foot.R",
}

# (rol del hueso, rol cuya posicion de bind marca el extremo del hueso,
#  landmark inicio, landmark fin)
LIMBS = [
    ("upper_arm_L", "lower_arm_L", "L_shoulder", "L_elbow"),
    ("lower_arm_L", "hand_L", "L_elbow", "L_wrist"),
    ("upper_arm_R", "lower_arm_R", "R_shoulder", "R_elbow"),
    ("lower_arm_R", "hand_R", "R_elbow", "R_wrist"),
    ("upper_leg_L", "lower_leg_L", "L_hip", "L_knee"),
    ("lower_leg_L", "foot_L", "L_knee", "L_ankle"),
    ("upper_leg_R", "lower_leg_R", "R_hip", "R_knee"),
    ("lower_leg_R", "foot_R", "R_knee", "R_ankle"),
]

VIS_THRESHOLD = 0.5


# --------------------------------------------------------------------------
# Algebra de rotaciones (matrices 3x3 en convencion de vector columna)
# --------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def frame_from(left: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Base ortonormal (izquierda, arriba, delante) a partir de dos vectores."""
    u = normalize(up)
    l = normalize(left - u * np.dot(left, u))
    f = np.cross(l, u)
    return np.column_stack([l, u, f])


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotacion minima que lleva el vector unitario a sobre b (Rodrigues)."""
    a, b = normalize(a), normalize(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.9999:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = normalize(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Matriz de rotacion -> cuaternion (w, x, y, z)."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def slerp_mat(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Interpolacion esferica entre dos rotaciones."""
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    qa, qb = mat_to_quat(a), mat_to_quat(b)
    d = float(np.dot(qa, qb))
    if d < 0:
        qb, d = -qb, -d
    if d > 0.9995:
        q = qa + t * (qb - qa)
    else:
        theta = np.arccos(d)
        q = (np.sin((1 - t) * theta) * qa + np.sin(t * theta) * qb) / np.sin(theta)
    return quat_to_mat(q / np.linalg.norm(q))


# --------------------------------------------------------------------------
# Captura y estimacion de pose (hilo de fondo)
# --------------------------------------------------------------------------

class KinectSource:
    """Captura sincrona de RGB + profundidad registrada via libfreenect."""

    def __init__(self, device_index: int = 0):
        if freenect is None:
            raise RuntimeError(
                "No se pudo importar freenect. Instalalo con: "
                ".venv/bin/python -m pip install freenect (requiere libfreenect nativa)."
            )
        self.device_index = device_index

    def read(self):
        try:
            depth, _ = freenect.sync_get_depth(index=self.device_index, format=freenect.DEPTH_REGISTERED)
            rgb, _ = freenect.sync_get_video(index=self.device_index)
        except Exception as exc:
            raise RuntimeError(f"No se pudo leer la Kinect: {exc}") from exc
        if depth is None or rgb is None:
            raise RuntimeError("La Kinect devolvio un frame vacio (USB/permisos?).")
        return np.asarray(depth, dtype=np.uint16), np.asarray(rgb, dtype=np.uint8)


def set_tilt(device_index: int, degrees: float) -> None:
    """Mueve el motor de inclinacion (-27..27 grados) antes de iniciar la captura."""
    ctx = freenect.init()
    dev = freenect.open_device(ctx, device_index)
    try:
        freenect.set_tilt_degs(dev, float(np.clip(degrees, -27, 27)))
        time.sleep(1.5)
    finally:
        freenect.close_device(dev)
        freenect.shutdown(ctx)


def enhance(rgb: np.ndarray) -> np.ndarray:
    """Realza imagenes oscuras: gamma adaptativa + CLAHE sobre la luminancia."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0]
    mean = float(l.mean())
    if mean < 110:
        gamma = np.clip(np.log(110 / 255) / np.log(max(mean, 5) / 255), 1.0, 3.5)
        lut = (np.power(np.arange(256) / 255.0, 1.0 / gamma) * 255).astype(np.uint8)
        l = cv2.LUT(l, lut)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(l)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class PoseEstimator:
    """MediaPipe Pose Landmarker en modo video."""

    def __init__(self, task_path: str):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.mp = mp
        self._last_ts = -1

    def process(self, rgb: np.ndarray, timestamp_ms: int):
        """Devuelve (landmarks imagen [33x4], landmarks mundo [33x3]) o None."""
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks or not result.pose_world_landmarks:
            return None
        img = np.array([[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks[0]], dtype=np.float32)
        world = np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]], dtype=np.float64)
        return img, world

    def close(self):
        self.landmarker.close()


def sample_depth(depth: np.ndarray, u: float, v: float, window: int = 15) -> Optional[float]:
    """Mediana de la profundidad valida (mm) alrededor de un pixel."""
    px = int(np.clip(round(u * W), 0, W - 1))
    py = int(np.clip(round(v * H), 0, H - 1))
    half = window // 2
    region = depth[max(py - half, 0):py + half + 1, max(px - half, 0):px + half + 1]
    valid = region[(region > 300) & (region < 6000)]
    if valid.size < 6:
        return None
    return float(np.median(valid))


def build_observation(img_lm: np.ndarray, world_lm: np.ndarray, depth: np.ndarray) -> dict:
    """Landmarks -> puntos 3D en espacio de persona + posicion en la sala.

    Espacio de persona (metros, origen en la pelvis): x = derecha de la imagen,
    y = arriba, z = hacia la camara. Para alguien de frente a la camara, x
    coincide con su lado izquierdo anatomico.
    """
    points, vis = {}, {}
    for name, idx in LM.items():
        x, y, z = world_lm[idx]
        points[name] = np.array([x, -y, -z])
        vis[name] = float(img_lm[idx, 3])
    pelvis = (points["L_hip"] + points["R_hip"]) / 2
    for name in points:
        points[name] = points[name] - pelvis

    # Posicion en la sala con la profundidad de la Kinect (torso = zona fiable).
    root = None
    samples = []
    for name in ("L_hip", "R_hip", "L_shoulder", "R_shoulder"):
        u, v = img_lm[LM[name], 0], img_lm[LM[name], 1]
        if 0 <= u <= 1 and 0 <= v <= 1:
            z = sample_depth(depth, u, v)
            if z is not None:
                samples.append(z)
    if samples:
        z = float(np.median(samples)) / 1000.0
        su = float((img_lm[LM["L_shoulder"], 0] + img_lm[LM["R_shoulder"], 0]
                    + img_lm[LM["L_hip"], 0] + img_lm[LM["R_hip"], 0]) / 4)
        root = np.array([(su * W - CX) * z / FX, z])  # (lateral, distancia)
    return {"points": points, "vis": vis, "root": root}


class TrackerThread:
    """Captura + realce + MediaPipe en segundo plano: el render nunca se bloquea."""

    def __init__(self, kinect: KinectSource, estimator: PoseEstimator, enhance_image: bool):
        self.kinect = kinect
        self.estimator = estimator
        self.enhance_image = enhance_image
        self.lock = threading.Lock()
        self.observation: Optional[dict] = None
        self.preview: Optional[np.ndarray] = None
        self.sequence = 0
        self.error: Optional[str] = None
        self.fps = 0.0
        self.brightness = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)

    def _loop(self):
        last = time.perf_counter()
        while self.running:
            try:
                depth, rgb = self.kinect.read()
            except RuntimeError as exc:
                with self.lock:
                    self.error = str(exc)
                time.sleep(0.2)
                continue
            brightness = float(rgb.mean())
            image = enhance(rgb) if self.enhance_image else rgb
            result = self.estimator.process(image, time.monotonic_ns() // 1_000_000)
            obs = None
            preview = image.copy()
            if result is not None:
                img_lm, world_lm = result
                obs = build_observation(img_lm, world_lm, depth)
                obs["t"] = time.perf_counter()
                for a, b in SKELETON_2D:
                    if img_lm[a, 3] > VIS_THRESHOLD and img_lm[b, 3] > VIS_THRESHOLD:
                        pa = (int(img_lm[a, 0] * W), int(img_lm[a, 1] * H))
                        pb = (int(img_lm[b, 0] * W), int(img_lm[b, 1] * H))
                        cv2.line(preview, pa, pb, (80, 255, 120), 3)
            now = time.perf_counter()
            with self.lock:
                self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)
                last = now
                self.brightness = brightness
                self.observation = obs
                self.preview = preview
                self.sequence += 1
                self.error = None

    def get(self):
        with self.lock:
            return self.observation, self.preview, self.sequence, self.error


# --------------------------------------------------------------------------
# Filtrado
# --------------------------------------------------------------------------

class OneEuro:
    """Filtro One Euro (Casiez et al. 2012) sobre vectores."""

    def __init__(self, min_cutoff=2.5, beta=4.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None
        self.t = None

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (x - self.x) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = self._alpha(dt, cutoff)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


# --------------------------------------------------------------------------
# Retargeting sobre el rig
# --------------------------------------------------------------------------

class RigDriver:
    """Cinematica directa sobre el bind pose real del esqueleto del modelo."""

    def __init__(self, actor, part: str = "modelRoot"):
        from panda3d.core import CharacterJoint

        self.order: list[str] = []
        self.parent: dict[str, Optional[str]] = {}
        self.bind_R: dict[str, np.ndarray] = {}
        self.bind_t: dict[str, np.ndarray] = {}

        def walk(group, parent):
            for child in group.getChildren():
                if isinstance(child, CharacterJoint):
                    name = child.getName()
                    m = child.getDefaultValue()
                    rows = np.array([[m.getCell(r, c) for c in range(3)] for r in range(3)])
                    rot = rows.T  # Panda usa vectores fila
                    rot = rot / np.linalg.norm(rot, axis=0, keepdims=True)
                    self.order.append(name)
                    self.parent[name] = parent
                    self.bind_R[name] = rot
                    self.bind_t[name] = np.array([m.getCell(3, c) for c in range(3)])
                    walk(child, name)
                else:
                    walk(child, parent)

        walk(actor.getPartBundle(part), None)
        missing = [n for n in RIG.values() if n not in self.parent]
        if missing:
            raise RuntimeError(f"El rig no tiene los huesos: {missing}")

        self.bind_WR, self.bind_Wt = self._fk(self.bind_R, self.bind_t)
        j = lambda role: RIG[role]
        self.up_b = normalize(self.bind_Wt[j("head")] - self.bind_Wt[j("hips")])
        self.left_b = self.bind_Wt[j("upper_arm_L")] - self.bind_Wt[j("upper_arm_R")]
        self.frame_b = frame_from(self.left_b, self.up_b)
        self.left_b = self.frame_b[:, 0]
        # C: espacio de persona (izq, arriba, delante) -> espacio del esqueleto.
        self.C = self.frame_b
        # Vector de cada hueso de extremidad hasta su extremo, en su propio
        # sistema local (direccion de reposo y longitud).
        self.limb_vec = {}
        for role, end_role, _, _ in LIMBS:
            name = j(role)
            self.limb_vec[name] = self.bind_WR[name].T @ (self.bind_Wt[j(end_role)] - self.bind_Wt[name])
        self.bind_foot_h = min(float(self.bind_Wt[j(f"foot_{s}")] @ self.up_b) for s in "LR")

        controlled = ["root", "body", "abdomen", "spine", "neck", "head", "foot_L", "foot_R"]
        controlled += [limb[0] for limb in LIMBS]
        self.controls = {RIG[r]: actor.controlJoint(None, part, RIG[r]) for r in controlled}
        # Peso (0..1) con que cada rol sigue al usuario; se suaviza en el tiempo.
        self.weights = {r: 0.0 for r in ["torso", "head"] + [limb[0] for limb in LIMBS]}

    def _fk(self, local_R: dict, local_t: dict):
        WR, Wt = {}, {}
        for name in self.order:
            p = self.parent[name]
            if p is None:
                WR[name], Wt[name] = local_R[name], local_t[name]
            else:
                WR[name] = WR[p] @ local_R[name]
                Wt[name] = Wt[p] + WR[p] @ local_t[name]
        return WR, Wt

    def bone_length(self, role_a: str, role_b: str) -> float:
        return float(np.linalg.norm(self.bind_Wt[RIG[role_a]] - self.bind_Wt[RIG[role_b]]))

    def solve(self, points: Optional[dict], vis: Optional[dict], dt: float) -> None:
        """Calcula y aplica la pose. points=None relaja el avatar al bind pose."""
        rate = 1.0 - np.exp(-dt * 12.0)

        def visible(*names):
            return points is not None and all(vis.get(n, 0.0) >= VIS_THRESHOLD for n in names)

        for limb in LIMBS:
            target = 1.0 if visible(limb[2], limb[3]) else 0.0
            self.weights[limb[0]] += (target - self.weights[limb[0]]) * rate
        torso_ok = visible("L_shoulder", "R_shoulder")
        self.weights["torso"] += ((1.0 if torso_ok else 0.0) - self.weights["torso"]) * rate
        head_ok = visible("L_ear", "R_ear", "L_shoulder", "R_shoulder")
        self.weights["head"] += ((1.0 if head_ok else 0.0) - self.weights["head"]) * rate

        C, Fb = self.C, self.frame_b
        to_b = lambda v: C @ v
        local_R = dict(self.bind_R)
        local_t = dict(self.bind_t)
        WR, Wt = {}, {}

        # Rotaciones objetivo (mundo del esqueleto) del tronco y la cabeza.
        R_body = R_spine = R_head = np.eye(3)
        if points is not None:
            hips_ok = visible("L_hip", "R_hip")
            shoulder_line = points["L_shoulder"] - points["R_shoulder"]
            hip_line = points["L_hip"] - points["R_hip"] if hips_ok else shoulder_line
            yaw_line = hip_line.copy()
            yaw_line[1] = 0.0
            R_body = frame_from(to_b(yaw_line), self.up_b) @ Fb.T
            chest = (points["L_shoulder"] + points["R_shoulder"]) / 2
            spine_up = chest if hips_ok else np.array([0.0, 1.0, 0.0])
            R_spine = frame_from(to_b(shoulder_line), to_b(spine_up)) @ Fb.T
            ears = (points["L_ear"] + points["R_ear"]) / 2
            R_head = frame_from(to_b(points["L_ear"] - points["R_ear"]), to_b(ears - chest)) @ Fb.T
        wt, wh = self.weights["torso"], self.weights["head"]
        R_body = slerp_mat(np.eye(3), R_body, wt)
        R_spine = slerp_mat(np.eye(3), R_spine, wt)
        R_head = slerp_mat(np.eye(3), R_head, min(wt, wh) if wh > 0 else 0.0)

        world_targets = {
            RIG["body"]: R_body,
            RIG["abdomen"]: slerp_mat(R_body, R_spine, 0.5),
            RIG["spine"]: R_spine,
            RIG["neck"]: slerp_mat(R_spine, R_head, 0.5),
            RIG["head"]: R_head,
        }
        limb_by_joint = {RIG[l[0]]: l for l in LIMBS}

        for name in self.order:
            p = self.parent[name]
            pR = WR[p] if p is not None else np.eye(3)
            pt = Wt[p] if p is not None else np.zeros(3)
            if name in world_targets:
                target = world_targets[name] @ self.bind_WR[name]
                local_R[name] = pR.T @ target
            elif name in limb_by_joint and points is not None and self.weights[limb_by_joint[name][0]] > 1e-3:
                role, child_role, a, b = limb_by_joint[name]
                inherited = pR @ self.bind_R[name]
                current_dir = inherited @ self.limb_vec[name]
                desired = to_b(points[b] - points[a])
                if np.linalg.norm(desired) > 1e-4:
                    world = rot_between(current_dir, desired) @ inherited
                    local_R[name] = slerp_mat(self.bind_R[name], pR.T @ world, self.weights[role])
            WR[name] = pR @ local_R[name]
            Wt[name] = pt + pR @ local_t[name]

        # Pies: cuelgan de la raiz, asi que se recolocan bajo cada tobillo.
        root = RIG["root"]
        for s in "LR":
            foot, shin = RIG[f"foot_{s}"], RIG[f"lower_leg_{s}"]
            pos = Wt[shin] + WR[shin] @ self.limb_vec[shin]
            local_t[foot] = WR[root].T @ (pos - Wt[root])
            local_R[foot] = WR[root].T @ (R_body @ self.bind_WR[foot])
            Wt[foot] = pos

        # Contacto con el suelo: el pie mas bajo queda a la altura del bind pose.
        lowest = min(float(Wt[RIG[f"foot_{s}"]] @ self.up_b) for s in "LR")
        local_t[root] = self.bind_t[root] + (self.bind_foot_h - lowest) * self.up_b

        self._apply(local_R, local_t)

    def _apply(self, local_R: dict, local_t: dict) -> None:
        from panda3d.core import LMatrix3f, Quat

        for name, ctrl in self.controls.items():
            r = local_R[name].T  # de vuelta a la convencion de Panda
            q = Quat()
            q.setFromMatrix(LMatrix3f(*r.flatten().tolist()))
            ctrl.setQuat(q)
            ctrl.setPos(*local_t[name].tolist())


def mirror_points(points: dict) -> dict:
    """Espejo: refleja x e intercambia izquierda/derecha (anatomia valida)."""
    out = {}
    for name, p in points.items():
        if name.startswith("L_"):
            other = "R_" + name[2:]
        elif name.startswith("R_"):
            other = "L_" + name[2:]
        else:
            other = name
        out[other] = np.array([-p[0], p[1], p[2]])
    return out


def make_test_person(t: float) -> tuple[dict, dict]:
    """Pose sintetica (espacio de persona): saluda con el brazo izquierdo y flexiona la pierna."""
    wave = float(np.sin(t * 6.0))
    lift = max(0.0, float(np.sin(t * 1.5)))
    p = {
        "nose": [0.0, 0.62, 0.1], "L_ear": [0.07, 0.62, 0.0], "R_ear": [-0.07, 0.62, 0.0],
        "L_shoulder": [0.18, 0.48, 0.0], "R_shoulder": [-0.18, 0.48, 0.0],
        "L_elbow": [0.42, 0.52, 0.0], "L_wrist": [0.45 + 0.12 * wave, 0.78, 0.05],
        "L_index": [0.46 + 0.12 * wave, 0.86, 0.05],
        "R_elbow": [-0.22, 0.2, 0.05], "R_wrist": [-0.2, -0.02, 0.15], "R_index": [-0.2, -0.08, 0.17],
        "L_hip": [0.1, 0.0, 0.0], "R_hip": [-0.1, 0.0, 0.0],
        "L_knee": [0.11, -0.43 + 0.3 * lift, 0.3 * lift], "L_ankle": [0.11, -0.85 + 0.35 * lift, 0.05 * lift],
        "R_knee": [-0.11, -0.43, 0.0], "R_ankle": [-0.11, -0.85, 0.0],
        "L_heel": [0.11, -0.88 + 0.35 * lift, -0.03], "R_heel": [-0.11, -0.88, -0.03],
        "L_foot": [0.11, -0.9 + 0.35 * lift, 0.12], "R_foot": [-0.11, -0.9, 0.12],
    }
    points = {k: np.array(v, dtype=np.float64) for k, v in p.items()}
    return points, {k: 1.0 for k in points}


# --------------------------------------------------------------------------
# Escena Panda3D
# --------------------------------------------------------------------------

AVATAR_HEIGHT = 1.75


def build_two_segment_arms(actor) -> None:
    """Crea brazo y antebrazo visibles a partir de la malla del brazo.

    El glTF del robot trae cada brazo como UN palo con una bolita (la mano),
    pegado al hueso del brazo superior: los huesos del codo y la muneca
    existen pero no mueven nada visible. Aqui se reutiliza ese palo+bolita
    dos veces, estirado a la longitud real de cada hueso:

    - brazo:     hombro -> codo   (la bolita hace de codo), ligado a UpperArm
    - antebrazo: codo -> muneca   (la bolita hace de mano), ligado a LowerArm

    Los nodos nuevos cuelgan de la armadura (como las piezas rigidas del
    modelo) y siguen al hueso con add_net_transform, asi comparten espacio de
    coordenadas con la malla original, que se oculta.
    """
    from panda3d.core import (Geom, GeomNode, GeomVertexData, GeomVertexReader,
                              GeomVertexWriter, OmniBoundingVolume, PandaNode)

    bundle = actor.getPartBundle("modelRoot")

    def follow(armature, joint, tag):
        node = PandaNode(tag)
        bundle.findChild(joint).addNetTransform(node)
        return armature.attachNewNode(node)

    for side in "LR":
        geom_np = next((n for n in actor.findAllMatches(f"**/Arm.{side}")
                        if n.node().isGeomNode()), None)
        if geom_np is None:
            continue
        armature = geom_np.getParent().getParent().getParent()
        upper_np = follow(armature, RIG[f"upper_arm_{side}"], f"UpperArm.{side}-seg")
        lower_np = follow(armature, RIG[f"lower_arm_{side}"], f"ForeArm.{side}-seg")
        wrist_np = follow(armature, RIG[f"hand_{side}"], f"Wrist.{side}-probe")
        actor.update(force=True)
        shoulder = np.array(upper_np.getPos(armature))
        elbow = np.array(lower_np.getPos(armature))
        wrist = np.array(wrist_np.getPos(armature))
        wrist_np.removeNode()

        gnode = geom_np.node()
        geom = gnode.getGeom(0)
        state = gnode.getGeomState(0)
        vdata = geom.getVertexData()
        to_arm = geom_np.getMat(armature)
        rd_v, rd_n = GeomVertexReader(vdata, "vertex"), GeomVertexReader(vdata, "normal")
        pos, nor = [], []
        while not rd_v.isAtEnd():
            q = to_arm.xformPoint(rd_v.getData3())
            pos.append([q[0], q[1], q[2]])
            nv = to_arm.xformVec(rd_n.getData3())
            nor.append([nv[0], nv[1], nv[2]])
        pos, nor = np.array(pos), np.array(nor)

        # Eje del palo (de la punta del hombro hacia la bolita) por PCA.
        c = pos.mean(axis=0)
        ax = np.linalg.svd(pos - c)[2][0]
        t = (pos - c) @ ax
        order = np.sort(t)
        cut = int(np.argmax(np.diff(order)))          # mayor hueco: palo | bolita
        gap = 0.5 * (order[cut] + order[cut + 1])
        # El palo debe quedar al lado del hombro (tronco = origen de la armadura).
        if (c + ax * t.min())[0] ** 2 + (c + ax * t.min())[2] ** 2 > \
                (c + ax * t.max())[0] ** 2 + (c + ax * t.max())[2] ** 2:
            ax, t, gap = -ax, -t, -gap
            ball_side = t < gap
        else:
            ball_side = t > gap
        stick_end = t[~ball_side]
        origin = c + ax * float(stick_end.mean())
        ball_t = float(t[ball_side].mean())
        old_len = ball_t - float(stick_end.mean())
        rel = pos - origin
        radial = rel - np.outer(rel @ ax, ax)
        t_rel = rel @ ax

        def segment(start, end, joint_np):
            direction = end - start
            length = float(np.linalg.norm(direction))
            rot = rot_between(ax, direction)
            t_new = t_rel + np.where(ball_side, length - old_len, 0.0)
            world = start + (np.outer(t_new, ax) + radial) @ rot.T
            normals = nor @ rot.T
            ref = armature.attachNewNode("ref")
            m = ref.getMat(joint_np)
            ref.removeNode()
            data = GeomVertexData(vdata)
            wv, wn = GeomVertexWriter(data, "vertex"), GeomVertexWriter(data, "normal")
            for q, nv in zip(world, normals):
                local = m.xformPoint((float(q[0]), float(q[1]), float(q[2])))
                wv.setData3(local)
                wn.setData3(m.xformVec((float(nv[0]), float(nv[1]), float(nv[2]))))
            g = Geom(data)
            for pi in range(geom.getNumPrimitives()):
                g.addPrimitive(geom.getPrimitive(pi))
            node = GeomNode(f"Segment.{side}")
            node.addGeom(g, state)
            node.setBounds(OmniBoundingVolume())
            joint_np.attachNewNode(node)

        segment(shoulder, elbow, upper_np)
        segment(elbow, wrist, lower_np)
        geom_np.hide()


def build_scene(base, model_path: str):
    """Carga el modelo, lo orienta hacia la camara, monta luces y suelo."""
    from direct.actor.Actor import Actor
    from panda3d.core import AmbientLight, CardMaker, DirectionalLight, LineSegs

    holder = base.render.attachNewNode("avatar-holder")
    pivot = holder.attachNewNode("avatar-pivot")
    actor = Actor(base.loader.loadModel(model_path))
    actor.reparentTo(pivot)
    build_two_segment_arms(actor)
    driver = RigDriver(actor)

    # El esqueleto vive bajo el nodo de armadura del glTF (RobotArmature, con
    # rotacion y escala x100); su transformacion lleva el espacio de los
    # huesos al de la escena.
    from panda3d.core import CharacterJointEffect, LPoint3f
    armature = actor
    for np_ in actor.findAllMatches("**"):
        if np_.node().getEffect(CharacterJointEffect.getClassType()) is not None:
            armature = np_.getParent()
            break

    def joint_pos(role):
        p = driver.bind_Wt[RIG[role]]
        return np.array(base.render.getRelativePoint(armature, LPoint3f(*p.tolist())))

    pos = {r: joint_pos(r) for r in ("head", "hips", "upper_arm_L", "upper_arm_R", "foot_L", "foot_R")}
    up = normalize(pos["head"] - pos["hips"])
    left = pos["upper_arm_L"] - pos["upper_arm_R"]
    forward = np.cross(normalize(left - up * (left @ up)), up)
    # Girar para que el avatar mire a la camara (-Y en Panda).
    actor.setH(np.degrees(np.arctan2(forward[0], -forward[1])))
    lo, hi = actor.getTightBounds(base.render)
    scale = AVATAR_HEIGHT / max(hi[2] - lo[2], 1e-6)
    pivot.setScale(scale)
    pivot.setZ(-lo[2] * scale)

    ambient = AmbientLight("ambient")
    ambient.setColor((0.55, 0.55, 0.62, 1))
    base.render.setLight(base.render.attachNewNode(ambient))
    sun = DirectionalLight("sun")
    sun.setColor((1.0, 0.97, 0.92, 1))
    sun_np = base.render.attachNewNode(sun)
    sun_np.setHpr(25, -40, 0)
    base.render.setLight(sun_np)
    fill = DirectionalLight("fill")
    fill.setColor((0.35, 0.4, 0.55, 1))
    fill_np = base.render.attachNewNode(fill)
    fill_np.setHpr(-150, -10, 0)
    base.render.setLight(fill_np)
    base.setBackgroundColor(0.07, 0.08, 0.12)

    cm = CardMaker("floor")
    cm.setFrame(-6, 6, -4, 8)
    floor = base.render.attachNewNode(cm.generate())
    floor.setP(-90)
    floor.setZ(-0.002)
    floor.setColor(0.12, 0.14, 0.18, 1)
    grid = LineSegs("grid")
    grid.setColor(0.24, 0.27, 0.34, 1)
    for i in range(-6, 7):
        grid.moveTo(i * 0.5, -2, 0)
        grid.drawTo(i * 0.5, 4, 0)
    for i in range(-4, 9):
        grid.moveTo(-3, i * 0.5, 0)
        grid.drawTo(3, i * 0.5, 0)
    base.render.attachNewNode(grid.create())

    base.disableMouse()
    base.camLens.setFov(50)
    base.cam.setPos(0, -4.0, 1.3)
    base.cam.lookAt(0, 0, 0.85)
    return holder, actor, driver


class CameraInset:
    """Vista de la camara (con esqueleto 2D) en una esquina de la ventana."""

    def __init__(self, base):
        from panda3d.core import CardMaker, Texture

        self.tex = Texture("kinect-rgb")
        self.tex.setup2dTexture(320, 240, Texture.TUnsignedByte, Texture.FRgb)
        cm = CardMaker("inset")
        cm.setFrame(0, 1, 0, 0.75)
        self.card = base.a2dBottomRight.attachNewNode(cm.generate())
        self.card.setScale(0.62)
        self.card.setPos(-0.64, 0, 0.02)
        self.card.setTexture(self.tex)
        self.visible = True

    def toggle(self):
        self.visible = not self.visible
        self.card.show() if self.visible else self.card.hide()

    def update(self, rgb: np.ndarray, mirror: bool):
        if not self.visible or rgb is None:
            return
        small = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
        if mirror:
            small = small[:, ::-1]
        # Panda guarda las texturas en BGR y de abajo hacia arriba.
        bgr = np.ascontiguousarray(small[::-1, :, ::-1])
        self.tex.setRamImage(bgr.tobytes())


def run(args) -> int:
    if cv2 is None:
        print("OpenCV no esta disponible.", file=sys.stderr)
        return 1

    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "window-title Avatar Kinect v1\nsync-video 1\nframebuffer-multisample 1\nmultisamples 4")
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import AntialiasAttrib, TextNode

    task_path = args.task
    if task_path is None:
        full = HERE / "models" / "pose_landmarker_full.task"
        task_path = str(full if full.exists() else HERE / "models" / "pose_landmarker_lite.task")

    try:
        kinect = KinectSource(args.device)
        if args.tilt is not None:
            set_tilt(args.device, args.tilt)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1

    estimator = PoseEstimator(task_path)
    base = ShowBase()
    base.render.setAntialias(AntialiasAttrib.MMultisample)

    try:
        holder, actor, driver = build_scene(base, args.model)
    except Exception as exc:
        print(f"No se pudo cargar el modelo {args.model}: {exc}", file=sys.stderr)
        base.destroy()
        return 1

    tracker = TrackerThread(kinect, estimator, args.enhance)
    tracker.start()
    inset = CameraInset(base)

    def text(pos, scale, color, align=TextNode.ALeft, parent=None):
        node = TextNode("text")
        node.setAlign(align)
        node.setTextColor(*color)
        node.setShadow(0.05, 0.05)
        np_ = (parent or base.a2dTopLeft).attachNewNode(node)
        np_.setScale(scale)
        np_.setPos(*pos)
        return node

    help_text = text((0.05, 0, -0.08), 0.05, (0.8, 0.85, 0.95, 1))
    status = text((0.05, 0, -0.16), 0.045, (1.0, 0.75, 0.55, 1))

    state = {
        "mirror": args.mirror, "test_until": 0.0, "last_seq": -1, "last_seen": 0.0,
        "filters": {}, "points": None, "vis": None, "root_ref": None, "root": np.zeros(2),
        "translate": args.translate,
    }

    def toggle_mirror():
        state["mirror"] = not state["mirror"]
        state["filters"].clear()
        state["root_ref"] = None

    def recenter():
        state["root_ref"] = None

    base.accept("m", toggle_mirror)
    base.accept("c", inset.toggle)
    base.accept("t", lambda: state.update(test_until=time.perf_counter() + 8.0))
    base.accept("r", recenter)
    base.accept("escape", base.userExit)

    def update(task):
        now = time.perf_counter()
        dt = min(base.clock.getDt(), 0.1)
        help_text.setText(f"M espejo [{'si' if state['mirror'] else 'no'}] | C camara | "
                          f"T pose de prueba | R recentrar | Esc salir")

        if now < state["test_until"]:
            pts, vis = make_test_person(now)
            driver.solve(mirror_points(pts) if state["mirror"] else pts, vis, dt)
            status.setText("POSE DE PRUEBA")
            return task.cont

        obs, preview, seq, error = tracker.get()
        if seq != state["last_seq"]:
            state["last_seq"] = seq
            inset.update(preview, state["mirror"])
            if obs is not None:
                pts, vis = obs["points"], obs["vis"]
                if state["mirror"]:
                    pts = mirror_points(pts)
                    vis = {("R_" + k[2:] if k.startswith("L_") else "L_" + k[2:] if k.startswith("R_") else k): v
                           for k, v in vis.items()}
                filtered = {}
                for name, p in pts.items():
                    f = state["filters"].setdefault(name, OneEuro())
                    filtered[name] = f(p, obs["t"])
                state["points"], state["vis"], state["last_seen"] = filtered, vis, now
                if obs["root"] is not None and state["translate"]:
                    root = obs["root"].copy()
                    if state["mirror"]:
                        root[0] = -root[0]
                    if state["root_ref"] is None:
                        state["root_ref"] = root.copy()
                    state["root"] = root - state["root_ref"]

        lost = now - state["last_seen"] > 0.6
        driver.solve(None if lost else state["points"], state["vis"], dt)

        # Desplazamiento por la sala (suavizado y limitado).
        target = np.clip(state["root"], -1.5, 1.5) if (state["translate"] and not lost) else np.zeros(2)
        k = 1.0 - np.exp(-dt * 6.0)
        holder.setX(holder.getX() + (target[0] - holder.getX()) * k)
        holder.setY(holder.getY() + (target[1] - holder.getY()) * k)

        with tracker.lock:
            fps, light = tracker.fps, tracker.brightness
        if error:
            status.setText(f"Kinect: {error}")
        elif lost:
            hint = " | ESCENA MUY OSCURA: enciende una luz" if light < 40 else ""
            status.setText(f"Buscando persona... colocate de pie a 1.5-3 m{hint}  ({fps:.0f} fps)")
        else:
            legs = "piernas visibles" if driver.weights["lower_leg_L"] > 0.5 else "piernas fuera de cuadro"
            status.setText(f"Persona detectada | {legs} | {fps:.0f} fps pose")
        return task.cont

    base.taskMgr.add(update, "kinect-update")
    try:
        base.run()
    finally:
        tracker.stop()
        estimator.close()
        try:
            freenect.sync_stop()
        except Exception:
            pass
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0, help="Indice de Kinect libfreenect")
    parser.add_argument("--model", default=str(HERE / "models" / "RobotExpressive.bam"),
                        help="Modelo .bam con rig (convertido con gltf2bam)")
    parser.add_argument("--task", default=None,
                        help="Modelo .task de MediaPipe (por defecto full si existe, si no lite)")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                        help="Modo espejo (por defecto activado; tecla M)")
    parser.add_argument("--translate", action=argparse.BooleanOptionalAction, default=True,
                        help="Mover el avatar por la sala segun la profundidad de la Kinect")
    parser.add_argument("--enhance", action=argparse.BooleanOptionalAction, default=True,
                        help="Realzar la imagen RGB para escenas oscuras")
    parser.add_argument("--tilt", type=float, default=None,
                        help="Inclinacion del motor de la Kinect en grados (-27..27), p.ej. --tilt 0")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
