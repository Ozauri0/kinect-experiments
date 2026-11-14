"""Lluvia Digital Interactiva para Kinect v1.

La captura utiliza libfreenect y la visualizacion utiliza exclusivamente
OpenCV. La simulacion de particulas se mantiene en arrays de NumPy para que
la parte costosa no dependa de un bucle Python por pixel.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import freenect
except ImportError:  # Permite mostrar un error util sin ocultar el problema.
    freenect = None


WINDOW_NAME = "Lluvia Digital Interactiva | Kinect v1"
DEPTH_SIZE = (640, 480)


@dataclass
class ParticleSystem:
    """Estado vectorial de N particulas en coordenadas de pantalla."""

    count: int
    width: int
    height: int
    rng: np.random.Generator
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    radius: np.ndarray
    hue: np.ndarray

    @classmethod
    def create(cls, count: int, width: int, height: int, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        return cls(
            count=count,
            width=width,
            height=height,
            rng=rng,
            x=rng.uniform(0, width, count).astype(np.float32),
            y=rng.uniform(-height, 0, count).astype(np.float32),
            vx=rng.uniform(-0.35, 0.35, count).astype(np.float32),
            vy=rng.uniform(1.5, 4.0, count).astype(np.float32),
            radius=rng.integers(1, 3, count).astype(np.int32),
            hue=rng.integers(145, 210, count).astype(np.uint8),
        )

    def reset_particle(self, indices: np.ndarray) -> None:
        """Reingresa particulas que salieron por el borde superior."""
        self.x[indices] = self.rng.uniform(0, self.width, len(indices))
        self.y[indices] = self.rng.uniform(-80, -2, len(indices))
        self.vx[indices] = self.rng.uniform(-0.35, 0.35, len(indices))
        self.vy[indices] = self.rng.uniform(1.5, 4.0, len(indices))

    def update(self, collision_mask: np.ndarray, scale: Tuple[float, float], dt: float) -> None:
        """Integra gravedad y colisiones usando indexacion vectorizada."""
        sx, sy = scale
        self.vy += np.float32(520.0 * dt)
        self.x += self.vx * np.float32(dt * 60.0)
        self.y += self.vy * np.float32(dt * 60.0)

        px = np.clip((self.x / sx).astype(np.int32), 0, collision_mask.shape[1] - 1)
        py = np.clip((self.y / sy).astype(np.int32), 0, collision_mask.shape[0] - 1)
        active = (self.y >= 0) & (self.y < self.height) & collision_mask[py, px]

        # Un rebote amortiguado produce acumulacion progresiva sobre la silueta.
        self.y[active] -= 3.0
        self.vy[active] *= -0.38
        self.vx[active] *= 0.96

        out = (self.y > self.height + 8) | (self.x < -10) | (self.x > self.width + 10)
        if np.any(out):
            self.reset_particle(np.flatnonzero(out))

    def draw(self, frame: np.ndarray) -> None:
        """Dibuja todas las particulas mediante escritura matricial y dilatacion."""
        visible = (self.y >= 0) & (self.y < self.height)
        px = self.x[visible].astype(np.int32)
        py = self.y[visible].astype(np.int32)
        px = np.clip(px, 0, self.width - 1)
        py = np.clip(py, 0, self.height - 1)

        layer = np.zeros_like(frame)
        # Azul/cian con una ligera variacion por particula.
        layer[py, px, 0] = 180
        layer[py, px, 1] = 220
        layer[py, px, 2] = self.hue[visible]
        layer = cv2.dilate(layer, np.ones((3, 3), np.uint8), iterations=1)
        frame[:] = cv2.add(frame, layer)


class KinectDepth:
    """Adaptador pequeno para libfreenect, aislando el hardware del algoritmo."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index

    def read(self) -> np.ndarray:
        if freenect is None:
            raise RuntimeError("No se pudo importar freenect/libfreenect.")
        try:
            depth, _ = freenect.sync_get_depth(index=self.device_index, format=freenect.DEPTH_MM)
        except Exception as exc:
            raise RuntimeError(f"No se pudo leer la Kinect: {exc}") from exc
        if depth is None or depth.size == 0:
            raise RuntimeError("La Kinect devolvio un mapa de profundidad vacio.")
        return depth.astype(np.uint16, copy=False)


def build_collision_mask(depth: np.ndarray, threshold: int, minimum: int = 250) -> np.ndarray:
    """Crea una mascara booleana: objetos validos entre minimo y umbral mm."""
    valid = (depth >= minimum) & (depth <= threshold)
    # Suaviza pequenos huecos del sensor sin iterar por pixel en Python.
    mask = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask.astype(bool)


def make_background(depth: np.ndarray, mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Convierte profundidad en fondo coloreado y resalta la superficie de colision."""
    normalized = np.clip(depth.astype(np.float32) / 4000.0 * 255.0, 0, 255).astype(np.uint8)
    background = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TWILIGHT)
    background[~mask] = (8, 5, 18)
    background[mask] = (background[mask] * 0.55 + np.array([80, 25, 120]) * 0.45).astype(np.uint8)
    return cv2.resize(background, size, interpolation=cv2.INTER_LINEAR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=1800, help="Numero de particulas (por defecto: 1800)")
    parser.add_argument("--depth-threshold", type=int, default=1800, help="Distancia maxima de colision en mm")
    parser.add_argument("--device", type=int, default=0, help="Indice de Kinect libfreenect")
    parser.add_argument("--width", type=int, default=960, help="Ancho de la ventana")
    parser.add_argument("--height", type=int, default=720, help="Alto de la ventana")
    parser.add_argument("--seed", type=int, default=None, help="Semilla opcional para repetir la demo")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.particles < 1 or args.depth_threshold <= 250:
        print("Los parametros deben tener particulas positivas y threshold > 250 mm.", file=sys.stderr)
        return 2

    sensor = KinectDepth(args.device)
    particles = ParticleSystem.create(args.particles, args.width, args.height, args.seed)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)
    threshold = args.depth_threshold
    previous = time.perf_counter()

    try:
        while True:
            depth = sensor.read()
            mask = build_collision_mask(depth, threshold)
            frame = make_background(depth, mask, (args.width, args.height))
            now = time.perf_counter()
            dt = min(now - previous, 1.0 / 15.0)
            previous = now
            particles.update(mask, (args.width / depth.shape[1], args.height / depth.shape[0]), dt)
            particles.draw(frame)

            cv2.putText(frame, f"Profundidad <= {threshold} mm | Particulas: {args.particles}", (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 240, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Q/Esc: salir | +/-: umbral | R: reiniciar lluvia", (18, args.height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 220), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("+"), ord("=")):
                threshold = min(threshold + 50, 4000)
            elif key in (ord("-"), ord("_")):
                threshold = max(threshold - 50, 300)
            elif key in (ord("r"), ord("R")):
                particles = ParticleSystem.create(args.particles, args.width, args.height)
    except RuntimeError as exc:
        print(f"Error de Kinect: {exc}", file=sys.stderr)
        print("Comprueba el USB, permisos/reglas udev y el controlador libusbK.", file=sys.stderr)
        return 1
    finally:
        if freenect is not None:
            try:
                freenect.sync_stop()
            except Exception:
                pass
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
