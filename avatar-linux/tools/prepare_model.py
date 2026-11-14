#!/usr/bin/env python
"""Preprocesador de modelos glTF para el avatar de Kinect.

RobotExpressive.glb combina mallas rigidas (partes del robot como nodos
hijos de los huesos) con mallas skinned (manos). El convertidor gltf2bam de
Panda3D solo vincula al esqueleto las mallas con atributos JOINTS/WEIGHTS,
por lo que este script:

1. Calcula la matriz de bind mundial de cada nodo (composicion TRS).
2. Para cada malla rigida, hornea en sus vertices la transformacion
   M = bind_world(hueso_padre) * local(nodo).
3. Borra la transformacion del nodo (queda identidad) y le asigna un skin
   de un solo joint que apunta al hueso padre dentro del skin principal.
4. Regenera el .glb con los nuevos atributos JOINTS_0/WEIGHTS_0.

El resultado es un glb 100% skinned que gltf2bam puede animar por huesos.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

COMP_FLOAT = 5126
COMP_USHORT = 5123
TYPE_VEC3 = "VEC3"
TYPE_VEC4 = "VEC4"


def compose_mat4(translation, rotation, scale) -> np.ndarray:
    """Matriz 4x4 (column-major, como glTF) a partir de TRS."""
    t = np.asarray(translation if translation is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    r = np.asarray(rotation if rotation is not None else [0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    s = np.asarray(scale if scale is not None else [1.0, 1.0, 1.0], dtype=np.float64)

    x, y, z, w = r
    rot = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot @ np.diag(s)
    m[:3, 3] = t
    return m


def node_local_matrix(node) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    return compose_mat4(node.get("translation"), node.get("rotation"), node.get("scale"))


def compute_world_matrices(gltf) -> dict[int, np.ndarray]:
    nodes = gltf["nodes"]
    world: dict[int, np.ndarray] = {}
    for scene in gltf.get("scenes", [])[:1]:
        def visit(nodeid: int, parent_mat: np.ndarray) -> None:
            local = node_local_matrix(nodes[nodeid])
            mat = parent_mat @ local
            world[nodeid] = mat
            for child in nodes[nodeid].get("children", []):
                visit(child, mat)
        for root_id in scene.get("nodes", []):
            visit(root_id, np.eye(4))
    return world


def read_accessor_bytes(bin_data, gltf, accessor) -> np.ndarray:
    bv = gltf["bufferViews"][accessor["bufferView"]]
    start = bv.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]
    if accessor["type"] == TYPE_VEC3:
        values = 3 * count
    elif accessor["type"] == TYPE_VEC4:
        values = 4 * count
    else:
        values = count
    dtype = np.float32 if accessor["componentType"] == COMP_FLOAT else np.uint16
    return np.frombuffer(bin_data, dtype=dtype, count=values, offset=start)


def write_accessor_bytes(bin_data, gltf, accessor, values: np.ndarray) -> None:
    bv = gltf["bufferViews"][accessor["bufferView"]]
    start = bv.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    raw = values.astype(values.dtype).tobytes()
    bin_data[start:start + len(raw)] = raw


def bake_primitive(bin_data, gltf, primitive, matrix: np.ndarray, parent_world: np.ndarray) -> None:
    """Hornea M = parent_world @ local en POSITION y NORMAL."""
    m = parent_world @ matrix
    rot3 = m[:3, :3]
    normal_mat = np.linalg.inv(rot3).T

    pos_acc = gltf["accessors"][primitive["attributes"]["POSITION"]]
    positions = read_accessor_bytes(bin_data, gltf, pos_acc).astype(np.float64).reshape(-1, 3)
    ones = np.ones((positions.shape[0], 1), dtype=np.float64)
    transformed = (m @ np.hstack([positions, ones]).T).T[:, :3]
    write_accessor_bytes(bin_data, gltf, pos_acc, transformed.astype(np.float32).reshape(-1))

    if "NORMAL" in primitive["attributes"]:
        norm_acc = gltf["accessors"][primitive["attributes"]["NORMAL"]]
        normals = read_accessor_bytes(bin_data, gltf, norm_acc).astype(np.float64).reshape(-1, 3)
        rotated = normals @ normal_mat.T
        rotated /= np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12)
        write_accessor_bytes(bin_data, gltf, norm_acc, rotated.astype(np.float32).reshape(-1))


def append_attribute(bin_data: bytearray, gltf, values: np.ndarray, component_type: int,
                     accessor_type: str) -> int:
    """Anexa un bufferView + accessor y devuelve el indice del accessor."""
    pad = (-len(bin_data)) % 4
    if pad:
        bin_data.extend(b"\x00" * pad)
    byte_offset = len(bin_data)
    raw = values.astype(values.dtype).tobytes()
    bin_data.extend(raw)
    byte_length = len(raw)

    buffer_view = {"buffer": 0, "byteOffset": byte_offset, "byteLength": byte_length}
    gltf["bufferViews"].append(buffer_view)
    accessor = {
        "bufferView": len(gltf["bufferViews"]) - 1,
        "componentType": component_type,
        "count": values.size // (4 if accessor_type == TYPE_VEC4 else 1),
        "type": accessor_type,
    }
    gltf["accessors"].append(accessor)
    return len(gltf["accessors"]) - 1


def main() -> None:
    src = Path(__file__).resolve().parent.parent / "models" / "RobotExpressive.glb"
    dst = Path(__file__).resolve().parent.parent / "models" / "RobotExpressive_rigged.glb"

    raw = src.read_bytes()
    assert struct.unpack("<I", raw[0:4])[0] == GLB_MAGIC, "No es un glb"
    json_len = struct.unpack("<I", raw[12:16])[0]
    gltf = json.loads(raw[20:20 + json_len].decode("utf-8"))
    bin_start = 20 + json_len + (-json_len) % 4
    bin_len = struct.unpack("<I", raw[bin_start:bin_start + 4])[0]
    bin_data = bytearray(raw[bin_start + 8: bin_start + 8 + bin_len])

    nodes = gltf["nodes"]
    world = compute_world_matrices(gltf)
    main_skin = gltf["skins"][0]
    joint_nodes = set(main_skin["joints"])

    parents: dict[int, int] = {}
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            parents[child] = i

    rigged = 0
    for nodeid, node in enumerate(nodes):
        if "mesh" not in node or "skin" in node:
            continue
        jointid = nodeid
        while jointid is not None and jointid not in joint_nodes:
            jointid = parents.get(jointid)
        if jointid is None:
            continue

        local = node_local_matrix(node)
        for primitive in gltf["meshes"][node["mesh"]]["primitives"]:
            if "POSITION" not in primitive["attributes"]:
                continue
            bake_primitive(bin_data, gltf, primitive, local, world[jointid])

            joint_index = main_skin["joints"].index(jointid)
            count = gltf["accessors"][primitive["attributes"]["POSITION"]]["count"]
            joints = np.full((count, 4), joint_index, dtype=np.uint16)
            weights = np.zeros((count, 4), dtype=np.float32)
            weights[:, 0] = 1.0

            acc_j = append_attribute(bin_data, gltf, joints, COMP_USHORT, TYPE_VEC4)
            acc_w = append_attribute(bin_data, gltf, weights, COMP_FLOAT, TYPE_VEC4)
            primitive["attributes"]["JOINTS_0"] = acc_j
            primitive["attributes"]["WEIGHTS_0"] = acc_w

        for key in ("translation", "rotation", "scale", "matrix"):
            node.pop(key, None)
        node["skin"] = 0
        rigged += 1

    # Se conservan los dos skins originales de las manos; el parche del
    # conversor gltf2bam gestiona skins que comparten el mismo nodo raiz.

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad = (-len(json_bytes)) % 4
    json_chunk = json_bytes + b" " * json_pad
    bin_chunk = bytes(bin_data)
    bin_pad = (-len(bin_chunk)) % 4
    bin_chunk += b"\x00" * bin_pad

    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out = struct.pack("<III", GLB_MAGIC, 2, total)
    out += struct.pack("<II", len(json_chunk), CHUNK_JSON) + json_chunk
    out += struct.pack("<II", len(bin_chunk), CHUNK_BIN) + bin_chunk
    dst.write_bytes(out)
    print(f"OK: {rigged} mallas rigidas convertidas a skinned -> {dst.name}")


if __name__ == "__main__":
    main()
