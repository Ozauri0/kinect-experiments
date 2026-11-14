#!/usr/bin/env python
"""Aplica el parche al conversor gltf2bam (panda3d-gltf).

El convertidor instalado asume que cada skin tiene un nodo raiz de esqueleto
unico. RobotExpressive.glb usa dos skins que comparten el mismo nodo raiz
(skeleton: 2), lo que produce KeyError en la conversion. Este parche permite
que varios skins compartan un unico Character.

Uso (una sola vez por entorno virtual):
    .venv/bin/python tools/patch_gltf_converter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

EDITS = [
    (
        """            if nodeid in self.skeletons:
                skinid = self.skeletons[nodeid]
                charinfo = CharInfo(node_name)
                charinfo.character.set_transform(get_node_transform(gltf_node))
                self.build_character(charinfo, nodeid, gltf_data, recurse=True)
                self.characters[skinid] = charinfo""",
        """            if nodeid in self.skeletons:
                skinids = self.skeletons[nodeid]
                charinfo = CharInfo(node_name)
                charinfo.character.set_transform(get_node_transform(gltf_node))
                self.build_character(charinfo, nodeid, gltf_data, recurse=True)
                for skinid in skinids:
                    self.characters[skinid] = charinfo""",
    ),
    (
        """            skinid = self.skeletons.get(nodeid, None)
            charinfo = self.characters.get(skinid, None)""",
        """            skinid = self.skeletons.get(nodeid, None)
            if isinstance(skinid, list):
                skinid = skinid[0]
            charinfo = self.characters.get(skinid, None)""",
    ),
    (
        """        self.skeletons[root_nodeid] = skinid""",
        """        self.skeletons.setdefault(root_nodeid, []).append(skinid)""",
    ),
    (
        """        if nodeid in self.skeletons:
            skinid = self.skeletons[nodeid]
            gltf_skin = gltf_data['skins'][skinid]""",
        """        if nodeid in self.skeletons:
            skinids = self.skeletons[nodeid]
            skinid = skinids[0] if isinstance(skinids, list) else skinids
            gltf_skin = gltf_data['skins'][skinid]""",
    ),
]


def main() -> int:
    import gltf

    converter = Path(gltf.__file__).resolve().parent / "_converter.py"
    text = converter.read_text(encoding="utf-8")

    applied = 0
    for old, new in EDITS:
        if old in text:
            if new not in text:
                text = text.replace(old, new, 1)
                applied += 1
        else:
            print(f"Patron no encontrado (quizas ya aplicado o version distinta): {old[:60]!r}...")

    converter.write_text(text, encoding="utf-8")
    print(f"Parche aplicado: {applied} ediciones en {converter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
