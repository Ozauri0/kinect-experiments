"""
Diagnóstico rápido: detecta la Kinect y muestra información de sensores.
No abre ventanas. Ejecutar una sola vez para verificar que todo funciona.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def main():
    from openni import openni2

    print("Kinect Xbox 360 - Diagnóstico")
    print("-" * 40)
    print(f"Python: {sys.version_info.major}.{sys.version_info.minor} ({'x64' if sys.maxsize > 2**32 else 'x86'})")

    try:
        openni2.initialize(str(SCRIPT_DIR))
    except Exception as e:
        print(f"[FAIL] OpenNI2: {e}")
        return 1

    print(f"[OK]  OpenNI2 v{openni2.get_version().major}.{openni2.get_version().minor}.{openni2.get_version().maintenance}")

    try:
        device = openni2.Device.open_any()
    except Exception:
        print("[FAIL] No se detectó Kinect. Verifica USB y alimentación.")
        openni2.unload()
        return 1

    info = device.get_device_info()
    print(f"[OK]  Dispositivo: {info.name.decode()} ({info.vendor.decode()})")

    available = []
    if device.has_sensor(openni2.SENSOR_COLOR):
        available.append("RGB")
    if device.has_sensor(openni2.SENSOR_DEPTH):
        available.append("Profundidad")
    if device.has_sensor(openni2.SENSOR_IR):
        available.append("IR")

    print(f"[OK]  Sensores: {', '.join(available)}")
    print(f"[OK]  Resolución esperada: 640x480 @ 30fps (RGB y profundidad)")

    device.close()
    openni2.unload()
    print("-" * 40)
    print("Kinect lista. Ejecuta test_kinect.py para ver los streams.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
