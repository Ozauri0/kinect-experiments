"""
Alternativa con freenect/libfreenect - NO requiere OpenNI2 ni MS SDK.

Requisitos:
  - Driver libusbK/WinUSB instalado con Zadig (https://zadig.akeo.ie/)
    Seleccionar "Xbox NUI Motor" o "Xbox NUI Camera"
  - pip install freenect numpy opencv-python

Usa ESC para salir.
"""

import sys
from pathlib import Path


def main():
    try:
        import freenect
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"[ERROR] Falta dependencia: {e}")
        print("pip install freenect numpy opencv-python")
        return 1

    print("Kinect Xbox 360 - freenect Viewer")
    print("ESC para salir | Ctrl+C para forzar cierre")
    print("-" * 50)

    ctx = freenect.init()
    if ctx is None:
        print("[ERROR] No se pudo inicializar freenect")
        return 1

    num = freenect.num_devices(ctx)
    print(f"Dispositivos: {num}")

    if num == 0:
        print("[ERROR] Kinect no detectada")
        print("Verifica: USB + alimentación + driver Zadig")
        freenect.shutdown(ctx)
        return 1

    dev = freenect.open_device(ctx, 0)
    if dev is None:
        print("[ERROR] No se pudo abrir el dispositivo")
        freenect.shutdown(ctx)
        return 1

    freenect.set_video_mode(dev, freenect.VIDEO_RGB)
    freenect.set_depth_mode(dev, freenect.DEPTH_11BIT)
    freenect.start_video(dev)
    freenect.start_depth(dev)

    print("RGB + Depth activos")
    print("-" * 50)

    try:
        while True:
            rgb, _ = freenect.sync_get_video()
            depth, _ = freenect.sync_get_depth()

            if rgb is not None:
                cv2.imshow("Kinect RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            if depth is not None:
                vis = (depth.astype(np.float32) / 2047 * 255).astype(np.uint8)
                cv2.imshow("Kinect Depth", cv2.applyColorMap(vis, cv2.COLORMAP_JET))

            if cv2.waitKey(10) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        freenect.stop_video(dev)
        freenect.stop_depth(dev)
        freenect.close_device(dev)
        freenect.shutdown(ctx)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
