"""
Kinect Xbox 360 v1 - Visualizador RGB + Profundidad en tiempo real.

Requisitos:
  - Kinect SDK 1.8 instalado (driver)
  - OpenNI2.dll y OpenNI2/Drivers/ junto a este script
  - pip install numpy opencv-python openni

Usa ESC para salir.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def main():
    import numpy as np
    import cv2
    from openni import openni2

    print("Kinect Xbox 360 - RGB + Depth Viewer")
    print("ESC para salir | Ctrl+C para forzar cierre")
    print("-" * 50)

    try:
        openni2.initialize(str(SCRIPT_DIR))
    except Exception as e:
        print(f"[ERROR] OpenNI2: {e}")
        return 1

    print(f"OpenNI2 v{openni2.get_version().major}.{openni2.get_version().minor}")

    try:
        device = openni2.Device.open_any()
    except Exception:
        print("[ERROR] Kinect no encontrada")
        openni2.unload()
        return 1

    info = device.get_device_info()
    print(f"Dispositivo: {info.name.decode()}")

    color_stream = None
    depth_stream = None

    if device.has_sensor(openni2.SENSOR_COLOR):
        color_stream = device.create_color_stream()
        color_stream.start()
        m = color_stream.get_video_mode()
        print(f"RGB:  {m.resolutionX}x{m.resolutionY} @ {m.fps}fps")

    if device.has_sensor(openni2.SENSOR_DEPTH):
        depth_stream = device.create_depth_stream()
        depth_stream.start()
        m = depth_stream.get_video_mode()
        print(f"Depth: {m.resolutionX}x{m.resolutionY} @ {m.fps}fps")

    if not color_stream and not depth_stream:
        print("[ERROR] Sin streams disponibles")
        device.close()
        openni2.unload()
        return 1

    print("-" * 50)

    try:
        while True:
            if color_stream:
                frame = color_stream.read_frame()
                data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8)
                data = data.reshape(frame.height, frame.width, 3)
                cv2.imshow("Kinect RGB", cv2.cvtColor(data, cv2.COLOR_RGB2BGR))

            if depth_stream:
                frame = depth_stream.read_frame()
                data = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16)
                data = data.reshape(frame.height, frame.width)
                vis = (data.astype(np.float32) / 2047 * 255).astype(np.uint8)
                cv2.imshow("Kinect Profundidad", cv2.applyColorMap(vis, cv2.COLORMAP_JET))

            if cv2.waitKey(10) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if color_stream:
            color_stream.stop()
        if depth_stream:
            depth_stream.stop()
        device.close()
        openni2.unload()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
