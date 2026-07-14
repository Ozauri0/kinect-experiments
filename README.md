# Kinect Xbox 360 Experiments

Experimentos con Kinect Xbox 360 (v1) usando Python, OpenNI2 y OpenCV.

## Requisitos

- **Kinect Xbox 360 v1** conectada (USB + alimentación AC)
- **Microsoft Kinect SDK 1.8** instalado (provee el driver)
- **Python 3.8+** (x64)
- Dependencias: `pip install -r requirements.txt`

## Estructura

```
├── check_device.py       # Diagnóstico rápido (sin ventanas)
├── test_kinect.py        # RGB + profundidad en tiempo real (OpenNI2)
├── test_freenect.py      # Alternativa con libfreenect (sin OpenNI2)
├── requirements.txt      # Dependencias Python
├── OpenNI2.dll           # OpenNI2 runtime (x64)
└── OpenNI2/
    └── Drivers/          # Drivers bridge para OpenNI2
        ├── Kinect.dll    # Bridge al MS Kinect SDK
        ├── PS1080.dll    # Driver directo PrimeSense
        └── PS1080.ini
```

## Uso rápido

```bash
# Verificar que la Kinect es detectada
python check_device.py

# Visualizar RGB + profundidad
python test_kinect.py
```

## Instalación de OpenNI2

Si necesitas instalar OpenNI2 manualmente:

1. Descarga [OpenNI 2.2 x64](https://structure.io/openni)
2. Copia `OpenNI2.dll` y la carpeta `OpenNI2/Drivers/` junto a los scripts

Este repositorio ya incluye estos archivos para conveniencia.

## Solución de problemas

| Problema | Solución |
|---|---|
| Kinect no detectada | Verifica USB + cable de alimentación AC conectado |
| `OpenNI2.dll not found` | Asegúrate de que `OpenNI2.dll` está junto a los scripts |
| `No module named 'openni'` | `pip install openni` |
| Conflicto de drivers | Solo puede haber un driver activo (MS SDK o libusbK) |

## Alternativa: freenect

Si prefieres no usar OpenNI2:

1. Instala el driver libusbK con [Zadig](https://zadig.akeo.ie/) (selecciona "Xbox NUI Motor")
2. `pip install freenect numpy opencv-python`
3. `python test_freenect.py`

**Nota:** freenect y el MS Kinect SDK usan drivers distintos. Son mutuamente excluyentes.

## Licencia

MIT
