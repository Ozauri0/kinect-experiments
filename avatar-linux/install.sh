#!/usr/bin/env bash
# Instalador de dependencias para "Lluvia Digital Interactiva" (Kinect v1).
# Uso: bash install.sh
# El script pide la contrasena de sudo de forma interactiva y segura.
set -euo pipefail

echo "== [1/5] Paquetes oficiales =="
sudo pacman -S --needed --noconfirm python python-pip base-devel cmake git \
  libusb freeglut glu libxmu

echo "== [2/5] libfreenect (AUR, con fallback a compilacion nativa) =="
HELPER=""
if command -v paru >/dev/null 2>&1; then
  HELPER="paru"
elif command -v yay >/dev/null 2>&1; then
  HELPER="yay"
fi

if [ -n "$HELPER" ] && "$HELPER" -S --needed libfreenect; then
  echo "libfreenect instalado desde AUR."
else
  echo "AUR no disponible o fallo. Compilando libfreenect nativo sin wrapper Python..."
  SRC_DIR="$(mktemp -d)"
  git clone --depth 1 https://github.com/OpenKinect/libfreenect.git "$SRC_DIR"
  cmake -S "$SRC_DIR" -B "$SRC_DIR/build" \
    -DBUILD_PYTHON=OFF -DBUILD_PYTHON3=OFF -DBUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$SRC_DIR/build" -j"$(nproc)"
  sudo cmake --install "$SRC_DIR/build"
  sudo ldconfig
  rm -rf "$SRC_DIR"
fi

echo "== [3/5] Entorno virtual Python =="
if [ ! -d .venv ]; then
  python -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install freenect

echo "== [4/5] Regla udev para acceso sin root =="
sudo sh -c 'printf "%s\n" \
  "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"045e\", ATTR{idProduct}==\"02ae\", MODE=\"0666\"" \
  "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"045e\", ATTR{idProduct}==\"02b0\", MODE=\"0666\"" \
  "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"045e\", ATTR{idProduct}==\"02ad\", MODE=\"0666\"" \
  > /etc/udev/rules.d/51-kinect.rules'
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "== [5/5] Verificacion =="
.venv/bin/python -c "import freenect, cv2, numpy; print('Dependencias OK:', cv2.__version__, numpy.__version__)"
echo
echo "Listo. Conecta la Kinect (si estaba enchufada, desconectala y reconectala)"
echo "y ejecuta: .venv/bin/python app.py"
