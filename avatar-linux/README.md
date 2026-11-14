# Lluvia Digital Interactiva

Demostracion en Python 3 para Kinect v1 (Xbox 360). La camara de profundidad
se convierte en una superficie de colision: las particulas caen, rebotan y se
acumulan visualmente sobre la silueta u objetos cercanos.

## Estructura de la demo

1. `ParticleSystem` mantiene posicion, velocidad, radio y color de todas las
   particulas en arrays NumPy.
2. `KinectDepth` encapsula la llamada de `libfreenect` y devuelve el mapa de
   profundidad en milimetros.
3. `build_collision_mask` aplica un umbral de distancia y una operacion
   morfologica para cerrar huecos del sensor.
4. El bucle principal actualiza fisica, colorea profundidad y muestra el
   resultado con OpenCV.

## Arch Linux / CachyOS

Se necesita una Kinect v1 con su adaptador de alimentacion y datos USB. La
biblioteca no esta en los repositorios oficiales de Arch; actualmente se
instala desde AUR. No existe un paquete AUR llamado `python-freenect`: el
binding Python se instala desde PyPI despues de instalar la biblioteca nativa.

Forma recomendada: ejecuta el instalador incluido y teclea la contrasena de
`sudo` cuando el sistema la solicite.

```bash
bash install.sh
```

Si prefieres hacerlo manualmente, los pasos son:

```bash
sudo pacman -Syu
sudo pacman -S --needed python python-pip python-numpy python-opencv base-devel cmake libusb
paru -S libfreenect
# Alternativa equivalente:
# yay -S libfreenect
```

Si `paru -S libfreenect` falla al compilar el wrapper con la version de
Python instalada (frecuente con Python 3.14), `install.sh` compila solo la
biblioteca nativa sin wrapper Python y el binding se instala despues desde
PyPI dentro del entorno virtual.

Para permitir acceso sin root, localizar el identificador USB con
`lsusb` y crear una regla, sustituyendo `045e` y `02b0` si la salida de
`lsusb` muestra valores diferentes:

```bash
sudo pacman -S --needed usbutils
lsusb
sudo install -d /etc/udev/rules.d
sudo sh -c 'printf "%s\n" \
  "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"045e\", ATTR{idProduct}==\"02ae\", MODE=\"0666\"" \
  "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"045e\", ATTR{idProduct}==\"02b0\", MODE=\"0666\"" \
  "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"045e\", ATTR{idProduct}==\"02ad\", MODE=\"0666\"" \
  > /etc/udev/rules.d/51-kinect.rules'
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Desconecta y conecta la Kinect despues de recargar las reglas. Prueba primero
la importacion y luego ejecuta:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install freenect
python -c "import freenect, cv2, numpy; print('Dependencias OK')"
python app.py
```

Si la escena esta lejos, aumenta el rango con `python app.py
--depth-threshold 2200`. Las teclas `+` y `-` cambian el umbral durante la
ejecucion; `R` reinicia las particulas y `Q` o `Esc` sale.

## Windows

1. Instala Python 3 de 64 bits y marca **Add Python to PATH**.
2. Conecta la Kinect directamente al equipo, no a un hub sin alimentacion.
3. Instala [Zadig](https://zadig.akeo.ie/), activa `Options > List All Devices`,
   selecciona el dispositivo USB de Kinect y reemplaza su controlador por
   **libusbK**. No reemplaces el controlador de teclado, raton o webcam.
4. En PowerShell, crea un entorno e instala los paquetes Python:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install freenect
```

El ultimo paquete proporciona el modulo Python cuando existe una wheel
compatible. Si PyPI no ofrece wheel para la version de Python instalada,
instala una distribucion de `libfreenect` que incluya `freenect.dll` y sus
bindings, agrega la carpeta de DLL al `PATH`, y compila el binding con CMake y
Visual Studio. El modulo debe poder responder a `import freenect`; la
aplicacion no usa APIs exclusivas de Windows.

Prueba y ejecuta desde la carpeta del proyecto:

```powershell
python -c "import freenect, cv2, numpy; print('Dependencias OK')"
python app.py
```

Si Windows muestra un error de DLL, normalmente el controlador Zadig no es
`libusbK`, falta la DLL en `PATH`, o se instalo un binding de 32 bits junto a
Python de 64 bits.

## Explicacion academica

En cada frame, Kinect entrega una matriz `D` de profundidad. La mascara de
colision se calcula matricialmente como `M = (D >= d_min) AND (D <= d_max)`;
los valores cero o invalidos se descartan. `d_max` es el umbral interactivo:
solo las superficies suficientemente cercanas se consideran obstaculos. Una
operacion morfologica de cierre sobre `M` rellena pequenos huecos producidos
por ruido o por puntos sin retorno del sensor.

Cada particula tiene posicion `(x, y)` y velocidad `(vx, vy)`, almacenadas en
vectores NumPy. La gravedad actualiza todas las velocidades y posiciones en
una sola operacion. Despues, las coordenadas se convierten de pantalla a la
rejilla 640x480 mediante escalado e indexacion avanzada: `M[y_pixel, x_pixel]`
devuelve simultaneamente la colision de todas las particulas, sin recorrer la
imagen con un bucle Python.

Cuando una particula colisiona, se desplaza ligeramente fuera de la superficie
y su velocidad vertical cambia de signo con amortiguacion. La perdida de
energia y la friccion horizontal hacen que rebote varias veces y termine
formando una acumulacion visual. El fondo usa el mismo mapa `D` coloreado por
OpenCV, de modo que la silueta y la superficie que reciben la lluvia siguen
siendo visibles mientras la mascara cambia en tiempo real.

## Avatar 3D con seguimiento de articulaciones (avatar.py)

Segunda demo: un modelo rigged (RobotExpressive, CC0) imita en tiempo real
los movimientos de la persona frente a la Kinect. La Kinect v1 solo entrega
RGB y profundidad, asi que el esqueleto se estima con MediaPipe Pose y la
profundidad de la Kinect situa a la persona en la sala.

```bash
.venv/bin/python avatar.py                 # espejo activado por defecto
.venv/bin/python avatar.py --no-mirror     # el avatar replica tu lado real
.venv/bin/python avatar.py --tilt 0        # endereza el motor de la Kinect al arrancar
.venv/bin/python avatar.py --no-translate  # avatar fijo en el centro
```

Teclas: `M` espejo, `C` mostrar/ocultar la vista de camara (con el esqueleto
2D detectado), `T` pose de prueba sin Kinect, `R` recentrar la posicion en la
sala, `Esc` salir.

Para que funcione bien:

- **Luz**: MediaPipe trabaja sobre la imagen RGB. Con solo una tira LED la
  imagen es casi negra; el programa la realza (CLAHE + gamma) pero una luz
  normal en la habitacion marca la diferencia. Si la escena es muy oscura, la
  barra de estado lo avisa.
- **Encuadre**: colocate a 2-3 m con el cuerpo entero visible. Si solo se ven
  la cabeza y los hombros, el avatar mueve brazos y torso y deja las piernas
  en reposo. Si la Kinect apunta al techo, usa `--tilt 0` (o un valor
  negativo para bajarla).
- Se usa `models/pose_landmarker_full.task` si existe (mas preciso) y si no
  `pose_landmarker_lite.task`.

### Preparacion del modelo

El glb del robot tiene dos skins que comparten raiz; el convertidor
`gltf2bam` necesita un parche (idempotente) para aceptarlo. Las piezas
rigidas del robot quedan colgadas de sus huesos y siguen al esqueleto sin mas
preparacion:

```bash
.venv/bin/python tools/patch_gltf_converter.py
.venv/bin/gltf2bam models/RobotExpressive.glb models/RobotExpressive.bam
```

El `.glb` trae cada brazo como **un solo palo con una bolita** pegado al
hueso del brazo superior, de modo que los huesos del codo y la muneca no mueven
nada visible. `avatar.py` lo corrige al cargar (`build_two_segment_arms`):
oculta esa malla y reutiliza el palo+bolita dos veces, estirado a la longitud
real de cada hueso, uno ligado a `UpperArm` (hombro-codo, la bolita hace de
codo) y otro a `LowerArm` (codo-muneca, la bolita hace de mano). No hace falta
tocar el modelo; los segmentos siguen a sus huesos con `add_net_transform`.

`tools/prepare_model.py` y `models/RobotExpressive_rigged.*` ya no se usan:
ese paso horneaba la transformacion de bind en los vertices y gltf2bam la
volvia a aplicar, asi que el robot se veia desmontado.

### Explicacion academica del retargeting

1. **Pose 3D**. MediaPipe entrega, ademas de los 33 landmarks en imagen,
   landmarks "de mundo" en metros centrados en la cadera. Se pasan a un
   espacio de persona `x` = derecha de la imagen, `y` = arriba, `z` = hacia la
   camara. Un filtro One Euro (suavizado adaptativo a la velocidad, con
   `min_cutoff=2.5` y `beta=4.0`) elimina el temblor sin anadir retardo en
   los movimientos rapidos. Si el avatar se ve nervioso, baja `min_cutoff`;
   si se ve con retardo, subelo.
2. **Profundidad Kinect**. Con la profundidad *registrada* (alineada al RGB)
   se toma la mediana en el torso y se aplica el modelo pinhole
   `X = (u - cx) * Z / fx` para desplazar el avatar lateralmente y en
   profundidad.
3. **Espejo**. Reflejar `x` invierte la quiralidad del esqueleto, asi que
   tambien se intercambian las etiquetas izquierda/derecha; el resultado
   vuelve a ser una anatomia valida.
4. **Calibracion del rig**. Se recorre la jerarquia real de huesos y se
   calcula la cinematica directa del bind pose (`R_mundo = R_padre * R_local`,
   `t_mundo = t_padre + R_padre * t_local`). Con la cabeza, la cadera y los
   hombros se construye la base `C = [izquierda, arriba, delante]` que lleva
   el espacio de persona al del esqueleto.
5. **Tronco y cabeza**. Se construyen bases ortonormales con la linea de
   hombros/caderas/orejas y el eje vertical; la rotacion objetivo es
   `R = F_objetivo * F_bind^T`. El abdomen y el cuello reciben la mitad
   (slerp) para que la flexion se reparta.
6. **Extremidades**. Recorriendo de padres a hijos, cada hueso parte de su
   orientacion heredada y gira lo minimo (formula de Rodrigues) hasta apuntar
   a la direccion medida. La rotacion local es `R_local = R_padre^T * R_mundo`.
   Si una articulacion tiene visibilidad baja, el hueso vuelve suavemente al
   reposo en lugar de seguir datos inventados.
7. **Brazo en dos segmentos**. Para que el codo se vea doblar, el modelo
   se amplia al cargar: cada brazo pasa de un palo rigido a dos segmentos
   (hombro-codo y codo-muneca) estirados a la longitud de sus huesos. El eje
   del palo se obtiene por PCA y la bolita se separa por el mayor hueco en su
   coordenada longitudinal; cada vertice se reproyecta como
   `P = inicio + R * (t' * eje + radial)` con `R` la rotacion minima del eje
   original a la direccion del hueso.
8. **Pies y suelo**. En este rig los pies cuelgan de la raiz, asi que se
   recolocan al final de cada espinilla y la raiz se desplaza para que el pie
   mas bajo toque el suelo (permite agacharse y levantar una pierna).

El retargeting no depende del modelo: para otro humanoide basta con cambiar
los nombres de huesos del diccionario `RIG` en `avatar.py`.

### Archivos del avatar

- `avatar.py`: aplicacion completa (captura, pose, filtrado, retargeting, escena).
- `avatar_old.py.bak`: version anterior, conservada solo como referencia.
- `models/RobotExpressive.bam`: modelo convertido (usado por defecto).
- `models/pose_landmarker_full.task` / `pose_landmarker_lite.task`: modelos de
  MediaPipe; se prefiere `full` si existe.

## Limitaciones conocidas

- `libfreenect` entrega la profundidad sin un esqueleto ni segmentacion
  semantica; el umbral considera cercanos tanto al usuario como a objetos.
- El valor de milimetros depende del modo y de la calibracion del sensor. El
  control del umbral existe para ajustar la escena real.
- La salida esperada es aproximadamente 30 FPS, pero USB, CPU y numero de
  particulas influyen. Para equipos modestos usa `--particles 800`.
- `avatar.py`: no se capturan dedos ni el giro del antebrazo, y la pose
  funciona mejor con la persona de frente y bien iluminada. Si la persona
  se da la vuelta, MediaPipe puede confundir izquierda y derecha.
