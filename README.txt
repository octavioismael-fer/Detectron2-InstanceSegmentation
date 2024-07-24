Recomendacion: Crear un entorno de trabajo para no tener problemas a la hora de ejecutar los scripts.

Antes de correr los respectivos script instalar el archivo "requirements.txt", contiene las librerias necesarias para el correcto funcionamiento de "Detector.py".

Nota: En caso de que "Detectron2" lanze algun tipo de error seguir los siguente pasos para que funcione correctamente.

Nota: Cambiar las rutas del proyecto a las que corresponde en tu entorno de trabajo, de lo contrario el script lanzara errores.

1- Dentro del directorio de trabajo (tener activado su entorno) ingresar:
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

1- Una vez dentro del directorio donde trabajara ingresar:
git clone https://github.com/facebookresearch/detectron2.git

2- Nos movemos hacia al directorio que se acaba de descargar:
cd detectron2

3- Dentro del directorio ingresamo lo siguente (no olvidar el "."):
pip install -e .

Eso bastaria para que funcione.