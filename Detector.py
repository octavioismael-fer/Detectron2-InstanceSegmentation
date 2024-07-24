from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
import os
import cv2

setup_logger()


class Detector:

    """
    Esta clase es la encargada de configurar y utilizar un modelo de segmentación de instancia 
    basado en Detectron2. Proporciona métodos para preprocesar imágenes y procesar videos, aplicando 
    segmentación de instancias y enmascarando ciertas áreas de los fotogramas.
    """

    def __init__(self, model_path, num_classes):

        """
        Esta función es la encargada de inicializar la configuración del modelo, cargar 
        los pesos del modelo entrenado y configurar el predictor de Detectron2.
        No retorna ningún valor.
        

        model_path: Ruta donde se encuentra el modelo entrenado.
        num_classes: Número de clases que el modelo puede detectar.
        """

        self.cfg = get_cfg()

        # Cargar la configuracion del modelo y el modelo preentrenado
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
        self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

        
        self.class_names = ["publicidad", "cancha", "grada"] 

    def preprocess_image(self, image):

        """
        Esta función es la encargada de realizar el preprocesamiento necesario en una imagen (por ejemplo, redimensionarla). 
        Retorna la imagen preprocesada.
        
        image: Imagen a preprocesar.
        """

        return cv2.resize(image, (1200, 1000))

    def onVideo(self, video_path, output_video_path, output_frame_dir):

        """
        Esta función es la encargada de procesar el video utilizando el modelo Detectron2.
        Aplica segmentación por instancias en cada frame del video, enmascara áreas clasificadas como "grada",
        guarda el video procesado y los fotogramas individuales, y muestra una vista previa.
        No retorna ningún valor.
        
        video_path: Ruta al video de entrada.
        output_video_path: Ruta para guardar el video procesado.
        output_frame_dir: Directorio para guardar los fotogramas procesados.
        """

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Crear la carpeta de salida si no existe
        os.makedirs(output_frame_dir, exist_ok=True)

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.preprocess_image(frame)
            predictions = self.predictor(frame)

            # Verificar que las predicciones contengan instancias detectadas
            if "instances" in predictions:
                instances = predictions["instances"]
                for i in range(len(instances)):
                    pred_class = instances.pred_classes[i].item()
                    class_name = self.class_names[pred_class]
                    score = instances.scores[i].item()

                    # Aplicar máscara negra a las áreas clasificadas como "grada"
                    if class_name == "grada":
                        mask = instances.pred_masks[i].numpy()
                        frame[mask] = [0, 0, 0]

            # Guardar el fotograma procesado como una imagen
            frame_filename = os.path.join(output_frame_dir, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

            # Mostrar el fotograma procesado
            cv2.imshow("Processed Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()


video_path = r'C:\Users\PC\Documents\Proyectos\newchallenge\videomuestras\video9.mp4'
output_video_path = r'C:\Users\PC\Documents\Proyectos\newchallenge\videomuestras\output_video.mp4'
output_frame_dir = r'C:\Users\PC\Documents\Proyectos\newchallenge\videomuestras\frames'

detector = Detector(model_path=r'C:\Users\PC\Documents\Proyectos\newchallenge\output\model_final.pth', num_classes=3)
detector.onVideo(video_path, output_video_path, output_frame_dir)