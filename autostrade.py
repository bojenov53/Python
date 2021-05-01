'''
from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd()

dt = ObjectDetection()
dt.setModelTypeAsRetinaNet()
dt.setModelPath(os.path.join(exec_path, "resnet50_coco_best_v2.1.0.h5"))
dt.loadModel()

list = dt.detectCustomObjectsFromImage(
    input_image=os.path.join(exec_path, "Firenze.jpg"), 
    output_image_path=os.path.join(exec_path, "new2.jpg"),
    minimum_percentage_probability=50,
    display_percentage_probability=False,
    display_object_name=True
    )
'''



from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

dt = VideoObjectDetection()
dt.setModelTypeAsYOLOv3()
dt.setModelPath( os.path.join(execution_path , "yolo.h5"))
dt.loadModel()

video_path = dt.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "autostrade_video.mp4"),
    output_file_path=os.path.join(execution_path, "traffic1"),
    frames_per_second=20, log_progress=True)


print(video_path)