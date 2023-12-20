from langchain.tools import BaseTool
import cv2
import ultralytics
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import os
import stat
# directory_path = 'C:\\Users\\Dheeraj\\Downloads\\aree-main\\'

# Check current directory permissions
# current_permissions = os.stat(directory_path)
# print(f"Current Permissions: {current_permissions.st_mode}")

# Add write permissions for the user
# os.chmod(directory_path, current_permissions.st_mode | stat.S_IWUSR)

class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"  # cuda

        # yolo_model = YOLO()
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class ObjectDetectionTool(BaseTool):
    name = "Object detector"
    description = "Use this tool when given the path to an image that you would like to detect objects. " \
                  "It will return a list of all detected objects. Each element in the list in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score."

    def _run(self, img_path):
        print(img_path)
        current_permissions = os.stat(img_path)
        print(f"Current Permissions: {current_permissions.st_mode}")
        os.chmod(img_path, current_permissions.st_mode | stat.S_IWUSR)
        image = Image.open(img_path).convert('RGB')

        # processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        # model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        # inputs = processor(images=image, return_tensors="pt")
        # outputs = model(**inputs)
        cv2.imshow(image)
        model= YOLO("last_capsv8seg.pt")
        output = model(source=image)
    # detection_img = output.render()[0]
        detections = ""
        for result in output:
             if result is not None:
                im_res=result.plot()
                data = result.boxes.data.cpu().tolist()
                h, w = result.orig_shape #if normalize else (1, 1)
                for i, row in enumerate(data):
                    box = {'x1': row[0] / w, 'y1': row[1] / h, 'x2': row[2] / w, 'y2': row[3] / h}
                    conf = row[4]
                    id = int(row[5])
                    name = result.names[id]
                    detections += '[{}, {}, {}, {}]'.format(float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))
                    detections += ' {}'.format(name)
                    detections += ' {}\n'.format(float(conf))

           
                    x = int((row[0] + row[2])/2)
                    y = int((row[1] + row[3])/2)
                    
                    
                    coordinate_text = "(" + str(round(x)) + "," + str(round(y)) + ")"
                    print(coordinate_text)
                    print('\n')
        return detections

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        # target_sizes = torch.tensor([image.size[::-1]])
        # results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        
        
                

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
