import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

input_folder = 'screenshots'
output_folder = 'results'

class_names = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"): 
        # Load image
        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)
        img_np = np.array(img)
        
        # Detection
        detections = model(img_np)
        results = []
        for i in range(len(detections.xywh)):
            detection = detections.xywh[i]
            tensor = torch.tensor(detection)
            tensor_cpu = tensor.cpu()
            results.append(tensor_cpu)
        results = torch.cat(results, dim=0)
        results = np.array(results)

        for box in results:
            x, y, w, h = map(int, box[:4])
            conf, class_id = box[4], int(box[5])
            img_np = cv2.rectangle(img_np, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)
            label = f"{class_names[class_id]}: {conf:.2f}" 
            img_np = cv2.putText(img_np, label, (x-w//2, y-h//2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img_np)

print("Done")
