import numpy as np
import time
import mss
import matplotlib.pyplot as plt
import random
import pyautogui
import torch

# save screenshot
save = True

def shot(save=False):
    # Adjust values based on screen resolution
    monitor = {'top': 0, 'left': 0, 'width': 2560, 'height': 1440}  
    img = mss.mss().grab(monitor)
    img_np = np.array(img)[..., :3] 
    if(save):
        x = random.randint(1,100)
        plt.imsave(f"../sreenshot{x}.png", img_np)
    return img_np # (1080, 1920, 3) values: [0, 255]


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

# main loop
time.sleep(3)
for i in range(10):
    # Get image of screen
    frame = shot(save=save)
    frame_height, frame_width = frame.shape[:2]
    print(f"screenshot size: {frame_height} {frame_width}")

    # Detection
    # results = np.array([[500,500,300,300,80,0], # x_center, y_center, width, height, class_confidence, class_id
            #    [600,600,300,300,90,0]])
               
    detections = model(frame)
    results = []
    for i in range(len(detections.xywh)):
        detection = detections.xywh[i]
        tensor = torch.tensor(detection)
        tensor_cpu = tensor.cpu()
        results.append(tensor_cpu)
    results = torch.cat(results, dim=0)
    results = np.array(results)
    # inference
    

    found = False
    for box_id, detection in enumerate(results):
        print(detection)
        # confidence = detection[4]
        # class_id = detection[5]
        # if confidence > 0.7 and class_id == 0:
        #     found = True
        #     move_to_x, move_to_y = detection[0], detection[1]
        #     break

    if found == True: # if found
        # move to first detection center (coords for other will change after crosshair move)
        # may optimalize to move to the closest

        # move mouse and shoot
        mouse_scale = 1 # 1.7
        x = int(move_to_x * mouse_scale)
        y = int(move_to_y * mouse_scale)
        print(f"moving mouse to: {x}, {y}")
        time.sleep(0.5)
        pyautogui.moveTo(x,y)
        time.sleep(0.1)
        pyautogui.click() 
    else: 
        print("no person found")
    