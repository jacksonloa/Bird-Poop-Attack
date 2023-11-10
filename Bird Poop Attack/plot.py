import numpy as np
import cv2 as cv
import easyocr
from util import img_process, crop_lp, character_recognize, lp_recognize
import os
from ultralytics import YOLO
from PIL import Image

img_folder = 'data/img'
label_folder = 'data/label'
output_folder = 'save_pos'  # 指定的輸出資料夾

# Load a model
model = YOLO('model/aolp-c/weights/best.pt')  # 預訓練的YOLOv8n模型

# 創建輸出資料夾
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(img_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # 讀取圖片
        img_path = os.path.join(img_folder, filename)
        label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + '.txt')
        if os.path.exists(label_path):
            # 讀取圖片
            img = cv.imread(img_path)
            crop_img = crop_lp(img)
            cv.namedWindow('window', cv.WINDOW_NORMAL)
            cv.resizeWindow('window', 800, 600)
            cv.imshow('window', crop_img)
            cv.waitKey(0)
            cv.destroyAllWindows()

            # Run batched inference on a list of images
            results = model(crop_img)  # 返回一個Results對象列表
            
            # Process results list
            h, w, _ = img.shape
            output_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.txt')
            x1_x2_list = []
            with open(output_file_path, 'w') as f:
                for result in results:
                    boxes = result.boxes.xyxy  # bbox輸出的Boxes對象
                    for idx in range(len(boxes)):
                        box = boxes[idx].tolist()
                        r_box = [round(num, 0) for num in box]

                        x1 = (int)(r_box[0])
                        y1 = (int)(r_box[1])
                        x2 = (int)(r_box[2])
                        y2 = (int)(r_box[3])

                        # 收集x1和x2值
                        x1_x2_list.append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})

            # 對x1和x2進行排序
            x1_x2_list.sort(key=lambda item: item['x1'])

            # 將排序後的x1和x2值一起寫入文件
            with open(output_file_path, 'w') as f:
                for item in x1_x2_list:
                    f.write(f"{item['x1']} {item['x2']}\n")