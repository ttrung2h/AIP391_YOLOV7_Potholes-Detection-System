import cv2
import os
import sys
import torch
import time
import argparse
start_time = time.time()
from detector import Detector
sys.path.append('yolov7')
from models.experimental import attempt_load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='weight/best.pt', help='path to weight file')
    parser.add_argument('--image', type=str, default=None, help='image url')
    parser.add_argument('--resultimage', type=str, default="result_image", help='folder contain result image')
    config = parser.parse_args()
    
    classes_to_filter = None
    weight_path = config.weight
    model = attempt_load(weight_path, map_location=torch.device('cuda:0'))  # load FP32 model
    print("--- %s seconds ---" % (time.time() - start_time))

    opt  = {
        "model" : model,
        "yaml"   : r"yolov7\data\mydataset.yaml",
        "img-size": 640, # default image size
        "conf-thres": 0.1, # confidence threshold for inference.
        "iou-thres" : 0.45, # NMS IoU threshold for inference.
        "device" : '1',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
        "classes" : classes_to_filter  # list of classes to filter or None

    }
    
    
    img_url = config.image
    img = cv2.imread(img_url)
    detector = Detector(img, opt)
    img,number_potholes = detector.detect_plotbox()
    
    os.makedirs(config.resultimage, exist_ok=True)
    cv2.imwrite(config.resultimage+'\detected.jpg',img)