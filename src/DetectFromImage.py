import cv2
import os
import sys
import torch
import time
start_time = time.time()
sys.path.append('D:\FPT\ky5\AIP391\My Project\data')
from src.detector import Detector
sys.path.append('D:\FPT\ky5\AIP391\My Project\data\yolov7')
from models.experimental import attempt_load

classes_to_filter = None
model = attempt_load(r"D:\FPT\ky5\AIP391\My Project\data\yolov7\runs\train\last.pt", map_location=torch.device('cuda:0'))  # load FP32 model
print("--- %s seconds ---" % (time.time() - start_time))

opt  = {
    "model" : model,
    "weights": r"D:\FPT\ky5\AIP391\My Project\data\yolov7\runs\train\last.pt", # Path to weights file default weights are for nano model
    "yaml"   : r"D:\FPT\ky5\AIP391\My Projecqt\data\yolov7\data\mydataset.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.1, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : '1',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter  # list of classes to filter or None

}
if __name__ == "__main__":
    img_url = r'D:\FPT\ky5\AIP391\My Project\data\test_img\G0010124.jpg'
    img = cv2.imread(img_url)
    detector = Detector(img, opt)
    img,number_potholes = detector.detect_plotbox()
    
    cv2.imwrite('detected.jpg',img)
    print("--- %s seconds ---" % (time.time() - start_time))
    # cv2.imshow('hiiii',img)
    # cv2.waitKey(0) 