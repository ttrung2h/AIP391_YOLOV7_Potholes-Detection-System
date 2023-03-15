import os
import sys
sys.path.append(r'D:\FPT\ky5\AIP391\My Project\Project\yolov7')

import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from utils.general import check_img_size, non_max_suppression,scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device

# set_logging()
class Detector:
    def __init__(self,img_from_source,opt ) -> None:
        self.img_from_source = img_from_source
        self.opt = opt

    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height6, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        # length = np.sqrt()
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def detect_plotbox(self):
        with torch.no_grad():
            imgsz = self.opt['img-size']
            device = select_device(self.opt['device'])
            half = device.type != 'cpu'
            model = self.opt['model']  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size
            if half:
                model.half()

            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
            img_reader = self.img_from_source
            # print(img_reader.shape)
            img = self.letterbox(img_reader, imgsz, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = model(img, augment= False)[0]
            # Apply NMS
            classes = None
            if self.opt['classes']:
                classes = []
                for class_name in self.opt['classes']:
                    print(class_name)
                    classes.append(self.opt['classes'].index(class_name))
            
            pred = non_max_suppression(pred, self.opt['conf-thres'], self.opt['iou-thres'], classes= classes, agnostic= False)
            for i, det in enumerate(pred):
                s = ''
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(img_reader.shape)[[1, 0, 1, 0]]
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_reader.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                number_object = 0
                for *xyxy, conf, cls in reversed(det):
                    number_object += 1
                    
                    # check probability > 40%
                    label = f'{names[int(cls)]}'
                    plot_one_box(xyxy, img_reader, label=label, color=colors[int(cls)], line_thickness=3)
                    # break
        return img_reader,number_object
