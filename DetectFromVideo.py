import cv2
import os
import sys
from detector import Detector
import time
import torch
from datetime import datetime
import geocoder
import json
import argparse

# load model
start_time = time.time()
sys.path.append('yolov7')
from models.experimental import attempt_load
model = attempt_load(r"weight\last.pt", map_location=torch.device('cuda:0'))  # load FP32 model
print("Done after %s seconds" % (time.time() - start_time))


# load config
classes_to_filter = None
opt  = {
    "model" : model,
    "weights": r"yolov7\runs\train\last.pt", # Path to weights file default weights are for nano model
    "yaml"   : r"yolov7\data\mydataset.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.1, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : '1',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter  # list of classes to filter or None

}

def export_information(number_potholes,frame_number,time,path_export):
    '''
    Export information to json file
    '''
    g = geocoder.ip('me')
    info = g.json
    info['number_potholes'] = number_potholes
    info['time'] = time
    info['frame_number'] = frame_number
    
    # export to json file
    with open(path_export, 'w') as outfile:
        json.dump(info, outfile,indent=4)
    return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='0', help='video url. If you do not provide any video url, the program will use your webcam')
    parser.add_argument('--trackingfolder', type=str, default='tracking_data', help='folder to save tracking data')
    config = parser.parse_args()
    video_url = config.video
    
    # Check must contain link video
    if config.video == None:
        print('Please input link of video')
        exit()
    
    # Check if webcam or vitual camera
    try:
        if int(config.video) >=0:
            video_url = int(config.video)
    except:
        pass
    
    # Set path to save data
    tracking_data_url = config.trackingfolder + "/"
    
    # Create folder to save data in day-month-year_hour-minute-second
    folderCurrentTimeChecking= tracking_data_url+datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    
    # Check if not exist mkdir data and create folder data
    if not os.path.exists(folderCurrentTimeChecking):
        os.makedirs(folderCurrentTimeChecking) 
    
    # capture from any video
    vid = cv2.VideoCapture(video_url)
    currentFrame = 0
    # # loop for extracting all frame in video and outloop when click 'q'
    while vid.isOpened():
        new_frame_time = time.time()
        success, frame = vid.read()
        detector = Detector(frame, opt)
        img,number_potholes = detector.detect_plotbox()
        cv2.imshow(f'Checking camera',frame)
        
        currentFrame+=1
        
        # check if frame is not over 10
        if number_potholes > 2:
            
            # create folder to save frame information
            path = folderCurrentTimeChecking+'/frame'+str(currentFrame)
            if not os.path.exists(path):
                os.makedirs(path)
                # write image and number of potholes to file
                cv2.imwrite(path+f'/frame{currentFrame}_detectedFrame.jpg',img)
                
                # get current time for saving information
                time_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                export_information(number_potholes,currentFrame,time_now,path+f'/frame{currentFrame}_detectedFrame.json')
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
   