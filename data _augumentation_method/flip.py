import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np  
import os
# Read the image and bounding box coordinates
TRAIN_DIR_IMAGE = r'train\images'
TRAIN_DIR_LABEL = r'train\labels'
TEST_DIR_IMAGE = r'test\images'
TEST_DIR_LABEL = r'test\labels'
class yoloFlipbbox:
    
    def __init__(self, imgfile,labelfile):
        self.imgfile = imgfile
        self.labelfile = labelfile    
    def save_rotated_image_and_bbox(self):
        
        img = cv2.imread(self.imgfile)
        f = open(self.labelfile, 'r')
        fr =f.readlines()
        flipped_img = cv2.flip(img, 1)  # Flip horizontally
        boxes = []
        for line in fr:
            class_namme,x, y, w, h = line.strip('\n').split(' ')
            x,y,w,h = float(x),float(y),float(w),float(h)
            flipped_x = 1 - x -w  # Calculate the new x-coordinate
            flipped_y = y  # The y-coordinate remains the same
            flipped_w = w  # The width remains the same
            flipped_h = h  # The height remains the same
            boxes.append([class_namme,flipped_x, flipped_y, flipped_w, flipped_h])
        
        # Save the flipped image and bounding box coordinates to a new file
        flipped_filename = self.imgfile.split('.')[0] + '_flipped.jpg'
        cv2.imwrite(flipped_filename, flipped_img)

        with open(self.labelfile.split('.')[0]+'_flipped.txt', 'w') as f:
            for box in boxes:
                f.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n')
        f.close()
        return flipped_img, boxes
def plot_bounding_box(image, annotation_list):
    '''
    Plots the bounding box on the image
    '''
    annotations = np.array(annotation_list).astype(np.float32)
    w, h = image.size
    print(annotation_list)
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
    plt.imshow(np.array(image))
    plt.show()

if __name__ == '__main__':
    TEST_imgs = [os.path.join(TEST_DIR_IMAGE ,x) for x in os.listdir(TEST_DIR_IMAGE)]
    TEST_labels = [os.path.join(TEST_DIR_LABEL ,x) for x in os.listdir(TEST_DIR_LABEL)]
    for i,img in enumerate(TEST_imgs):
        label = TEST_labels[i]
        flipBox = yoloFlipbbox(img,label)
        flipBox.save_rotated_image_and_bbox()
