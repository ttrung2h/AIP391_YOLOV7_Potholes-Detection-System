import torch
import torchvision
from PIL import Image
import random 
import os
TRAIN_DIR_IMAGE = r'D:\FPT\ky5\AIP391\My Project\data\data_potholes\train\images'
TRAIN_DIR_LABEL = r'D:\FPT\ky5\AIP391\My Project\data\data_potholes\train\labels'
TEST_DIR_IMAGE = r'D:\FPT\ky5\AIP391\My Project\data\data_potholes\test\images'
TEST_DIR_LABEL = r'D:\FPT\ky5\AIP391\My Project\data\data_potholes\test\labels'
class YoloColorFillter:
    def __init__(self, imgfile,labelfile):
        self.imgfile = imgfile
        self.labelfile = labelfile
    def color_fillter(self):
        
        # set up the random variables
        brightness = random.uniform(0.5, 1)
        contrast = random.uniform(0.5, 1)
        saturation = random.uniform(0.5, 1)
        hue = random.uniform(0, 0.5)
        # Define the transforms
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            torchvision.transforms.ToTensor(),
        ])

        # Load the image
        img = Image.open(self.imgfile)

        # Apply the transforms
        img_transformed = transform(img)

        # Save the transformed image
        img_transformed_pil = torchvision.transforms.functional.to_pil_image(img_transformed)
        img_transformed_pil.save(self.imgfile.split('.')[0] + '_color.jpg')
        
        # save label
        f = open(self.labelfile, 'r')
        fr =f.readlines()
        boxes = []
        for line in fr:
            class_namme,x, y, w, h = line.strip('\n').split(' ')
            x,y,w,h = float(x),float(y),float(w),float(h)
            boxes.append([class_namme,x, y, w, h])
        with open(self.labelfile.split('.')[0]+'_color.txt', 'w') as f:
            # print("Meow")
            for box in boxes:
                f.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n')
        f.close()

train_imgs = [os.path.join(TEST_DIR_IMAGE ,x) for x in os.listdir(TEST_DIR_IMAGE)]
train_labels = [os.path.join(TEST_DIR_LABEL ,x) for x in os.listdir(TEST_DIR_LABEL)]
for i,img_file in enumerate(train_imgs):
        label_file = train_labels[i]
        yolo = YoloColorFillter(img_file,label_file).color_fillter()