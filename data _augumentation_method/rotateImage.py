import os 
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
TRAIN_DIR_IMAGE = r'train\images'
TRAIN_DIR_LABEL = r'train\labels'
TEST_DIR_IMAGE = r'test\images'
TEST_DIR_LABEL = r'test\labels'
# rotate image and bounding box
class yoloRotatebbox:
    def __init__(self, filename, image_ext,labelfile, angle):
        assert os.path.isfile(filename + image_ext)
        assert os.path.isfile(labelfile + '.txt')

        self.filename = filename
        self.image_ext = image_ext
        self.angle = angle
        self.labelfile = labelfile

        # Read image using cv2
        self.image = cv2.imread(self.filename + self.image_ext, 1)
        # create a 2D-rotation matrix
        rotation_angle = self.angle * np.pi / 180
        self.rot_matrix = np.array(
            [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
    def rotate_image(self):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """
        height, width = self.image.shape[:2]  # image shape has 3 dimensions
        image_center = (width / 2,
                        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origin) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(self.image, rotation_mat, (bound_w, bound_h))
        return rotated_mat
    
    # create function to write new image
    def write_image(self):
        cv2.imwrite(self.filename + '_rotated' + self.image_ext, self.rotate_image())
    def yoloFormattocv(self,x1, y1, x2, y2, H, W):
        bbox_width = x2 * W
        bbox_height = y2 * H
        center_x = x1 * W
        center_y = y1 * H
        voc = []
        voc.append(center_x - (bbox_width / 2))
        voc.append(center_y - (bbox_height / 2))
        voc.append(center_x + (bbox_width / 2))
        voc.append(center_y + (bbox_height / 2))
        return [int(v) for v in voc]
    
    def rotateYolobbox(self):

        new_height, new_width = self.rotate_image().shape[:2]

        f = open( self.labelfile + '.txt', 'r')

        f1 = f.readlines()

        new_bbox = []

        H, W = self.image.shape[:2]

        for x in f1:
            bbox = x.strip('\n').split(' ')
            if len(bbox) > 1:
                (center_x, center_y, bbox_width, bbox_height) = self.yoloFormattocv(float(bbox[1]), float(bbox[2]),
                                                                               float(bbox[3]), float(bbox[4]), H, W)
                
                # shift the origin to the center of the image.
                upper_left_corner_shift = (center_x - W / 2, -H / 2 + center_y)
                upper_right_corner_shift = (bbox_width - W / 2, -H / 2 + center_y)
                lower_left_corner_shift = (center_x - W / 2, -H / 2 + bbox_height)
                lower_right_corner_shift = (bbox_width - W / 2, -H / 2 + bbox_height)

                new_lower_right_corner = [-1, -1]
                new_upper_left_corner = []

                for i in (upper_left_corner_shift, upper_right_corner_shift, lower_left_corner_shift,
                          lower_right_corner_shift):
                    new_coords = np.matmul(self.rot_matrix, np.array((i[0], -i[1])))
                    x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
                    if new_lower_right_corner[0] < x_prime:
                        new_lower_right_corner[0] = x_prime
                    if new_lower_right_corner[1] < y_prime:
                        new_lower_right_corner[1] = y_prime

                    if len(new_upper_left_corner) > 0:
                        if new_upper_left_corner[0] > x_prime:
                            new_upper_left_corner[0] = x_prime
                        if new_upper_left_corner[1] > y_prime:
                            new_upper_left_corner[1] = y_prime
                    else:
                        new_upper_left_corner.append(x_prime)
                        new_upper_left_corner.append(y_prime)
                #             print(x_prime, y_prime)
                    new_center = []
                    new_shape = []
                    new_center.append((new_lower_right_corner[0] + new_upper_left_corner[0]) / 2)
                    new_center.append((new_lower_right_corner[1] + new_upper_left_corner[1]) / 2)
                    new_shape.append(new_lower_right_corner[0] - new_upper_left_corner[0])
                    new_shape.append(new_lower_right_corner[1] - new_upper_left_corner[1])
                new_bbox.append([bbox[0], new_center[0]/new_width, new_center[1]/new_height,
                                 new_shape[0]/new_width, new_shape[1]/new_height])

        return new_bbox
    # write new bounding box to file
    def write_bbox(self):
        new_bbox = self.rotateYolobbox()
        f = open(self.labelfile + '_rotated.txt', 'w')
        for b in new_bbox:
            f.write(b[0] + ' ' + str(b[1]) + ' ' + str(b[2]) + ' ' + str(b[3]) + ' ' + str(b[4]) + '\n')

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
    
    train_imgs = [os.path.join(TRAIN_DIR_IMAGE ,x) for x in os.listdir(TRAIN_DIR_IMAGE)]
    train_labels = [os.path.join(TRAIN_DIR_LABEL ,x) for x in os.listdir(TRAIN_DIR_LABEL)]
    for i,img in enumerate(train_imgs):
        image_file = img[:-4]
        label = train_labels[i][:-4]
        image_ext = img[-4:]
        angle = random.randint(-45, 45)
        rotated = yoloRotatebbox(image_file, image_ext,label, angle)
        rotated.write_image()
        rotated.write_bbox()
 
    
    