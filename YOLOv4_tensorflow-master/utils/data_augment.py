# coding:utf-8
# data augment
import numpy as np
import cv2
import random

# resize image and keep the value of width / heigh
def keep_image_shape_resize(bgr_img, size=[416, 416]):
    '''
    bgr_img : cv2 image (BGR)
    size : [resize_w, resize_h]
    '''
    ori_h , ori_w , _= bgr_img.shape
    target_h, target_w = size[1], size[0]
    # resize 
    scale = min(target_h/ori_h, target_w/ori_w)
    nw, nh = int(scale * ori_w), int(scale * ori_h)
    img_resize = cv2.resize(bgr_img, (nw, nh))
    img = np.full(shape=[target_h, target_w, 3], fill_value=0, dtype=np.uint8)
    dw, dh = (target_w - nw)//2, (target_h - nh)//2
    img[dh:(nh+dh), dw:(nw+dw), :] = img_resize
    return img, nw, nh


# flip the image vertically
def flip_img(bgr_img):
    '''
        bgr_img:cv2 image (BGR)
    '''
    bgr_img = cv2.flip(bgr_img, 1)
    return bgr_img

# gray
def gray_img(bgr_img):
    ''' bgr_img: cv2 image (BGR) '''
    tmp = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    bgr_img = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
    return bgr_img

# random erase image
def erase_img(bgr_img, size_area=[20, 100]):
    '''
    bgr_img: cv2 image (BGR)
    size_area: the erase area size of image
    '''
    min_size = size_area[0]
    max_size = size_area[1]
    height = bgr_img.shape[0]
    width = bgr_img.shape[1]
    erase_w = random.randint(min_size, max_size)
    erase_h = random.randint(min_size, max_size)
    x = random.randint(0, width - erase_w)
    y = random.randint(0, height - erase_h)
    value = random.randint(0, 255)
    bgr_img[y:y+erase_h, x:x+erase_w, : ] = value
    return bgr_img

# 1.0 - pixels.value
def invert_img(bgr_img):
    ''' bgr_img:cv2 image (BGR) '''
    bgr_img = 255 - bgr_img
    return bgr_img

# rotate image
def random_rotate_img(bgr_img, angle_min=-6, angle_max=6):
    ''' bgr_img:cv2 image (BGR) '''
    angle = np.random.randint(angle_min, angle_max)
    height = bgr_img.shape[0]
    width = bgr_img.shape[1]
    M = cv2.getRotationMatrix2D((height/2.0, width/2.0),angle,1)
    bgr_img = cv2.warpAffine(bgr_img,M,(width,height))
    return bgr_img