# coding:utf-8
# test yolov4.weights

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import config
from utils import tools
from src.YOLO import YOLO
import cv2
import numpy as np
import os
from os import path
import time

def read_img(img_name, width, height, keep_img_shape = config.keep_img_shape):
    img_ori = tools.read_img(img_name)
    if img_ori is None:
        return None, None
    if not keep_img_shape:
        img = cv2.resize(img_ori, (width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # add by rdc ,follow @Jiachenyin1
        target_h , target_w = height , width
        ori_h , ori_w , _= img_ori.shape
        scale = min(target_h / ori_h , target_w / ori_w)
        nw  , nh = int(scale * ori_w) , int(scale * ori_h)
        image_resized = cv2.resize(img_ori , (nw , nh))  ## width and height
        img = np.full(shape = [target_h , target_w , 3] , fill_value= 0, dtype=np.uint8)
        dh , dw = (target_h - nh)//2 , (target_w - nw)//2
        img[dh:(nh+dh) , dw:(nw+dw),:] = image_resized
        img_ori = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)
    img = img/255.0
    # [416, 416, 3] => [1, 416, 416, 3]
    img = np.expand_dims(img, 0)
    return img, img_ori

def save_img(img, name):
    '''
    img: mat
    '''
    save_dir = "coco_save"
    if not path.isdir(save_dir):
        os.mkdir(save_dir)
    img_name = path.join(save_dir, name)
    cv2.imwrite(img_name, img)
    return 


def main():
    anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
    yolo = YOLO(80, anchors)

    inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
    feature_y1, feature_y2, feature_y3 = yolo.forward(inputs, isTrain=False)
    pre_boxes, pre_score, pre_label = yolo.get_predict_result(feature_y1, feature_y2, feature_y3, 80, 
                                                                                                score_thresh=config.val_score_thresh, iou_thresh=config.iou_thresh, max_box=config.max_box)

    init = tf.compat.v1.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        ckpt = tf.compat.v1.train.get_checkpoint_state("./yolo_weights")
        if ckpt and ckpt.model_checkpoint_path:
            print("restore: ", ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            exit(1)

        # id to names
        word_dict = tools.get_word_dict("./data/coco.names")
        # color of corresponding names
        color_table = tools.get_color_table(80)

        width = 608
        height = 608
        
        val_dir = "./coco_test_img"
        for name in os.listdir(val_dir):
            img_name = path.join(val_dir, name)
            if not path.isfile(img_name):
                print("'%s' is not a file" %img_name)
                continue

            start = time.perf_counter()

            img, img_ori = read_img(img_name, width, height)
            if img is None:
                continue
            boxes, score, label = sess.run([pre_boxes, pre_score, pre_label], feed_dict={inputs:img})
            
            end = time.perf_counter()
            print("%s\t, time:%f s" %(img_name, end-start))

            img_ori = tools.draw_img(img_ori, boxes, score, label, word_dict, color_table)

            cv2.imshow('img', img_ori)
            cv2.waitKey(0)

            save_img(img_ori, name)

if __name__ == "__main__":
    main()
