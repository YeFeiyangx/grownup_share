# coding:utf-8
# test on voc

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import config
from utils import tools
from utils import data_augment
from src.YOLO import YOLO
import cv2
import numpy as np
from src import Log
import os
from os import path
import time

def read_img(img_name, width, height):
    img_ori = tools.read_img(img_name)
    if img_ori is None:
        return None, None
    if config.keep_img_shape:
        img, nw, nh = data_augment.keep_image_shape_resize(img_ori, size=[width, height])
    else:
        img = cv2.resize(img_ori, (width, height))
        nw, nh = None, None

    show_img = img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.0
    # [416, 416, 3] => [1, 416, 416, 3]
    img = np.expand_dims(img, 0)
    return img, nw, nh, img_ori, show_img

def save_img(img, name):
    '''
    img: mat 
    name: saved name
    '''
    voc_save_dir = config.voc_save_dir
    if not path.isdir(voc_save_dir):
        Log.add_log("message: create folder'"+str(voc_save_dir)+"'")
        os.mkdir(voc_save_dir)
    img_name = path.join(voc_save_dir, name)
    cv2.imwrite(img_name, img)
    return 


def main():
    yolo = YOLO(config.voc_class_num, config.voc_anchors)

    inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
    feature_y1, feature_y2, feature_y3 = yolo.forward(inputs, isTrain=False)
    pre_boxes, pre_score, pre_label = yolo.get_predict_result(feature_y1, feature_y2, feature_y3, config.voc_class_num, 
                                                                                                score_thresh=config.val_score_thresh, iou_thresh=config.iou_thresh, max_box=config.max_box)

    init = tf.compat.v1.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        ckpt = tf.compat.v1.train.get_checkpoint_state(config.voc_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            Log.add_log("message: load ckpt model:'"+str(ckpt.model_checkpoint_path)+"'")
        else:
            Log.add_log("message:can not find  ckpt model")
            # exit(1)
        
        # dictionary of name of corresponding id
        word_dict = tools.get_word_dict(config.voc_names)
        # dictionary of per names
        color_table = tools.get_color_table(config.voc_class_num)

        width = config.width
        height = config.height
        
        for name in os.listdir(config.voc_test_dir):
            img_name = path.join(config.voc_test_dir, name)
            if not path.isfile(img_name):
                print("'%s' is not file" %img_name)
                continue

            start = time.perf_counter()

            img, nw, nh, img_ori, show_img = read_img(img_name, width, height)
            if img is None:
                Log.add_log("message:'"+str(img)+"' is None")
            boxes, score, label = sess.run([pre_boxes, pre_score, pre_label], feed_dict={inputs:img})
            
            end = time.perf_counter()
            print("%s\t, time:%f s" %(img_name, end-start))

            if config.keep_img_shape:
                # modify coordinates
                dw = (width - nw)/2
                dh = (height - nh)/2
                for i in range(len(boxes)):
                    boxes[i][0] = (boxes[i][0] * width - dw)/nw
                    boxes[i][1] = (boxes[i][1] * height - dh)/nh
                    boxes[i][2] = (boxes[i][2] * width - dw)/nw
                    boxes[i][3] = (boxes[i][3] * height - dh)/nh
           
            img_ori = tools.draw_img(img_ori, boxes, score, label, word_dict, color_table)

            cv2.imshow('img', img_ori)
            cv2.waitKey(0)

            if config.save_img:
                save_img(img_ori, name)
            pass

if __name__ == "__main__":
    Log.add_log("message: into val.main()")
    main()
