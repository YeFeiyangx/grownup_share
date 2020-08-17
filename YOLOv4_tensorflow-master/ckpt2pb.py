# coding:utf-8
# convert ckpt model to pb model

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
# for save pb file
from tensorflow.python.framework import graph_util

# change this to your ckpt model directory
ckpt_file_dir = "./VOC"
# your pb model name
pd_dir = path.join(ckpt_file_dir, "model.pb")
# change this to your class_num
class_num = config.voc_class_num
# your anchors
anchors = config.voc_anchors

def main():
    yolo = YOLO(class_num, anchors)

    # Placeholder:0
    inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
    feature_y1, feature_y2, feature_y3 = yolo.forward(inputs, isTrain=False)
    # concat_9:0, concat_10:0, concat_11:0
    pre_boxes, pre_score, pre_label = yolo.get_predict_result(feature_y1, feature_y2, feature_y3, class_num, 
                                                                                                score_thresh=config.val_score_thresh, iou_thresh=config.iou_thresh, max_box=config.max_box)

    init = tf.compat.v1.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        ckpt = tf.compat.v1.train.get_checkpoint_state(ckpt_file_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            Log.add_log("message: ckpt model:'"+str(ckpt.model_checkpoint_path)+"'")
        else:
            Log.add_log("message:no ckpt model")
            exit(1)

        # save  PB model
        out_graph = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,['Placeholder','yolo/Conv_1/BiasAdd', 'yolo/Conv_9/BiasAdd', 'yolo/Conv_17/BiasAdd', 'concat_9', 'concat_10', 'concat_11'])  # "yolo/Conv_13/BiasAdd"
        saver_path = tf.train.write_graph(out_graph,"",pd_dir,as_text=False)
        print("saver path: ",saver_path)

if __name__ == "__main__":
    Log.add_log("message: convert ckpt model to pb model")
    main()