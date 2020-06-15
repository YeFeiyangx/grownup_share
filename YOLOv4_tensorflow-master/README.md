# YOLOv4_tensorflow | [中文说明](README.cn.md)
* Implement yolov4 with pure tensorflow
*  implemented the part of data augment strategies
* continuous update the code
</br>

* rdc01234@163.com

## introductions
* run the following command.
```
python val.py
```
* if have no error, it's ok

## convert yolov4.weights to fit our code
* refer to [this weights convert file](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/convert_weight.py), i converted yolov4.weights to this project.
* you can download yolov4.weights from [google](www.google.com)
* **put yolov4.weights into the "yolo_weights" folder, and run the command.**
```
python convert_weight.py
python test_yolo_weights.py
```
* the ckpt weights file wound exits in the 'yolo_weights' folder
* and you'll see some images like this, it seems perfect

![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/coco_save/dog.jpg)

* the weights_name.txt contains all model layer's name of the network 

## train on VOC2007 and VOC2012
* open config.py and modify the **voc_root_dir** to the root of your VOC dataset, modiry the **voc_dir_ls** to the name of VOC dataset witch  you want to train </br>
* run the command
```
python train_voc.py
```
* put test images into **voc_test_pic** folder, and run the following command after the model training finished.</br>
```
python val_voc.py
```
* it's the result of our code training(input_size:416*416, batch_size:2, lr:2e-4, optimizer:momentum) for a day(364999 steps), not bad

![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/voc_save/000302.jpg)
![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/voc_save/000288.jpg)

* **all configuration parameters are in the config.py, you can modify them according to your actual situation**
* in addition, the bug of loss sometimes become Nan  when i training on VOC should has been repaired.
* it's the image of loss value, and seems that the lr is too lower(2e-4), we should set it larger.
```
python show_loss.py 20 300
```

![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/loss.png)

## train with own dataset
* The jpg image and the corresponding json file which marked with **labelme** are stored in the folder **./data/JPEGImages**, just like what I do in the ./data/JPEGImages  folder
* and then, go to the folder **./data**, execute the following python command, it automatically generates label files and train.txt
```
python generate_labels.py
```
* excute the python command, to get anchor box
```
python k_means.py
```
* open config.py, write the anchor box to line 6, just like this
```
anchors = 12,19, 19,27, 18,37, 21,38, 23,38, 26,39, 31,38, 39,44, 67,96
```
* and, now, modify the content in **data/train.names** to the category name that you need to train, and change the **class_num** in config.py to your own category number.
* **all configuration parameters are in the config.py, you can modify them according to your actual situation**
* ok, that's all, execute the command
```
python train.py
```
* put test images into **test_pic** folder, and run the following command after the model training finished.
```
python val.py
```
* this image is the result of training 5000 steps (25 minutes) with 123 pictures, it looks not bad. 

![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/save/62.jpg)

## convert ckpt model to pb model
* open ckpt2pb.py , and modify 'ckpt_file_dir', 'class_num', 'anchors'. run.
```
python ckpt2pb.py
```
* you will see a pb model in 'ckpt_file_dir'

## some tips with config.py and train the model
1. the parameters of **width and height** in config.py should be 608, but i have not a powerful GPU, that is why i set them as 416
2. learning rate do not set too large
3. when the loss value is Nan, please lower your learning rate.

## Thanks
Thanks to the following friends for their valuable comments on code improvements.</br>
1. [Jiachenyin1](https://github.com/Jiachenyin1)

## my device
GPU : 1660ti (ASUS) 6G</br>
CPU : i5 9400f</br>
mem : 16GB</br>
os  : ubuntu 18.04</br>
cuda: 10.2</br>
cudnn : 7</br>
python : 3.6.9</br>
tensorflow-gpu:1.14.0</br>
numpy : 1.18.1</br>
opencv-python : 4.1.2.30</br>
