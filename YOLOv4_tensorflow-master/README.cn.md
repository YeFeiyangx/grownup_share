# YOLOv4_tensorflow | [English introductions](README.md)
* yolov4的纯tensorflow实现.
* 数据增强没有完全实现
* 持续更新
</br>

* rdc01234@163.com

## 使用说明
* 执行命令
```
python val.py
```
* 如果没有报错, 就没问题

## 转换 yolov4.weights
* 参考[这个权重转换文件](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/convert_weight.py), 我将 yolov4.weights 转换到了自己的代码中
* [yolov4.weights百度云链接](https://pan.baidu.com/s/1oMTW2dI8IrgGcqxn90fr_Q) 提取码: pqe2
* **将下载好的 yolov4.weights 放到 yolo_weights 文件夹下, 执行命令**
```
python convert_weight.py
python test_yolo_weights.py
```
* 会在 yolo_weights 文件夹下生成 ckpt 权重文件
* 并且你将会看到这样的画面,完美

![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/coco_save/dog.jpg)
* weights_name.txt 文件中存放的是图模型的卷积层和bn的名字

## 在 VOC2007 和 VOC2012 数据集上训练
* 打开 config.py ,将 voc_root_dir 修改为自己VOC数据集存放的根目录, voc_dir_ls 修改为自己想要训练的VOC数据集名
* 执行命令
```
python train_voc.py
```
* 训练完成后,将测试图片放到 voc_test_pic 文件夹下,执行命令
```
python val_voc.py
```
* 训练一天(364999步)的结果(input_size:416*416, batch_size:2, lr:2e-4, optimizer:momentum)，还不错

![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/voc_save/000302.jpg)
![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/voc_save/000288.jpg)
* **所有的配置参数都在 config.py 中，你可以按照自己的实际情况来修改**
* 此外，在VOC上训练时的 loss nan 问题应该已经被解决了.
* 这是我训练的损失图，学习率貌似有点太小了
```
python show_loss.py 20 300
```

![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/loss.png)

## 在自己的数据集上训练
* ./data/JPEGImages 文件夹中存放用**labelme**标注**json文件**的jpg图片和对应的json文件, 参考我给的  ./data/JPEGImages 文件夹下的格式
* 然后在 ./data 文件夹下执行 python 命令, 会自动产生 label 文件和 train.txt 文件
```
python generate_labels.py
```
* 继续执行命令,得到 anchor box
```
python k_means.py
```
* 打开 config.py, 将得到的 anchor box 写入到第六行，就像这样
```
anchors = 12,19, 19,27, 18,37, 21,38, 23,38, 26,39, 31,38, 39,44, 67,96
```
* 接下来，修改 data/train.names 中的内容为你需要训练的分类名字(不要用中文),并且将 config.py 中的分类数改为自己的分类数
* **所有的配置参数都在 config.py 中，你可以按照自己的实际情况来修改**
* 配置完成,执行命令
```
python train.py
```
* 训练完成后,将测试图片放到 test_pic 文件夹下,执行命令验证训练结果
```
python val.py
```
* 这是我用123张图片训练了 5000 步(25分钟)的结果，效果还不错

![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/save/62.jpg)

## 将 ckpt 模型转换为 pb 模型
* 打开ckpt2pb.py 文件, 修改里面的 'ckpt_file_dir', "class_num", "anchors"参数，执行命令
```
python ckpt2pb.py
```
* 在 'ckpt_file_dir' 目录下会看到生成的 pb 模型

## 有关 config.py 和训练的提示
1. config.py 中的 width 和 height 应该是 608，显存不够才调整为 416 的
2. 学习率不宜设置太高
3. 如果出现NAN的情况，请降低学习率

## 致谢
感谢以下同仁对仓库代码改进提供的宝贵意见</br>
1. [Jiachenyin1](https://github.com/Jiachenyin1)

## 自己的设备
GPU : 1660ti (华硕猛禽) 6G</br>
CPU : i5 9400f</br>
mem : 16GB</br>
os  : ubuntu 18.04</br>
cuda: 10.2</br>
cudnn : 7</br>
python : 3.6.9</br>
tensorflow-gpu:1.14.0</br>
numpy : 1.18.1</br>
opencv-python : 4.1.2.30</br>
