# coding:utf-8
# load voc dataset
import numpy as np
from src import Log
from utils import tools
from utils import data_augment
import random
import cv2
import os
from os import path

class Data():
    def __init__(self, voc_root_dir, voc_dir_ls, voc_names, class_num, batch_size, anchors, agument, width=608, height=608, data_debug=False):
        self.data_dirs = [path.join(path.join(voc_root_dir, voc_dir), "JPEGImages") for voc_dir in voc_dir_ls] 
        self.class_num = class_num  # classify number
        self.batch_size = batch_size
        self.anchors = np.asarray(anchors).astype(np.float32).reshape([-1, 2]) / [width, height]     #[9,2]
        print("anchors:\n", self.anchors)

        self.imgs_path = []
        self.labels_path = []

        self.num_batch = 0      # total batch number
        self.num_imgs = 0       # total number of images

        self.data_debug = data_debug

        self.width = width
        self.height = height
        self.agument = agument  # data agument strategy

        self.smooth_delta = 0.01 # label smooth delta

        self.names_dict = tools.word2id(voc_names)    # dictionary of name to id

        self.__init_args()
    
    # initial all parameters
    def __init_args(self):
        Log.add_log("data agument strategy : "+str(self.agument))
        # data augment strategy
        self.multi_scale_img = self.agument[0] # multiscale zoom the image
        self.keep_img_shape = self.agument[1]   # keep image's shape when we reshape the image
        self.flip_img = self.agument[2]    # flip image
        self.gray_img = self.agument[3]        # gray image
        self.label_smooth = self.agument[4]    # label smooth strategy
        self.erase_img = self.agument[5]        # random erase image
        self.invert_img = self.agument[6]                  # invert image pixel
        self.rotate_img = self.agument[7]           # random rotate image

        Log.add_log("message: begin to initial images path")

        # init imgs path
        for voc_dir in self.data_dirs:
            for img_name in os.listdir(voc_dir):
                img_path = path.join(voc_dir, img_name)
                label_path = img_path.replace("JPEGImages", "Annotations")
                label_path = label_path.replace(img_name.split('.')[-1], "xml")
                if not path.isfile(img_path):
                    Log.add_log("warning:VOC image'"+str(img_path)+"'is not a file")
                    continue
                if not path.isfile(label_path):
                    Log.add_log("warning:VOC label'"+str(label_path)+"'if not a file")
                    continue
                self.imgs_path.append(img_path)
                self.labels_path.append(label_path)
                self.num_imgs += 1        
        Log.add_log("message:initialize VOC dataset complete,  there are "+str(self.num_imgs)+" pictures in all")
        
        if self.num_imgs <= 0:
            raise ValueError("there are 0 pictures to train in all")
        
        return
        
    # read image 
    def read_img(self, img_file):
        '''
        read img_file, and resize it
        return:img, RGB & float
        '''
        img = tools.read_img(img_file)
        if img is None:
            return None

        if self.keep_img_shape:
            # keep image shape when we reshape the image
            img, new_w, new_h = data_augment.keep_image_shape_resize(img, size=[self.width, self.height])
        else:
            img = cv2.resize(img, (self.width, self.height))
            new_w, new_h = None, None

        if self.flip_img:
            # flip image
            img = data_augment.flip_img(img)
        
        if self.gray_img and (np.random.random() < 0.2):
            # probility of gray image is 0.2
            img = data_augment.gray_img(img)
        
        if self.erase_img and (np.random.random() < 0.3):
            # probility of random erase image is 0.3
            img = data_augment.erase_img(img, size_area=[20, 100])
        
        if self.invert_img and (np.random.random() < 0.1):
            # probility of invert image is 0.1
            img = data_augment.invert_img(img)

        if self.rotate_img:
            # rotation image
            img = data_augment.random_rotate_img(img)

        test_img = img

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img/255.0
        return img, new_w, new_h, test_img
    
    # read label file
    def read_label(self, label_file, names_dict, anchors, new_w, new_h):
        '''
        parsement label_file, and generates label_y1, label_y2, label_y3
        new_w: the truth value of image width when we resize it
        new_h: the truth value of image height when we resize it
        return:label_y1, label_y2, label_y3
        '''
        contents = tools.parse_voc_xml(label_file, names_dict)  
        if not contents:
            return None, None, None

        # flip the label
        if self.flip_img:
            for i in range(len(contents)):
                contents[i][1] = 1.0 - contents[i][1]

        if self.keep_img_shape:
            x_pad = (self.width - new_w) // 2
            y_pad = (self.height - new_h) // 2

        label_y1 = np.zeros((self.height // 32, self.width // 32, 3, 5 + self.class_num), np.float32)
        label_y2 = np.zeros((self.height // 16, self.width // 16, 3, 5 + self.class_num), np.float32)
        label_y3 = np.zeros((self.height // 8, self.width // 8, 3, 5 + self.class_num), np.float32)

        y_true = [label_y3, label_y2, label_y1]
        ratio = {0:8, 1:16, 2:32}

        test_result = []

        for label in contents:
            label_id = int(label[0])
            box = np.asarray(label[1: 5]).astype(np.float32)   # the value saved in label is x,y,w,h
            if self.keep_img_shape:
                # modify Coordinates
                box[0:2] = (box[0:2] * [new_w, new_h ] + [x_pad, y_pad]) / [self.width, self.height]
                box[2:4] = (box[2:4] * [new_w, new_h]) / [self.width, self.height]  
            
            test_result.append([box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2])
            
            best_giou = 0
            best_index = 0
            for i in range(len(anchors)):
                min_wh = np.minimum(box[2:4], anchors[i])
                max_wh = np.maximum(box[2:4], anchors[i])
                giou = (min_wh[0] * min_wh[1]) / (max_wh[0] * max_wh[1])
                if giou > best_giou:
                    best_giou = giou
                    best_index = i
            
            # 012->0, 345->1, 678->2
            x = int(np.floor(box[0] * self.width / ratio[best_index // 3]))
            y = int(np.floor(box[1] * self.height / ratio[best_index // 3]))
            k = best_index % 3

            y_true[best_index // 3][y, x, k, 0:4] = box
            # label smooth
            label_value = 1.0  if not self.label_smooth else ((1-self.smooth_delta) + self.smooth_delta * 1 / self.class_num)
            y_true[best_index // 3][y, x, k, 4:5] = label_value
            y_true[best_index // 3][y, x, k, 5:-1] = 0.0 if not self.label_smooth else self.smooth_delta / self.class_num
            y_true[best_index // 3][y, x, k, 5 + label_id] = label_value
        
        return label_y1, label_y2, label_y3, test_result


    # load batch_size images
    def __get_data(self):
        '''
        load  batch_size labels and images
        return:imgs, label_y1, label_y2, label_y3
        '''
        # random resize the image per ten batch 
        if self.multi_scale_img and (self.num_batch % 10 == 0):
            random_size = random.randint(13, 23) * 32
            self.width = self.height = random_size
        
        imgs = []
        labels_y1, labels_y2, labels_y3 = [], [], []

        count = 0
        while count < self.batch_size:
            curr_index = random.randint(0, self.num_imgs - 1)
            img_name = self.imgs_path[curr_index]
            label_name = self.labels_path[curr_index]

            # probility of  flip image is 0.5
            if self.agument[2]  and (np.random.random() < 0.5):
                self.flip_img = True
                # print("flip")
            else:
                self.flip_img = False

            img, new_w, new_h, test_img = self.read_img(img_name)
            label_y1, label_y2, label_y3, test_result = self.read_label(label_name, self.names_dict, self.anchors, new_w, new_h)
            
            # show data agument result
            if self.data_debug:
                test_img = tools.draw_img(test_img, test_result, None, None, None, None)
                cv2.imshow("letterbox_img", test_img)
                cv2.waitKey(0)
            
            if img is None:
                Log.add_log(" VOC file'" + img_name + "'is None")
                continue
            if label_y1 is None:
                Log.add_log("VOC file'" + label_name + "'is None")
                continue
            imgs.append(img)
            labels_y1.append(label_y1)
            labels_y2.append(label_y2)
            labels_y3.append(label_y3)

            count += 1

        self.num_batch += 1
        imgs = np.asarray(imgs)
        labels_y1 = np.asarray(labels_y1)
        labels_y2 = np.asarray(labels_y2)
        labels_y3 = np.asarray(labels_y3)
        
        return imgs, labels_y1, labels_y2, labels_y3

    # Iterator
    def __next__(self):
        '''    get batch images    '''
        return self.__get_data()

    


