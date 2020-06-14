#%%

import numpy as np
import os
import random
import sys
from PIL import Image, ImageEnhance, ImageDraw

PATH = os.path.abspath('.') + '/data/train'

#%%
def relation_gen(file_dir):
    output_relation_dict = dict()
    output_relation_dict_reverse = dict()
    for root, dirs, files in os.walk(file_dir):
        for subfile in files:
            if subfile.endswith('jpeg'):
                _name = subfile[:-5]+'.json'
                output_relation_dict[subfile] = _name
                output_relation_dict_reverse[_name] = subfile
    return output_relation_dict, output_relation_dict_reverse
        
p2l_relation_dict, l2p_relation_dict = relation_gen(PATH)

#%%
# ================================================start================================================
# ================================================start================================================
## 图片翻转和平移&扩缩的信息观测
import pandas as pd
import json
from collections import defaultdict

json_transfer_dict = defaultdict(lambda: defaultdict(lambda: defaultdict()))
label_analysis_df = pd.DataFrame(data=None,columns=['pic_name','label_name', 'label_index', 'label_total_num', 'label', 'num', 'points', 
                                                    'left_down', 'right_up', 'left', 'down', 'right', 'up',
                                                    'img_length', 'img_hight', 
                                                    'left_dis',  'down_dis', 'right_dis', 'up_dis'])
len_set = set()
row_num = 0
for _label_name in l2p_relation_dict.keys():
    with open(os.path.join(PATH, _label_name),'r') as fp:
        _name = _label_name[:-5]
        _pic_name = _name + '.jpeg'
        _num = int(_name.split('_')[1][1:])
        temp = json.loads(fp.read())
        _label_total_num = len(temp['shapes'])
        len_set.add(_label_total_num)
    with Image.open(os.path.join(PATH, l2p_relation_dict[_label_name]),'r') as im:
        _length, _hight = im.size
        _length = int(_length)
        _hight = int(_hight)
    for _obj_index in range(_label_total_num):
        _label = temp['shapes'][_obj_index]['label']
        
        _corner_0 = tuple(temp['shapes'][_obj_index]['points'][0])
        _corner_1 = tuple(temp['shapes'][_obj_index]['points'][1])
        _left = min(int(_corner_0[0]), int(_corner_1[0]))
        _right = max(int(_corner_0[0]), int(_corner_1[0]))
        _down = min(int(_corner_0[1]), int(_corner_1[1]))
        _up = max(int(_corner_0[1]), int(_corner_1[1]))
        _left_down = (_left, _down)
        _right_up = (_right, _up)
        _point = (_left_down, _right_up)
        _left_dis = int(_left_down[0])
        _down_dis = int(_left_down[1])
        _right_dis = int(_length) - int(_right_up[0])
        _up_dis = int(_hight) - int(_right_up[1])
        json_transfer_dict[_pic_name][_obj_index]['label_index'] = _obj_index
        json_transfer_dict[_pic_name][_obj_index]['label_total_num'] = _label_total_num
        json_transfer_dict[_pic_name][_obj_index]['label'] = _label
        json_transfer_dict[_pic_name][_obj_index]['num'] = _num
        json_transfer_dict[_pic_name][_obj_index]['points'] = _point
        json_transfer_dict[_pic_name][_obj_index]['left_down'] = _left_down
        json_transfer_dict[_pic_name][_obj_index]['right_up'] = _right_up
        json_transfer_dict[_pic_name][_obj_index]['left'] = _left
        json_transfer_dict[_pic_name][_obj_index]['right'] = _right
        json_transfer_dict[_pic_name][_obj_index]['down'] = _down
        json_transfer_dict[_pic_name][_obj_index]['up'] = _up
        json_transfer_dict[_pic_name][_obj_index]['img_length'] = _length
        json_transfer_dict[_pic_name][_obj_index]['img_hight'] = _hight
        json_transfer_dict[_pic_name][_obj_index]['left_dis'] = _left_dis
        json_transfer_dict[_pic_name][_obj_index]['down_dis'] = _down_dis
        json_transfer_dict[_pic_name][_obj_index]['right_dis'] = _right_dis
        json_transfer_dict[_pic_name][_obj_index]['up_dis'] = _up_dis
        json_transfer_dict[_pic_name][_obj_index]['max_ratio'] = 2
        _temp_list = [_pic_name, _label_name, _obj_index, _label_total_num, _label, _num, _point, _left_down, _right_up, _left, _down, _right, _up, _length, _hight, _left_dis, _down_dis, _right_dis, _up_dis]
        label_analysis_df.loc[row_num] = _temp_list
        row_num+=1

label_analysis_df.sort_values(by='num', inplace=True)
label_analysis_df.reset_index(drop=True,inplace=True)
print(label_analysis_df.head(3))
#%%
label_analysis_df.to_csv(os.path.join(os.path.abspath('.')+'/observation_file', 'image_analysis.csv'))
label_analysis_df.to_pickle(os.path.join(os.path.abspath('.')+'/observation_file','image_analysis.pkl'))

#%%
set(label_analysis_df['label']) # {'gy', 'lk', 'sy'}

# %%
label_analysis_df['label'].value_counts()
"""
sy    912 -> 增加对比色,左右翻转
gy    626 -> 增加对比色,左右翻转
lk    110 -> 数据要大大增加，增加各种对比色,左右翻转

对结果输入前，也要做对比色加强，分为七个程度，最后投票选出结果
"""

# %%
# 1538~1647 : 110 * 320  :141
# 912~1537  : 626 * 60   :1418
# 0~911     : 912 * 40   :1927
print('lk num:', label_analysis_df[label_analysis_df['label']=='lk'].shape[0])
print('gy num:', label_analysis_df[label_analysis_df['label']=='gy'].shape[0])
print('sy num:', label_analysis_df[label_analysis_df['label']=='sy'].shape[0])

lk_name_set = set(label_analysis_df[label_analysis_df['label']=='lk']['pic_name'].tolist())
gy_name_set = set(label_analysis_df[label_analysis_df['label']=='gy']['pic_name'].tolist())
sy_name_set = set(label_analysis_df[label_analysis_df['label']=='sy']['pic_name'].tolist())

## result -> set()
for i in [lk_name_set,gy_name_set,sy_name_set]:
    for j in [lk_name_set,gy_name_set,sy_name_set]:
        if i == j:
            pass
        else:
            print(i&j)

# %%
# 'label_name', 'label_index', 'label_total_num', 'label', 'num', 'points', 'left_down', 'right_up', 
# 'img_length', 'img_hight', 'left_dis',  'down_dis', 'right_dis', 'up_dis'
label_analysis_df
augment_info =pd.DataFrame(data=None,columns=None)
for _label_name,_df in label_analysis_df.groupby(['label_name']):
    if len(_df)==1:
        augment_info = augment_info.append(_df, ignore_index=True)
    else:
        _left_dis = int(_df['left_dis'].min())
        _down_dis = int(_df['down_dis'].min())
        _right_dis = int(_df['right_dis'].min())
        _up_dis = int(_df['up_dis'].min())
        _sub_df = _df.iloc[0]
        _sub_df['left_dis'] = _left_dis
        _sub_df['down_dis'] = _down_dis
        _sub_df['right_dis'] = _right_dis
        _sub_df['up_dis'] = _up_dis
        augment_info = augment_info.append(_sub_df, ignore_index=True)
augment_info[['left_dis', 'down_dis', 'right_dis', 'up_dis']] = augment_info[['left_dis', 'down_dis', 'right_dis', 'up_dis']].astype(int)
augment_info.sort_values(by='num',inplace=True)
augment_info.reset_index(drop=True,inplace=True)
augment_info.head(3)

# %%
# ================================================start================================================
# ================================================start================================================
import matplotlib.image as plimg
import copy

color_std_100 = []
color_std_30 = []
## 图片对比度与饱和度的信息观测
for i in range(len(augment_info)):
    _series = augment_info.iloc[i]
    _pic_name = _series['pic_name']
    with Image.open(os.path.join(PATH, _pic_name),'r') as im:
        _d = im.split()[0]
        d_arr = plimg.pil_to_array(_d)
        d_arr_100 = copy.deepcopy(d_arr.reshape([-1,]))
        d_arr_100.sort()
        d_arr_30 = d_arr_100[int(0.2*len(d_arr_100)):int(0.5*len(d_arr_100))]
        color_std_100.append(round(d_arr_100.std()+0.001, 1))
        color_std_30.append(round(d_arr_30.std()+0.001, 1))
augment_info['color_std_100'] = color_std_100
augment_info['color_std_30'] = color_std_30

#%%
augment_info['brightness_contrast'] = augment_info['color_std_100'] - augment_info['color_std_30']
augment_info.head(3)

# %%
info_table = augment_info[['left_dis', 'down_dis', 'right_dis', 'up_dis', 'brightness_contrast','color_std_100','color_std_30']].describe()
simlar_tag = (round(info_table['brightness_contrast']['75%']/10, 1) * 10)
difference_tag = (round(info_table['brightness_contrast']['25%']/10, 1) * 10)

print('simlar_tag:',simlar_tag) 
print('difference_tag:',difference_tag)

#%%
# >simlar_tag     return 1为降低对比度;
# <difference_tag return 0为增加对比度;
# orthers         return 2为随机。
augment_info['bc_similar'] = 2

def bc_tag_label(row):
    if row.brightness_contrast > simlar_tag:
        return 1
    elif row.brightness_contrast < difference_tag:
        return 0
    else:
        return 2

augment_info['bc_similar'] = augment_info.apply(lambda x:bc_tag_label(x), axis=1)

#%%
print(augment_info[['left_dis', 'down_dis', 'right_dis', 'up_dis']].describe())

crop_tag = 25

#%%
# > crop_tag     return 1为裁剪对象;
# <=crop_tag     return 0为保持对象;

def crop_tag_label(row):
    crop_tag_list = [0,0,0,0]
    if row.left_dis > crop_tag:
        crop_tag_list[0] = 1
    if row.down_dis > crop_tag:
        crop_tag_list[1] = 1
    if row.right_dis > crop_tag:
        crop_tag_list[2] = 1
    if row.up_dis > crop_tag:
        crop_tag_list[3] = 1
    return crop_tag_list

augment_info['crop_tag_list'] = augment_info.apply(lambda x:crop_tag_label(x), axis=1)

def crop_num_label(row):
    crop_num_list = [0,0,0,0]
    if row.crop_tag_list[0] == 1:
        crop_num_list[0] = row.left_dis
    if row.crop_tag_list[1] == 1:
        crop_num_list[1] = row.down_dis
    if row.crop_tag_list[2] == 1:
        crop_num_list[2] = row.right_dis
    if row.crop_tag_list[3] == 1:
        crop_num_list[3] = row.up_dis
    return crop_num_list

augment_info['crop_num_list'] = augment_info.apply(lambda x:crop_num_label(x), axis=1)

augment_info.head(3)

#%%
augment_info[['img_length', 'img_hight']] = augment_info[['img_length', 'img_hight']].astype(int)
# 1538~1647 : 110 * 320  :141     :lk
# 912~1537  : 626 * 60   :1418    :gy
# 0~911     : 912 * 40   :1927    :sy
print('==============lk==============')
print('max img_length:', augment_info[augment_info['num']>=1538]['img_length'].max())
print('max img_hight:', augment_info[augment_info['num']>=1538]['img_hight'].max())
length_limit_lk = 1000
hight_limit_lk = 550
print('==============gy==============')
print('max img_length:', augment_info[(augment_info['num']>=912) & (augment_info['num']<=1537)]['img_length'].max())
print('max img_hight:', augment_info[(augment_info['num']>=912) & (augment_info['num']<=1537)]['img_hight'].max())
length_limit_gy = 700
hight_limit_gy = 300
print('==============sy==============')
print('max img_length:', augment_info[augment_info['num']<=911]['img_length'].max())
print('max img_hight:', augment_info[augment_info['num']<=911]['img_hight'].max())
length_limit_sy = 400
hight_limit_sy = 300

Limit_Length_by_Label_Dict = defaultdict(lambda: defaultdict())
Limit_Length_by_Label_Dict['lk']['length_limit'] = length_limit_lk
Limit_Length_by_Label_Dict['lk']['hight_limit'] = hight_limit_lk
Limit_Length_by_Label_Dict['gy']['length_limit'] = length_limit_gy
Limit_Length_by_Label_Dict['gy']['hight_limit'] = hight_limit_gy
Limit_Length_by_Label_Dict['sy']['length_limit'] = length_limit_sy
Limit_Length_by_Label_Dict['sy']['hight_limit'] = hight_limit_sy

#%%
augment_info['max_ratio'] = 2
augment_info.to_pickle(os.path.join(os.path.abspath('.') + '/observation_file','augment_info.pkl'))


#%%
augment_info['length_limit'] = 0
augment_info['hight_limit'] = 0
for i in range(len(augment_info)):
    _name = augment_info.loc[i]['pic_name']
    _max_ratio = augment_info.loc[i]['max_ratio']
    _label = augment_info.loc[i]['label']
    json_transfer_dict[_name][0]['max_ratio'] = 2
    augment_info.loc[i]['length_limit'] = Limit_Length_by_Label_Dict[_label]['length_limit']
    augment_info.loc[i]['hight_limit'] = Limit_Length_by_Label_Dict[_label]['hight_limit']
    json_transfer_dict[_name][0]['length_limit'] = Limit_Length_by_Label_Dict[_label]['length_limit']
    json_transfer_dict[_name][0]['hight_limit'] = Limit_Length_by_Label_Dict[_label]['hight_limit']

# %%
def random_contrast(img, simlar_tag):
    if simlar_tag == 1:
        delta = np.random.uniform(-0.4, -0.05) + 1
    elif simlar_tag == 0:
        delta = np.random.uniform(0.05, 0.5) + 1
    else:
        prob = np.random.uniform(0, 1)
        if prob < 0.5:
            delta = np.random.uniform(-0.3, -0.05) + 1
        else:
            delta = np.random.uniform(0.05, 0.4) + 1
    img = ImageEnhance.Contrast(img).enhance(delta)
    return img

def random_brightness(img, simlar_tag):
    if simlar_tag == 1:
        delta = np.random.uniform(-0.3, -0.05) + 1
    elif simlar_tag == 0:
        delta = np.random.uniform(0.05, 0.6) + 1
    else:
        prob = np.random.uniform(0, 1)
        if prob < 0.5:
            delta = np.random.uniform(-0.3, -0.05) + 1
        else:
            delta = np.random.uniform(0.05, 0.5) + 1
    # delta = 2
    img = ImageEnhance.Brightness(img).enhance(delta)
    return img


#%%
augment_info[augment_info['img_length']==914]

# %%
def random_crop_expand(img, limits, shapes, boxes, boxes_tag, label_boxes, label_names, max_ratios, expand_range=[0.05,0.2]):
    pass


#%%
pic_name_list = list(p2l_relation_dict.keys())
pic_name_list = pic_name_list[:10]
for _pic_name in pic_name_list:
    with Image.open(os.path.join(PATH, pic_name),'r') as im:
        pass
pic_name = pic_name_list[0]
with Image.open(os.path.join(PATH, pic_name),'r') as im:
    new_im = random_brightness(im,0)
    new_im.save(os.path.join(os.path.abspath('.')+'/imag_test', 'aug_'+pic_name),'JPEG')


def box_crop(boxes, labels, crop, img_shape):
    x, y, w, h = map(float, crop)
    im_w, im_h = map(float, img_shape)

    boxes = boxes.copy()
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (boxes[:, 0] + boxes[:, 2] / 2) * im_w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_box = np.array([x, y, x + w, y + h])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
    boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
    labels = labels * mask.astype('float32')
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 3] - boxes[:, 1]) / h
    

    return boxes, labels, mask.sum()
def random_crop(img, boxes, labels, scales=[0.1, 0.5], max_side_length=2.0, constraints=None, max_trial=50):
    if random.random() > 0.6:
        return img, boxes, labels
    if len(boxes) == 0:
        return img, boxes, labels

    if not constraints:
        constraints = [
                (0.1, 1.0),
                (0.3, 1.0),
                (0.5, 1.0),
                (0.7, 1.0),
                (0.9, 1.0),
                (0.0, 1.0)]

    w, h = img.size
    crops = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), \
                                          min(max_ratio, 1 / scale / scale))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_box = np.array([[
                (crop_x + crop_w / 2.0) / w,
                (crop_y + crop_h / 2.0) / h,
                crop_w / float(w),
                crop_h /float(h)
                ]])

            iou = box_iou_xywh(crop_box, boxes)
            if min_iou <= iou.min() and max_iou >= iou.max():
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        crop_boxes, crop_labels, box_num = box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:
            continue
        img = img.crop((crop[0], crop[1], crop[0] + crop[2], 
                        crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        return img, crop_boxes, crop_labels
    return img, boxes, labels




#%%
def random_expand(img, gtboxes, keep_ratio=True):
    if np.random.uniform(0, 1) < train_parameters['image_distort_strategy']['expand_prob']:
        return img, gtboxes

    max_ratio = train_parameters['image_distort_strategy']['expand_max_ratio']    
    w, h = img.size
    c = 3
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow -w)
    off_y = random.randint(0, oh -h)

    out_img = np.zeros((oh, ow, c), np.uint8)
    for i in range(c):
        out_img[:, :, i] = train_parameters['mean_rgb'][i]

    out_img[off_y: off_y + h, off_x: off_x + w, :] = img
    gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
    gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
    gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

    return Image.fromarray(out_img), gtboxes
