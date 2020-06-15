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
#######################
#    down             #
#left                 #
#                     #
#                     #
#               right #
#             up      #
#######################
label_analysis_df.sort_values(by='num', inplace=True)
label_analysis_df.reset_index(drop=True,inplace=True)
print(label_analysis_df.head(3))

#%%
def area_get(row):
    area = (row.right-row.left)*(row.up-row.down)
    return area
label_analysis_df['area'] = label_analysis_df.apply(lambda x:area_get(x), axis=1)

#%%
## 20*20 多扫描一点
## 10*10 次之
## 40*40 再次之
%matplotlib inline 
label_analysis_df[label_analysis_df['area']==43824]         # 43824 max
label_analysis_df[label_analysis_df['area']==9]             # 1120  min
print(label_analysis_df[label_analysis_df['area']<=1600].shape[0]/label_analysis_df.shape[0])
label_analysis_df[label_analysis_df['area']<=1600][['area']].hist()

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
        json_transfer_dict[row.pic_name][0]['bc_similar'] = 1
        return 2
    elif row.brightness_contrast < difference_tag:
        json_transfer_dict[row.pic_name][0]['bc_similar'] = 0
        return 2
    else:
        json_transfer_dict[row.pic_name][0]['bc_similar'] = 2
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
        print('dark_contrast')
        delta = np.random.uniform(-0.4, -0.05) + 1
    elif simlar_tag == 0:
        print('light_contrast')
        delta = np.random.uniform(0.05, 0.5) + 1
    else:
        prob = np.random.uniform(0, 1)
        if prob < 0.5:
            print('dark_contrast')
            delta = np.random.uniform(-0.3, -0.05) + 1
        else:
            print('light_contrast')
            delta = np.random.uniform(0.05, 0.4) + 1
    img = ImageEnhance.Contrast(img).enhance(delta)
    img = ImageEnhance.Brightness(img).enhance(delta)
    return img

# def random_brightness(img, simlar_tag):
#     if simlar_tag == 1:
#         print('dark_brightness')
#         delta = np.random.uniform(-0.3, -0.05) + 1
#     elif simlar_tag == 0:
#         print('light_brightness')
#         delta = np.random.uniform(0.05, 0.6) + 1
#     else:
#         prob = np.random.uniform(0, 1)
#         if prob < 0.5:
#             print('dark_brightness')
#             delta = np.random.uniform(-0.3, -0.05) + 1
#         else:
#             print('light_brightness')
#             delta = np.random.uniform(0.05, 0.5) + 1
#     # delta = 2
#     img = ImageEnhance.Brightness(img).enhance(delta)
#     return img


#%%
augment_info[augment_info['img_length']==914]

# %%
def save_file(path, item):
    
    # 先将字典对象转化为可写入文本的字符串
    item = json.dumps(item)

    try:
        with open(path, "w", encoding='utf-8') as f:
            f.write(item)
            print("^_^ write success")
    except Exception as e:
        print("write error==>", e)
            
def random_crop_expand(img, limits, shapes, boxes, boxes_tag, label_boxes, label_names, max_ratios, similar_tag, gen_num=0, expand_range=[-0.15, 0.3]):
    """
    :param img 图片对象
    :param limits [400, 300]      图片最长尺度
    :param shapes [219, 250]      图片尺寸
    :param boxes [101, 0, 0, 0]   可切割长度
    :param boxes_tag [1, 0, 0, 0] 可变化的边
    :param label_boxes [((101, 1), (209, 248))]  标签的标记框
    :param label_names ['sy']     图片的标签
    :param max_ratios             最大可扩展尺寸
    """
    feasible_crop_tag = np.nonzero(boxes_tag)[0]
    crop_num = random.randrange(1,len(feasible_crop_tag)+1)
    print(crop_num,feasible_crop_tag)
    crop_index_list = random.sample(set(feasible_crop_tag), crop_num)
    print('raw shapes:',shapes)
    _crop_left = 0
    _crop_down = 0
    _crop_right = shapes[0]
    _crop_up = shapes[1]
    if random.random() < 0.6:
        print('raw label_boxes:',label_boxes)
        for i in crop_index_list:
            max_crop_size = boxes[i]
            crop_size = random.randrange(crop_tag, max_crop_size)
            print('crop_size:', crop_size)
            if i == 0:
                _crop_left += crop_size
            elif i == 2:
                _crop_right -= crop_size
            elif i == 3:
                _crop_up -= crop_size
            else:
                _crop_down += crop_size
            for _label_box in label_boxes:
                if i == 0:
                    _label_box[0] = _label_box[0] - crop_size
                    _label_box[2] = _label_box[2] - crop_size
                elif i == 1:
                    _label_box[1] = _label_box[1] - crop_size
                    _label_box[3] = _label_box[3] - crop_size
                else:
                    pass
        print('change label_boxes:',label_boxes)
        box = [_crop_left, _crop_down, _crop_right, _crop_up]
        img_crop = img.crop(box)
    else:
        print('label_boxes not change')
        img_crop = copy.deepcopy(img)
    crop_length, crop_hight = img_crop.size
    expand_coef = random.uniform(expand_range[0],expand_range[1])
    if expand_coef > -0.05 and expand_coef < 0:
        expand_coef = -0.05
    elif expand_coef < 0.05 and expand_coef >= 0:
        expand_coef = 0.05
    expand_coef = min([limits[0]/crop_length - 1, limits[1]/crop_hight - 1, expand_coef])
    crop_length = int(crop_length * (1+expand_coef))
    crop_hight = int(crop_hight * (1+expand_coef))
    for _label_box in label_boxes:
        _label_box[0] = int(_label_box[0] * (1+expand_coef))
        _label_box[1] = int(_label_box[1] * (1+expand_coef))
        _label_box[2] = int(_label_box[2] * (1+expand_coef))
        _label_box[3] = int(_label_box[3] * (1+expand_coef))
        if _label_box[2] - _label_box[0] <= 1:
            _label_box[2] += 1
        if _label_box[3] - _label_box[1] <= 1:
            _label_box[3] += 1
    print('expand_coef:',expand_coef)
    print('change2 label_boxes:', label_boxes)
    img_expand = img_crop.resize((crop_length, crop_hight))
    if random.random() < 0.5:
        img_expand = img_expand.transpose(Image.FLIP_LEFT_RIGHT)
        for _label_box in label_boxes:
            _label_box[2], _label_box[0] = crop_length - _label_box[0], crop_length - _label_box[2]
    print('similar_tag:',similar_tag)
    print('size:', crop_length, crop_hight)
    new_im = random_contrast(img_expand, similar_tag)
    new_im.save(os.path.join(os.path.abspath('.')+'/imag_test', 'aug_'+str(gen_num)+'_'+_pic_name),'JPEG')
    json_dict = defaultdict(list)
    for i in range(len(label_boxes)):
        sub_dict = dict()
        sub_dict['points']=[[label_boxes[i][0],label_boxes[i][1]],[label_boxes[i][2],label_boxes[i][3]]]
        sub_dict['label']=label_names[i]
        json_dict['shapes'].append(sub_dict)
        
    save_file(os.path.join(os.path.abspath('.')+'/imag_test', 'aug_'+str(gen_num)+'_'+_pic_name[:-4]+'json'),json_dict)
    pass

#%%
pic_name_list = list(p2l_relation_dict.keys())
pic_name_list = pic_name_list[:2]
gene_num_param = 0
for _pic_name in pic_name_list:
    with Image.open(os.path.join(PATH, _pic_name),'r') as im:
        total_index = json_transfer_dict[_pic_name][0]['label_total_num']
        shapes_param = [json_transfer_dict[_pic_name][0]['img_length'], json_transfer_dict[_pic_name][0]['img_hight']]
        similar_tag_param = json_transfer_dict[_pic_name][0]['bc_similar']
        _series = augment_info[augment_info['pic_name']==_pic_name]
        boxes_param = _series['crop_num_list'].tolist()[0]
        boxes_tag_parm = _series['crop_tag_list'].tolist()[0]
        max_ratios_param = _series['max_ratio'].tolist()[0]
        label_names_list = []
        label_boxes_list = []
        print('===========================', total_index)
        for _index in range(total_index):
            label_names_param = json_transfer_dict[_pic_name][_index]['label']
            label_boxes_param = [json_transfer_dict[_pic_name][_index]['left'],json_transfer_dict[_pic_name][_index]['down'],json_transfer_dict[_pic_name][_index]['right'],json_transfer_dict[_pic_name][_index]['up']]
            label_names_list.append(label_names_param)
            label_boxes_list.append(label_boxes_param)
        limits_param = [Limit_Length_by_Label_Dict[label_names_param]['length_limit'],Limit_Length_by_Label_Dict[label_names_param]['hight_limit']]
        random_crop_expand(im,limits_param,shapes_param,boxes_param,boxes_tag_parm,label_boxes_list,label_names_list,max_ratios_param,similar_tag_param,gene_num_param)
        gene_num_param += 1
        pass
    


# %%
augment_info[['img_length', 'img_hight']] = augment_info[['img_length', 'img_hight']].astype(int)
# 1538~1647 : 110 * 320  :141     :lk
# 912~1537  : 626 * 60   :1418    :gy
# 0~911     : 912 * 40   :1927    :sy

#%%
augment_info.label_total_num.value_counts()

# %%
# 929 466
# lk 1643
lk_mul_label = augment_info[(augment_info['label_total_num'] >= 2) & (augment_info['label'] == 'lk')]['num'].tolist()
gy_mul_label = augment_info[(augment_info['label_total_num'] >= 3) & (augment_info['label'] == 'gy')]['num'].tolist()
sy_mul_label = augment_info[(augment_info['label_total_num'] >= 3) & (augment_info['label'] == 'sy')]['num'].tolist()

# %%
gene_num_param = 0
# while gene_num_param <= 20000:
while gene_num_param <= 2:
    if random.random()<0.15:
        _pic_num = random.choice(gy_mul_label)
    else:
        _pic_num = random.randint(1538,1647)
    _pic_name = 'train_r'+str(_pic_num)+'.jpeg'
    with Image.open(os.path.join(PATH, _pic_name),'r') as im:
        total_index = json_transfer_dict[_pic_name][0]['label_total_num']
        shapes_param = [json_transfer_dict[_pic_name][0]['img_length'], json_transfer_dict[_pic_name][0]['img_hight']]
        similar_tag_param = json_transfer_dict[_pic_name][0]['bc_similar']
        _series = augment_info[augment_info['pic_name']==_pic_name]
        boxes_param = _series['crop_num_list'].tolist()[0]
        boxes_tag_parm = _series['crop_tag_list'].tolist()[0]
        max_ratios_param = _series['max_ratio'].tolist()[0]
        label_names_list = []
        label_boxes_list = []
        print('===========================', total_index)
        for _index in range(total_index):
            label_names_param = json_transfer_dict[_pic_name][_index]['label']
            label_boxes_param = [json_transfer_dict[_pic_name][_index]['left'],json_transfer_dict[_pic_name][_index]['down'],json_transfer_dict[_pic_name][_index]['right'],json_transfer_dict[_pic_name][_index]['up']]
            label_names_list.append(label_names_param)
            label_boxes_list.append(label_boxes_param)
        limits_param = [Limit_Length_by_Label_Dict[label_names_param]['length_limit'],Limit_Length_by_Label_Dict[label_names_param]['hight_limit']]
        random_crop_expand(im,limits_param,shapes_param,boxes_param,boxes_tag_parm,label_boxes_list,label_names_list,max_ratios_param,similar_tag_param,gene_num_param)
        gene_num_param += 1
        pass
    
gene_num_param = 0
# while gene_num_param <= 30000:
while gene_num_param <= 3:
    if random.random()<0.15:
        _pic_num = random.choice(lk_mul_label)
    else:
        _pic_num = random.randint(912,1537)
    _pic_name = 'train_r'+str(_pic_num)+'.jpeg'
    with Image.open(os.path.join(PATH, _pic_name),'r') as im:
        total_index = json_transfer_dict[_pic_name][0]['label_total_num']
        shapes_param = [json_transfer_dict[_pic_name][0]['img_length'], json_transfer_dict[_pic_name][0]['img_hight']]
        similar_tag_param = json_transfer_dict[_pic_name][0]['bc_similar']
        _series = augment_info[augment_info['pic_name']==_pic_name]
        boxes_param = _series['crop_num_list'].tolist()[0]
        boxes_tag_parm = _series['crop_tag_list'].tolist()[0]
        max_ratios_param = _series['max_ratio'].tolist()[0]
        label_names_list = []
        label_boxes_list = []
        print('===========================', total_index)
        for _index in range(total_index):
            label_names_param = json_transfer_dict[_pic_name][_index]['label']
            label_boxes_param = [json_transfer_dict[_pic_name][_index]['left'],json_transfer_dict[_pic_name][_index]['down'],json_transfer_dict[_pic_name][_index]['right'],json_transfer_dict[_pic_name][_index]['up']]
            label_names_list.append(label_names_param)
            label_boxes_list.append(label_boxes_param)
        limits_param = [Limit_Length_by_Label_Dict[label_names_param]['length_limit'],Limit_Length_by_Label_Dict[label_names_param]['hight_limit']]
        random_crop_expand(im,limits_param,shapes_param,boxes_param,boxes_tag_parm,label_boxes_list,label_names_list,max_ratios_param,similar_tag_param,gene_num_param)
        gene_num_param += 1
        pass
    
gene_num_param = 0
# while gene_num_param <= 30000:
while gene_num_param <= 3:
    if random.random()<0.15:
        _pic_num = random.choice(sy_mul_label)
    else:
        _pic_num = random.randint(0,911)
    _pic_name = 'train_r'+str(_pic_num)+'.jpeg'
    with Image.open(os.path.join(PATH, _pic_name),'r') as im:
        total_index = json_transfer_dict[_pic_name][0]['label_total_num']
        shapes_param = [json_transfer_dict[_pic_name][0]['img_length'], json_transfer_dict[_pic_name][0]['img_hight']]
        similar_tag_param = json_transfer_dict[_pic_name][0]['bc_similar']
        _series = augment_info[augment_info['pic_name']==_pic_name]
        boxes_param = _series['crop_num_list'].tolist()[0]
        boxes_tag_parm = _series['crop_tag_list'].tolist()[0]
        max_ratios_param = _series['max_ratio'].tolist()[0]
        label_names_list = []
        label_boxes_list = []
        print('===========================', total_index)
        for _index in range(total_index):
            label_names_param = json_transfer_dict[_pic_name][_index]['label']
            label_boxes_param = [json_transfer_dict[_pic_name][_index]['left'],json_transfer_dict[_pic_name][_index]['down'],json_transfer_dict[_pic_name][_index]['right'],json_transfer_dict[_pic_name][_index]['up']]
            label_names_list.append(label_names_param)
            label_boxes_list.append(label_boxes_param)
        limits_param = [Limit_Length_by_Label_Dict[label_names_param]['length_limit'],Limit_Length_by_Label_Dict[label_names_param]['hight_limit']]
        random_crop_expand(im,limits_param,shapes_param,boxes_param,boxes_tag_parm,label_boxes_list,label_names_list,max_ratios_param,similar_tag_param,gene_num_param)
        gene_num_param += 1
        pass

# %%
