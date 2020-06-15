# coding:utf-8
# implement YOLO
import tensorflow as tf
from src import module
import numpy as np
slim = tf.contrib.slim

class YOLO():
    def __init__(self,class_num, anchors, width=608, height=608):
        self.class_num = class_num
        self.anchors = np.asarray(anchors).reshape([-1, 3, 2])
        self.width = width
        self.height = height
        pass

    def forward(self, inputs, batch_norm_decay=0.9, weight_decay=0.0005, isTrain=True, reuse=False):
        # set batch norm params
        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': isTrain,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            # [N, 19, 19, 512], [N, 38, 38, 256], [N, 76, 76, 128]
            route_1, route_2, route_3 = module.extraction_feature(inputs, batch_norm_params, weight_decay)
            
            with slim.arg_scope([slim.conv2d], 
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                with tf.variable_scope('yolo'):
                    # features of y1
                    # [N, 76, 76, 128] => [N, 76, 76, 256]
                    net = module.conv(route_1, 256)
                    # [N, 76, 76, 256] => [N, 76, 76, 255]
                    net = slim.conv2d(net, 3*(4+1+self.class_num), 1,
                                                        stride=1, normalizer_fn=None,
                                                        activation_fn=None, biases_initializer=tf.zeros_initializer())  # ,scope="feature_y3"
                    feature_y3 = net    # yolo/Conv_1/BiasAdd:0

                    # features of  y2
                    # [N, 76, 76, 128] => [N, 38, 38, 256]
                    net = module.conv(route_1, 256, stride=2)
                    # [N, 38, 38, 512]
                    net = tf.concat([net, route_2], -1)
                    net = module.yolo_conv_block(net, 512, 2, 1)
                    route_147 = net
                    # [N, 38, 38, 256] => [N, 38, 38, 512]
                    net = module.conv(net, 512)
                    # [N, 38, 38, 512] => [N, 38, 38, 255]
                    net = slim.conv2d(net, 3*(4+1+self.class_num), 1,
                                                        stride=1, normalizer_fn=None,
                                                        activation_fn=None, biases_initializer=tf.zeros_initializer())  # , scope="feature_y2"
                    feature_y2 = net    # yolo/Conv_9/BiasAdd:0

                    # features of  y3
                    # [N, 38, 38, 256] => [N, 19, 19, 512]
                    net = module.conv(route_147, 512, stride=2)
                    net = tf.concat([net, route_3], -1)
                    net = module.yolo_conv_block(net, 1024, 3, 0)
                    # [N, 19, 19, 1024] => [N, 19, 19, 255]
                    net = slim.conv2d(net, 3*(4+1+self.class_num), 1,
                                                        stride=1, normalizer_fn=None,
                                                        activation_fn=None, biases_initializer=tf.zeros_initializer())  #, scope="feature_y1"
                    feature_y1 = net    # yolo/Conv_17/BiasAdd:0

        return feature_y1, feature_y2, feature_y3

    # IOU, GIOU
    def IOU(self, pre_xy, pre_wh, valid_yi_true):
        '''
            pre_xy : [13, 13, 3, 2]
            pre_wh : [13, 13, 3, 2]
            valid_yi_true : [V, 5 + class_num] or [V, 4]
            return:
                iou, giou : [13, 13, 3, V]
        '''

        # [13, 13, 3, 2] ==> [13, 13, 3, 1, 2]
        pre_xy = tf.expand_dims(pre_xy, -2)
        pre_wh = tf.expand_dims(pre_wh, -2)

        # [V, 2]
        yi_true_xy = valid_yi_true[..., 0:2]
        yi_true_wh = valid_yi_true[..., 2:4]

        # left top of Intersection : [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
        intersection_left_top = tf.maximum((pre_xy - pre_wh / 2), (yi_true_xy - yi_true_wh / 2))
        # right bottom of Intersection
        intersection_right_bottom = tf.minimum((pre_xy + pre_wh / 2), (yi_true_xy + yi_true_wh / 2))
        # left top of Union 
        combine_left_top = tf.minimum((pre_xy - pre_wh / 2), (yi_true_xy - yi_true_wh / 2))
        # right bottom of Union
        combine_right_bottom = tf.maximum((pre_xy + pre_wh / 2), (yi_true_xy + yi_true_wh / 2))

        # height and width of intersection [13, 13, 3, V, 2] == > [13, 13, 3, V, 2]
        intersection_wh = tf.maximum(intersection_right_bottom - intersection_left_top, 0.0)
        # width and height of union
        combine_wh = tf.maximum(combine_right_bottom - combine_left_top, 0.0)
        
        # area of intersection : [13, 13, 3, V]
        intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]
        # area of prediction box  : [13, 13, 3, 1]
        pre_area = pre_wh[..., 0] * pre_wh[..., 1]
        # area of truth box : [V]
        true_area = yi_true_wh[..., 0] * yi_true_wh[..., 1]
        # [V] ==> [1, V]
        true_area = tf.expand_dims(true_area, axis=0)
        # iou : [13, 13, 3, V]
        iou = intersection_area / (pre_area + true_area - intersection_area + 1e-10)    # avoid to divide zero

        # area of union : [13, 13, 3, V, 1] ==> [13, 13, 3, V] 
        combine_area = combine_wh[..., 0] * combine_wh[..., 1]
        # giou : [13, 13, 3, V]
        giou = (intersection_area+1e-10) / combine_area # add a number to guarantee  giou is exists
        
        return iou, giou

    # compute CIOU loss
    def __my_CIOU_loss(self, pre_xy, pre_wh, yi_box):
        '''
        the formula of CIOU_LOSS is refers to http://bbs.cvmart.net/topics/1436
        pre_xy:[batch_size, 13, 13, 3, 2]
        pre_wh:[batch_size, 13, 13, 3, 2]
        yi_box:[batch_size, 13, 13, 3, 4]
        return:[batch_size, 13, 13, 3, 1]
        '''
        # [batch_size, 13, 13, 3, 2]
        yi_true_xy = yi_box[..., 0:2]
        yi_true_wh = yi_box[..., 2:4]

        # top dowm left right
        pre_lt = pre_xy - pre_wh/2
        pre_rb = pre_xy + pre_wh/2
        truth_lt = yi_true_xy - yi_true_wh / 2
        truth_rb = yi_true_xy + yi_true_wh / 2

        # left top of intersection : [batch_size, 13, 13, 3,2]
        intersection_left_top = tf.maximum(pre_lt, truth_lt)
        intersection_right_bottom = tf.minimum(pre_rb, truth_rb)
        # width and height of intersection : [batch_size, 13, 13, 3, 2]
        intersection_wh = tf.maximum(intersection_right_bottom - intersection_left_top, 0.0)
        # area of intersection : [batch_size, 13, 13, 3, 1]
        intersection_area = intersection_wh[..., 0:1] * intersection_wh[..., 1:2]
        # left top of union 
        combine_left_top = tf.minimum(pre_lt, truth_lt)
        # right bottom of union
        combine_right_bottom = tf.maximum(pre_rb, truth_rb)
        # width and height of union
        combine_wh = tf.maximum(combine_right_bottom - combine_left_top, 0.0)

        # diagonal line of union : [batch_size, 13, 13, 3, 1]
        C = tf.square(combine_wh[..., 0:1]) + tf.square(combine_wh[..., 1:2])
        # diagonal line  of center point:[batch_size, 13, 13, 3, 1]
        D = tf.square(yi_true_xy[..., 0:1] - pre_xy[..., 0:1]) + tf.square(yi_true_xy[..., 1:2] - pre_xy[..., 1:2])

        # area of box : [batch_size, 13, 13, 3, 1]
        pre_area = pre_wh[..., 0:1] * pre_wh[..., 1:2]
        true_area = yi_true_wh[..., 0:1] * yi_true_wh[..., 1:2]

        # iou : [batch_size, 13, 13, 3, 1]
        iou = intersection_area / (pre_area + true_area - intersection_area)

        pi = 3.14159265358979323846

        # [batch_size, 13, 13, 3, 1]
        v = 4 / (pi * pi) * tf.square( 
                                    tf.subtract(
                                        tf.math.atan(yi_true_wh[..., 0:1] / yi_true_wh[..., 1:2]),
                                        tf.math.atan(pre_wh[...,0:1] / pre_wh[..., 1:2])
                                        )
                                    )

        # trade-off 
        # alpha
        alpha = v / (1.0 - iou + v)
        ciou_loss = 1.0 - iou + D / C +  alpha * v
        return ciou_loss

    # get low iou place between truth and prediction box
    def __get_low_iou_mask(self, pre_xy, pre_wh, yi_true, use_iou=True, ignore_thresh=0.5):
        '''
        pre_xy:[batch_size, 13, 13, 3, 2]
        pre_wh:[batch_size, 13, 13, 3, 2]
        yi_true:[batch_size, 13, 13, 3, 5+class_num]
        use_iou:use iou as meters
        ignore_thresh:thresh of iou or giou
        return: [batch_size, 13, 13, 3, 1]
        return a mask where iou or giou lower than thresh
        '''
        # confidence:[batch_size, 13, 13, 3, 1]
        conf_yi_true = yi_true[..., 4:5]

        # mask of low iou
        low_iou_mask = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        # batch_size
        N = tf.shape(yi_true)[0]
        
        def loop_cond(index, low_iou_mask):
            return tf.less(index, N)        
        def loop_body(index, low_iou_mask):
            # truth of all label : [13, 13, 3, class_num+5] & [13, 13, 3, 1] == > [V, class_num + 5]
            valid_yi_true = tf.boolean_mask(yi_true[index], tf.cast(conf_yi_true[index, ..., 0], tf.bool))
            # compute iou/ giou : [13, 13, 3, V]
            iou, giou = self.IOU(pre_xy[index], pre_wh[index], valid_yi_true)

            # [13, 13, 3]
            if use_iou:
                best_giou = tf.reduce_max(iou, axis=-1)
            else:
                best_giou = tf.reduce_max(giou, axis=-1)
            # [13, 13, 3]
            low_iou_mask_tmp = best_giou < ignore_thresh
            # [13, 13, 3, 1]
            low_iou_mask_tmp = tf.expand_dims(low_iou_mask_tmp, -1)
            # write
            low_iou_mask = low_iou_mask.write(index, low_iou_mask_tmp)
            return index+1, low_iou_mask

        _, low_iou_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, low_iou_mask])
        # stack:[batch_size, 13, 13, 3, 1]
        low_iou_mask = low_iou_mask.stack()
        return low_iou_mask

    # get mask of low possibility area
    def __get_low_prob_mask(self, prob, prob_thresh=0.25):
        '''
        prob:[batch_size, 13, 13, 3, class_num]
        prob_thresh:thresh
        return: bool [batch_size, 13, 13, 3, 1]
        return a mask where possibility is all lower than thresh
        '''
        # [batch_size, 13, 13, 3, 1]
        max_prob = tf.reduce_max(prob, axis=-1, keepdims=True)
        low_prob_mask = max_prob < prob_thresh        
        return low_prob_mask

    def __decode_feature(self, yi_pred, curr_anchors):
        '''
        yi_pred:[batch_size, 13, 13, 3 * (class_num + 5)]
        curr_anchors:[3,2], truth value
        return:
            xy:[batch_size, 13, 13, 3, 2], float
            wh:[batch_size, 13, 13, 3, 2], float
            conf:[batch_size, 13, 13, 3, 1]
            prob:[batch_size, 13, 13, 3, class_num]
        '''
        shape = tf.shape(yi_pred) 
        shape = tf.cast(shape, tf.float32)
        # [batch_size, 13, 13, 3, class_num + 5]
        yi_pred = tf.reshape(yi_pred, [shape[0], shape[1], shape[2], 3, 5 + self.class_num])
        # yi_pred = tf.reshape(yi_pred, [shape[0], shape[1], shape[2], 1, 5 + self.class_num])
        # shape : [batch_size,13,13,3,2] [batch_size,13,13,3,2] [batch_size,13,13,3,1] [batch_size,13,13,3, class_num]
        xy, wh, conf, prob = tf.split(yi_pred, [2, 2, 1, self.class_num], axis=-1)

        ''' compute offset of x and y ''' 
        offset_x = tf.range(shape[2], dtype=tf.float32) #width
        offset_y = tf.range(shape[1], dtype=tf.float32) # height
        offset_x, offset_y = tf.meshgrid(offset_x, offset_y)
        offset_x = tf.reshape(offset_x, (-1, 1))
        offset_y = tf.reshape(offset_y, (-1, 1))
        offset_xy = tf.concat([offset_x, offset_y], axis=-1)
        # [13, 13, 1, 2]
        offset_xy = tf.reshape(offset_xy, [shape[1], shape[2], 1, 2])
        
        xy = tf.math.sigmoid(xy) + offset_xy    
        xy = xy / [shape[2], shape[1]]

        wh = tf.math.exp(wh) * curr_anchors
        wh = wh / [self.width, self.height]

        return xy, wh, conf, prob

    # compute loss of yolov4
    def __compute_loss_v4(self, xy, wh, conf, prob, yi_true, cls_normalizer=1.0, ignore_thresh=0.5, 
                                                                prob_thresh=0.25, score_thresh=0.25, iou_normalizer=0.07):
        '''
        xy:[batch_size, 13, 13, 3, 2]
        wh:[batch_size, 13, 13, 3, 2]
        conf:[batch_size, 13, 13, 3, 1]
        prob:[batch_size, 13, 13, 3, class_num]
        yi_true:[batch_size, 13, 13, 3, class_num]
        return: total loss

        xy_loss: loss of x and y
        wh_loss: loss of width and height
        conf_loss: loss of confidence
        class_loss: loss of possibility
        '''
        # mask of low iou 
        low_iou_mask = self.__get_low_iou_mask(xy, wh, yi_true, ignore_thresh=ignore_thresh)
        # mask of low prob
        low_prob_mask = self.__get_low_prob_mask(prob, prob_thresh=prob_thresh)        
        # mask of low iou or low prob
        low_iou_prob_mask = tf.math.logical_or(low_iou_mask, low_prob_mask)
        low_iou_prob_mask = tf.cast(low_iou_prob_mask, tf.float32)

        # batch_size
        N = tf.shape(xy)[0]
        N = tf.cast(N, tf.float32)

        # [batch_size, 13, 13, 3, 1]
        conf_scale = wh[..., 0:1] * wh[..., 1:2]
        conf_scale = tf.where(tf.math.greater(conf_scale, 0),
                                                        tf.math.sqrt(conf_scale), conf_scale)
        conf_scale = conf_scale * cls_normalizer                                                        
        conf_scale = tf.math.square(conf_scale)
        # [batch_size, 13, 13, 3, 1]
        no_obj_mask = 1.0 - yi_true[..., 4:5]
        conf_loss_no_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                            labels=yi_true[:,:,:,:,4:5], logits=conf
                                                            ) * conf_scale * no_obj_mask * low_iou_prob_mask
        # [batch_size, 13, 13, 3, 1]
        obj_mask = yi_true[..., 4:5]        
        conf_loss_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                            labels=yi_true[:,:,:,:,4:5], logits=conf
                                                            ) * np.square(cls_normalizer) * obj_mask        
        # loss of confidence
        conf_loss = conf_loss_obj + conf_loss_no_obj
        conf_loss = tf.clip_by_value(conf_loss, 0.0, 1e3)
        conf_loss = tf.reduce_sum(conf_loss) / N

        # ciou_loss
        yi_true_ciou = tf.where(tf.math.less(yi_true[..., 0:4], 1e-10),
                                                tf.ones_like(yi_true[..., 0:4]), yi_true[..., 0:4])
        pre_xy = tf.where(tf.math.less(xy, 1e-10),
                                                tf.ones_like(xy), xy)
        pre_wh = tf.where(tf.math.less(wh, 1e-10),
                                                tf.ones_like(wh), wh)
        ciou_loss = self.__my_CIOU_loss(pre_xy, pre_wh, yi_true_ciou)
        ciou_loss = tf.where(tf.math.greater(obj_mask, 0.5), ciou_loss, tf.zeros_like(ciou_loss))
        ciou_loss = tf.square(ciou_loss * obj_mask) * iou_normalizer
        ciou_loss = tf.clip_by_value(ciou_loss, 0, 1e3)
        ciou_loss = tf.reduce_sum(ciou_loss) / N
        ciou_loss = tf.clip_by_value(ciou_loss, 0, 1e4)

        # loss of x and y
        xy = tf.clip_by_value(xy, 1e-10, 1e4)
        xy_loss = obj_mask * tf.square(yi_true[..., 0: 2] - xy)
        xy_loss = tf.clip_by_value(xy_loss, 0.0, 1e3)
        xy_loss = tf.reduce_sum(xy_loss) / N
        xy_loss = tf.clip_by_value(xy_loss, 0.0, 1e4)

        # loss of w and h
        wh_y_true = tf.where(condition=tf.math.less(yi_true[..., 2:4], 1e-10),
                                        x=tf.ones_like(yi_true[..., 2: 4]), y=yi_true[..., 2: 4])
        wh_y_pred = tf.where(condition=tf.math.less(wh, 1e-10),
                                        x=tf.ones_like(wh), y=wh)
        wh_y_true = tf.clip_by_value(wh_y_true, 1e-10, 1e10)
        wh_y_pred = tf.clip_by_value(wh_y_pred, 1e-10, 1e10)
        wh_y_true = tf.math.log(wh_y_true)
        wh_y_pred = tf.math.log(wh_y_pred)

        wh_loss = obj_mask * tf.square(wh_y_true - wh_y_pred)
        wh_loss = tf.clip_by_value(wh_loss, 0.0, 1e3)
        wh_loss = tf.reduce_sum(wh_loss) / N
        wh_loss = tf.clip_by_value(wh_loss, 0.0, 1e4)
        
        # loss of possibility
        # [batch_size, 13, 13, 3, class_num]
        prob_score = prob * conf
        
        high_score_mask = prob_score > score_thresh
        high_score_mask = tf.cast(high_score_mask, tf.float32)
        
        class_loss_no_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                        labels=yi_true[...,5:5+self.class_num],
                                                        logits=prob 
                                                    ) * low_iou_prob_mask * no_obj_mask * high_score_mask
        
        class_loss_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                        labels=yi_true[...,5:5+self.class_num],
                                                        logits=prob
                                                    ) * obj_mask

        class_loss = class_loss_no_obj + class_loss_obj        
        class_loss = tf.clip_by_value(class_loss, 0.0, 1e3)
        class_loss = tf.reduce_sum(class_loss) / N

        loss_total = xy_loss + wh_loss + conf_loss + class_loss + ciou_loss
        return loss_total

    # get loss of yolov4
    def get_loss_v4(self, feature_y1, feature_y2, feature_y3, y1_true, y2_true, y3_true, cls_normalizer=1.0, ignore_thresh=0.5, prob_thresh=0.25, score_thresh=0.25):
        '''
        feature_y1:[batch_size, 13, 13, 3*(5+class_num)]
        feature_y2:[batch_size, 26, 26, 3*(5+class_num)]
        feature_y3:[batch_size, 52, 52, 3*(5+class_num)]
        y1_true: label of y1
        y2_true: label of y2
        y3_true: label of y2
        return:total_loss
        '''
        # y1
        xy, wh, conf, prob = self.__decode_feature(feature_y1, self.anchors[2])
        loss_y1 = self.__compute_loss_v4(xy, wh, conf, prob, y1_true, cls_normalizer=1.0, 
                                                                                    ignore_thresh=ignore_thresh, prob_thresh=prob_thresh, score_thresh=score_thresh)

        # y2
        xy, wh, conf, prob = self.__decode_feature(feature_y2, self.anchors[1])
        loss_y2 = self.__compute_loss_v4(xy, wh, conf, prob, y2_true, cls_normalizer=1.0, 
                                                                                    ignore_thresh=ignore_thresh, prob_thresh=prob_thresh, score_thresh=score_thresh)

        # y3
        xy, wh, conf, prob = self.__decode_feature(feature_y3, self.anchors[0])
        loss_y3 = self.__compute_loss_v4(xy, wh, conf, prob, y3_true, cls_normalizer=1.0, 
                                                                                    ignore_thresh=ignore_thresh, prob_thresh=prob_thresh, score_thresh=score_thresh)

        return loss_y1 + loss_y2 + loss_y3

    # NMS
    def __nms(self, boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_threshold=0.5):
        '''
        boxes:[1, V, 4]
        score:[1, V, class_num]
        return:????
            boxes:[V, 4]
            score:[V,]
        '''
        boxes_list, label_list, score_list = [], [], []
        max_boxes = tf.constant(max_boxes, dtype='int32')

        # [V, 4]
        boxes = tf.reshape(boxes, [-1, 4])
        # [V, class_num]
        score = tf.reshape(scores, [-1, num_classes])

        mask = tf.greater_equal(score, tf.constant(score_thresh))
        for i in range(num_classes):
            # Step 3: Apply the mask to scores, boxes and pick them out
            filter_boxes = tf.boolean_mask(boxes, mask[:,i])
            filter_score = tf.boolean_mask(score[:,i], mask[:,i])
            nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                    scores=filter_score,
                                                    max_output_size=max_boxes,
                                                    iou_threshold=iou_threshold, name='nms_indices')
            label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
            boxes_list.append(tf.gather(filter_boxes, nms_indices))
            score_list.append(tf.gather(filter_score, nms_indices))

        # stack
        boxes = tf.concat(boxes_list, axis=0)
        score = tf.concat(score_list, axis=0)
        label = tf.concat(label_list, axis=0)

        return boxes, score, label

    def __get_pred_box(self, feature_y1, feature_y2, feature_y3):
        '''
        feature_y1:[1, 13, 13, 3*(class_num + 5)]
        feature_y1:[1, 26, 52, 3*(class_num + 5)]
        feature_y1:[1, 26, 52, 3*(class_num + 5)]
        return:
            boxes:[1, V, 4]:[x_min, y_min, x_max, y_max] float
            conf:[1, V, 1]
            prob:[1, V, class_num]
        '''
        # decode y1
        xy1, wh1, conf1, prob1 = self.__decode_feature(feature_y1, self.anchors[2])
        conf1, prob1 = tf.sigmoid(conf1), tf.sigmoid(prob1)

        # decode y2
        xy2, wh2, conf2, prob2 = self.__decode_feature(feature_y2, self.anchors[1])
        conf2, prob2 = tf.sigmoid(conf2), tf.sigmoid(prob2)

        # decode y3
        xy3, wh3, conf3, prob3 = self.__decode_feature(feature_y3, self.anchors[0])
        conf3, prob3 = tf.sigmoid(conf3), tf.sigmoid(prob3)

        def _reshape(xy, wh, conf, prob):
            # [1, 13, 13, 3, 1]
            x_min = xy[..., 0: 1] - wh[..., 0: 1] / 2.0
            x_max = xy[..., 0: 1] + wh[..., 0: 1] / 2.0
            y_min = xy[..., 1: 2] - wh[..., 1: 2] / 2.0
            y_max = xy[..., 1: 2] + wh[..., 1: 2] / 2.0

            # [1, 13, 13, 3, 4]
            boxes = tf.concat([x_min, y_min, x_max, y_max], -1)
            shape = tf.shape(boxes)
            # [1, 13*13*3, 4]
            boxes = tf.reshape(boxes, (shape[0], shape[1] * shape[2]* shape[3], shape[4]))

            # [1, 13 * 13 * 3, 1]
            conf = tf.reshape(conf, (shape[0], shape[1] * shape[2]* shape[3], 1))

            # [1, 13 * 13 * 3, class_num]
            prob = tf.reshape(prob, (shape[0], shape[1] * shape[2]* shape[3], -1))
        
            return boxes, conf, prob

        # reshape
        # [batch_size, 13*13*3, 4], [batch_size, 13*13*3, 1], [batch_size, 13*13*3, class_num]
        boxes_y1, conf_y1, prob_y1 = _reshape(xy1, wh1, conf1, prob1)
        boxes_y2, conf_y2, prob_y2 = _reshape(xy2, wh2, conf2, prob2)
        boxes_y3, conf_y3, prob_y3 = _reshape(xy3, wh3, conf3, prob3)

        # stack
        # [1, 13*13*3, 4] & [1, 26*26*3, 4] & [1, 52*52*3, 4] ==> [1,  V, 4]
        boxes = tf.concat([boxes_y1, boxes_y2, boxes_y3], 1)
        conf = tf.concat([conf_y1, conf_y2, conf_y3], 1)
        prob = tf.concat([prob_y1, prob_y2, prob_y3], 1)

        return boxes, conf, prob

    # get prediction result
    def get_predict_result(self, feature_y1, feature_y2, feature_y3, class_num, score_thresh=0.5, iou_thresh=0.5, max_box=200):
        '''
        feature_y1:[batch_size, 13, 13, 3*(class_num+5)]
        feature_y2:[batch_size, 26, 26, 3*(class_num+5)]
        feature_y3:[batch_size, 52, 52, 3*(class_num+5)]
        class_num: classify number
        return:
            boxes:[V, 4] include [x_min, y_min, x_max, y_max]
            score:[V, 1]
            label:[V, 1]
        '''
        boxes, conf, prob = self.__get_pred_box(feature_y1, feature_y2, feature_y3)
        pre_score = conf * prob
        boxes, score, label = self.__nms(boxes, pre_score, class_num, max_boxes=max_box, score_thresh=score_thresh, iou_threshold=iou_thresh)
        return boxes, score, label
        # return conf, prob, tf.constant(1)

