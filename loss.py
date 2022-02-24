import tensorflow as tf
import tensorflow.keras.backend as K

def iou(box1, box2):
    # box = (x, y, w, h) -> x1, y1, x2, y2
    box1_x1 = box1[...,0] - box1[...,2] / 2.
    box1_y1 = box1[...,1] - box1[...,3] / 2.
    box1_x2 = box1[...,0] + box1[...,2] / 2.
    box1_y2 = box1[...,1] + box1[...,3] / 2.

    box2_x1 = box2[...,0] - box2[...,2] / 2.
    box2_y1 = box2[...,1] - box2[...,3] / 2.
    box2_x2 = box2[...,0] + box2[...,2] / 2.
    box2_y2 = box2[...,1] + box2[...,3] / 2.

    box1_area = box1[...,2] * box1[...,3]
    box2_area = box2[...,2] * box2[...,3]

    # obtain x1, y1, x2, y2 of the intersection
    x1 = tf.maximum(box1_x1, box2_x1)
    y1 = tf.maximum(box1_y1, box2_y1)
    x2 = tf.minimum(box1_x2, box2_x2)
    y2 = tf.minimum(box1_y2, box2_y2)

    # compute the width and height of the intersection
    w = tf.maximum(0., x2 - x1)
    h = tf.maximum(0., y2 - y1)


    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def yolo_loss(y_true, y_pred):

    gt_bbox = tf.identity(y_true[...,0:4])
    gt_conf = tf.identity(y_true[...,4])
    gt_class_prob = tf.identity(y_true[...,10:])

    pred_bbox1 = tf.identity(y_pred[...,0:4])
    pred_conf1 = tf.identity(y_pred[...,4])

    pred_bbox2 = tf.identity(y_pred[...,5:9])
    pred_conf2 = tf.identity(y_pred[...,9])
    
    pred_class_prob = tf.identity(y_pred[...,10:])

    # iou 계산 
    iou1 = iou(gt_bbox, pred_bbox1) #?x7x7
    iou2 = iou(gt_bbox, pred_bbox2) #?x7x7

    # responsible 
    best_iou1_region = tf.cast(iou1 > iou2, dtype='float32') #?x7x7
    best_iou2_region = 1. - best_iou1_region

    # confidence 가 1인 위치 표시 
    obj = tf.cast(gt_conf == 1., dtype='float32')

    gt_x = gt_bbox[...,0]
    gt_y = gt_bbox[...,1]
    gt_w = gt_bbox[...,2]
    gt_h = gt_bbox[...,3]

    pred_x1 = pred_bbox1[...,0]
    pred_y1 = pred_bbox1[...,1]
    pred_w1 = pred_bbox1[...,2]
    pred_h1 = pred_bbox1[...,3]

    pred_x2 = pred_bbox2[...,0]
    pred_y2 = pred_bbox2[...,1]
    pred_w2 = pred_bbox2[...,2]
    pred_h2 = pred_bbox2[...,3]


    lambda_coord = 5.
    lambda_noobj = 0.5

    loss_xy = lambda_coord * K.sum( obj * best_iou1_region * K.square(gt_x - pred_x1) + 
                                    obj * best_iou2_region * K.square(gt_x - pred_x2) +
                                    obj * best_iou1_region * K.square(gt_y - pred_y1) +
                                    obj * best_iou2_region * K.square(gt_y - pred_y2))

    loss_wh = lambda_coord * K.sum( obj * best_iou1_region * K.square(K.sqrt(gt_w) - K.sqrt(pred_w1)) + 
                                    obj * best_iou2_region * K.square(K.sqrt(gt_w) - K.sqrt(pred_w2)) +
                                    obj * best_iou1_region * K.square(K.sqrt(gt_h) - K.sqrt(pred_h1)) +
                                    obj * best_iou2_region * K.square(K.sqrt(gt_h) - K.sqrt(pred_h2)))
    
    loss_conf_obj = K.sum( obj * best_iou1_region * K.square(1.0 - pred_conf1) + 
                           obj * best_iou2_region * K.square(1.0 - pred_conf2)) 

    loss_conf_noobj = lambda_noobj * K.sum( (1 - obj * best_iou1_region) * K.square(0. - pred_conf1) + 
                                            (1 - obj * best_iou2_region) * K.square(0. - pred_conf2)) 

    loss_clss_prob = K.sum ( tf.expand_dims(obj, axis=-1) * K.square(gt_class_prob - pred_class_prob))
    
    f_loss = loss_xy + loss_wh + loss_conf_obj + loss_conf_noobj + loss_clss_prob
    
    return f_loss
