import tensorflow as tf

BATCH_SZIE = 2

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
    w = tf.maximum(0., x2 - x1 + 1.)
    h = tf.maximum(0., y2 - y1 + 1.)


    inter = w * h
    iou = inter / (box1_area + box2_area - inter)

    return iou

def my_loss(y_true, y_pred):

    gt_bbox = tf.identity(y_true[...,0:4])
    gt_conf = tf.identity(y_true[...,4])
    gt_class_prob = tf.identity(y_true[...,10:])

    pred_bbox1 = tf.identity(y_pred[...,0:4])
    pred_conf1 = tf.identity(y_pred[...,4])

    pred_bbox2 = tf.identity(y_pred[...,5:9])
    pred_conf2 = tf.identity(y_pred[...,9])
    
    pred_class_prob = tf.identity(y_pred[...,10:])

    # iou 계산 
    iou1 = iou(gt_bbox, pred_bbox1)
    iou2 = iou(gt_bbox, pred_bbox2)
    
    # iou 디멘젼 확장 후 동일한 값으로 복사 : bbox 정보 복사를 위해서 사용
    iou1_4 = tf.expand_dims(iou1, axis=-1)
    iou1_4 = tf.repeat(iou1_4, repeats=4, axis=-1)
    iou2_4 = tf.expand_dims(iou2, axis=-1)
    iou2_4 = tf.repeat(iou2_4, repeats=4, axis=-1)
    
    # 텐서는 슬라이싱 복사 안됨 pred_bbox[iou1>iou2] =  pred_bbox1[iou1>iou2] 
    # tf.where을 사용하여 복사 
    # iou1.shape = 2x3x3, pred_bbox1.shape = 2x3x3x4로 shape가 다르면 tf.where이 동작하지 않기 때문에 iou1_4 형태로 dim 확장 후 복사 
    pred_bbox = tf.where(tf.greater_equal(iou1_4,iou2_4), pred_bbox1, pred_bbox2)
    pred_conf = tf.where(tf.greater_equal(iou1,iou2), pred_conf1, pred_conf2)
    
    # confidence 가 1인 위치 표시 
    coord = gt_conf == 1.
    # confidence 가 0인 위치 표시
    noobj = gt_conf == 0.

    gt_x = gt_bbox[coord][...,0]
    gt_y = gt_bbox[coord][...,1]
    gt_w = gt_bbox[coord][...,2]
    gt_h = gt_bbox[coord][...,3]
    gt_conf_coord = gt_conf[coord]
    gt_conf_noobj = gt_conf[noobj]
    gt_class_prob_coord = gt_class_prob[coord]

    pred_x = pred_bbox[coord][...,0]
    pred_y = pred_bbox[coord][...,1]
    pred_w = pred_bbox[coord][...,2]
    pred_h = pred_bbox[coord][...,3]
    pred_conf_coord = pred_conf[coord]
    pred_conf_noobj = pred_conf[noobj]
    pred_class_prob_coord = pred_class_prob[coord]

    lambda_coord = 5.
    lambda_noobj = 0.5
    
    loss_xy_coord = lambda_coord * tf.reduce_sum(tf.square(gt_x - pred_x) + tf.square(gt_y - pred_y))
    # tf.sqrt 계산 시 tf.maximum 부분 제외하면 nan으로 계산되는 오류 있음
    loss_wh_coord = lambda_coord * tf.reduce_sum(tf.square(tf.sqrt(tf.maximum(gt_w, 1e-9)) - tf.sqrt(tf.maximum(pred_w, 1e-9))) + tf.square(tf.sqrt(tf.maximum(gt_h, 1e-9)) - tf.sqrt(tf.maximum(pred_h, 1e-9))))
    loss_conf_coord = tf.reduce_sum(tf.square(gt_conf_coord - pred_conf_coord))
    loss_conf_noobj = lambda_noobj * tf.reduce_sum(tf.square(gt_conf_noobj - pred_conf_noobj))      
    loss_class_prob_coord = tf.reduce_sum(tf.square(gt_class_prob_coord - pred_class_prob_coord))

    f_loss = loss_xy_coord + loss_wh_coord + loss_conf_coord + loss_conf_noobj + loss_class_prob_coord

    # tf.print('\ncoord = ', coord)
    # tf.print('\n gt_x = ', gt_x)
    # tf.print('\n gt_y = ', pred_x)

    # tf.print('\loss_xy_coord =', loss_xy_coord)
    # tf.print('loss_wh_coord = ', loss_wh_coord)
    # tf.print('loss_conf_coord = ', loss_conf_coord)    
    # tf.print('loss_conf_noobj = ', loss_conf_noobj)
    # tf.print('loss_class_prob_coord = ', loss_class_prob_coord)

    return f_loss
