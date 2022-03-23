import cv2 
import colorsys
import random
import numpy as np 

def random_color():  
    h = random.uniform(0, 1)
    hsv = [h, 0.9, 1]
    color = colorsys.hsv_to_rgb(*hsv)
    color = [int(x * 255) for x in color]
    return color

def do_iou(bbox_info1, bbox_info2):
    # box = (x, y, w, h) -> x1, y1, x2, y2
    box1_x1 = bbox_info1['x'] - bbox_info1['width'] / 2.
    box1_y1 = bbox_info1['y'] - bbox_info1['height'] / 2.
    box1_x2 = bbox_info1['x'] + bbox_info1['width'] / 2.
    box1_y2 = bbox_info1['y'] + bbox_info1['height'] / 2.

    box2_x1 = bbox_info2['x'] - bbox_info2['width'] / 2.
    box2_y1 = bbox_info2['y'] - bbox_info2['height'] / 2.
    box2_x2 = bbox_info2['x'] + bbox_info2['width'] / 2.
    box2_y2 = bbox_info2['y'] + bbox_info2['height'] / 2.

    
    box1_area = bbox_info1['width'] * bbox_info1['height']
    box2_area = bbox_info2['width'] * bbox_info2['height']


    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    # compute the width and height of the intersection
    w = max(0., x2 - x1)
    h = max(0., y2 - y1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

# nms
def do_nms(bbox_info_list, iou_th):
    nms = [] 

    total = len(bbox_info_list)

    for i in range(total):
        bbox_info_i = bbox_info_list[i]
        discard = False

        for j in range(i + 1, total):           
            bbox_info_j = bbox_info_list[j]

            if do_iou(bbox_info_i, bbox_info_j) > iou_th:
                if bbox_info_j['confidence'] > bbox_info_i['confidence']:
                    discard = True
        if discard == False:
            nms.append(bbox_info_i)
    return nms

def draw_rectangle(img, bbox_info_list):
    out = img.copy()

    rows, cols = out.shape[:2]
    for i, bbox_info in enumerate(bbox_info_list):
        x = bbox_info['x']
        y = bbox_info['y']
        w = bbox_info['width']
        h = bbox_info['height']
        confidence = bbox_info['confidence']
        cl = bbox_info['class']

        box_color = np.array(random_color())/255.0

        pt1 = (int((x - w/2) * cols), int((y - h/2) * rows))
        pt2 = (int((x + w/2) * cols), int((y + h/2) * rows))
        # print(f'{i}:: {x * cols:.2f}, {y*rows:.2f}, {w*cols:.2f}, {h*rows:.2f}, {cl}')
        cv2.rectangle(out, pt1, pt2, box_color, 1)
        cv2.putText(out, f'{cl}:{confidence:.2f}', (pt1[0], pt1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

    return out 

def get_bbox_info_list_from_predict_single_image(predict_result, labels=None):
    # 이미지 하나에 대한 결과만 도출 함 

    features = predict_result
    bbox_info_list = []
    
    for j in range(2):

        if j == 0:
            confidence = features[...,4]
            bbox = features[...,0:4]
        else:
            confidence = features[...,9]
            bbox = features[...,5:9]
    
        cl_list = features[...,10:]         
    
        for m in range(7):
            for n in range(7):
                if confidence[m,n] > 0.5:
                    bbox_info = dict()
                    # labels 가 없으면 숫자로 표시
                    if labels == None:
                        bbox_info['class'] = np.argmax(cl_list[m,n])
                    else:
                        bbox_info['class'] = labels[np.argmax(cl_list[m,n])]
                    # yolo feature 결과는 각 m,n을 기준으로 0 ~ 1 사이로 표시되어 있음 
                    # 이 값을 0 ~ 7 값의 범위로 변환 (yolov1 feature 7x7)
                    bbox_info['x'] = (bbox[m,n,0] + n) /7.
                    bbox_info['y'] = (bbox[m,n,1] + m) /7.
                    bbox_info['width'] = bbox[m,n,2] 
                    bbox_info['height'] = bbox[m,n,3] 
                    bbox_info['confidence'] = confidence[m,n]
                    bbox_info_list.append(bbox_info)

    return bbox_info_list

