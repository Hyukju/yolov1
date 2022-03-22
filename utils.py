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
        print('bbox_info_i', i, bbox_info_i)
        discard = False

        for j in range(i + 1, total):           
            bbox_info_j = bbox_info_list[j]
            print('bbox_info_j', j, bbox_info_j)

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