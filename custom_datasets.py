import cv2 
import numpy as np 
import colorsys
import random
import os 
import argparse

def random_color():  
    h = random.uniform(0, 1)
    hsv = [h, 0.9, 1]
    color = colorsys.hsv_to_rgb(*hsv)
    color = [int(x * 255) for x in color]
    return color

def write_yolo_label_txt(filename, yolo_bboxes):
    with open(filename, 'w') as f:
        for bbox in yolo_bboxes:
            str_bbox = list(map(str, bbox))
            str_ = ' '.join(str_bbox) + '\n'
            f.write(str_)

def write_random_text_images(text_list, class_name='class_image', image_size=(448,448), num_images=10, max_objects=10, use_debug=False):
    
    rows, cols = image_size

    fontFace=cv2.FONT_HERSHEY_SIMPLEX 

    text_shuffle_list = text_list.copy()

    os.makedirs(class_name, exist_ok=True)

    for i in range(num_images):
        img = np.zeros((rows, cols, 3), dtype='uint8')
        yolo_bboxes = []
        # randint => [low, high)
        num_objects = np.random.randint(1, max_objects + 1)
        for _ in range(num_objects):
            # shuffle 
            random.shuffle(text_shuffle_list)
            # random text color and size
            text = text_shuffle_list[0]
            fontScale = np.random.randint(8,16) / 10.  
            thickness = np.random.randint(2,5)
            color = random_color() 
            # get text size
            text_size, base_line =cv2.getTextSize(text, fontFace,fontScale,thickness)
            text_width, text_height = text_size
            org_x = np.random.randint(0, cols - text_width)
            org_y = np.random.randint(text_height, rows-text_height)
            # yolo bboxes info (class, x, y, width, height)
            class_id = text_list.index(text)
            yolo_x = round((org_x + text_width/2.0) / cols, 4)
            yolo_y = round((org_y + base_line - (text_height + base_line)/2.0) /rows, 4)
            yolo_w = round(text_width / cols, 4)
            yolo_h = round((text_height + base_line) / rows, 4)
            yolo_bbox = [class_id, yolo_x, yolo_y, yolo_w, yolo_h]
            yolo_bboxes.append(yolo_bbox)
            # draw text 
            cv2.putText(img,text,(org_x, org_y),fontFace,fontScale,color,thickness)

        img_filename = os.path.join(class_name, f'{class_name}-{i:04d}.png')
        txt_filename = os.path.join(class_name, f'{class_name}-{i:04d}.txt')
        cv2.imwrite(img_filename, img)
        write_yolo_label_txt(txt_filename, yolo_bboxes)

        if use_debug:
            for bbox in yolo_bboxes:
                class_id, x, y, w, h = bbox
                pt1 = (int((x - w/2) * cols), int((y - h/2) * rows))
                pt2 = (int((x + w/2) * cols), int((y + h/2) * rows))
                cv2.rectangle(img, pt1, pt2, (0,200,100), 1)
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__=='__main__':
    # How to use
    # python custom_datasets.py --ojbect dog,cat,duck --image_size 448 --num_image 100 --max_objects 10

    parser = argparse.ArgumentParser(description='Make coustom datasets')
    
    parser.add_argument('--objects', type=str, default='dog,cat,duck', help='text1, text2, text3, ...')
    parser.add_argument('--image_size', type=int, default=448, help='image size (with = height)')
    parser.add_argument('--num_images', type=int, default=100, help='number of images created')
    parser.add_argument('--max_objects', type=int, default=5, help='maximum objects in image')

    args = parser.parse_args()
    objects = args.objects.split(',')
    image_size = args.image_size
    num_images = args.num_images
    max_objects = args.max_objects
    class_name = '_'.join(objects)

    print('objects =', objects)
    print('class name =', class_name)
    print('image size =', (image_size, image_size))
    print('num images =', num_images)
    print('max objects =', max_objects)

    write_random_text_images(objects, class_name=class_name, image_size=(image_size,image_size), num_images=num_images, max_objects=max_objects, use_debug=False)    

