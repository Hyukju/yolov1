import cv2
import os 
import numpy as  np 

IMG_FILE_EXTS = ['.png', '.jpg', '.tif', '.tiff', '.bmp']

def load_dataset(img_dir, resize_shape=(224,224), num_classes=1, num_bboxes=2):
    # resize_shape = (width, height)
    # class info : num_bboxes * 5(= p, x, y, wi, h ) + num_classes
    length_cls_info = num_bboxes * 5 + num_classes

    x_train = [] 
    y_train = []

    # load file lists
    file_list = os.listdir(img_dir)
    img_file_list = [os.path.join(img_dir, x) for x in file_list if os.path.splitext(x)[1].lower() in IMG_FILE_EXTS]

    # check image file 과 label file 수 확인 
    total = len(img_file_list)

    x_train = np.zeros((total, resize_shape[1], resize_shape[0], 3), dtype='float32')    
    y_train = np.zeros((total, 7,7, length_cls_info), dtype='float32')    
    # load image files 
    for i, img_file in enumerate(img_file_list):
        # decompose filename and extention
        filename, _ = os.path.splitext(img_file)        
        label_file = filename + '.txt'
       
        # read image, resize and bgr->rgb
        img = resize_image(img_file, resize_shape)
        # load labels
        label_list = read_label_file(label_file)
    
        # convert labels -> yolo features = ?x7x7x(5xnum_bboxes + num_classes) 
        yolo_feature = np.zeros((7,7,length_cls_info), dtype='float32')
        yolo_feature = convert_bbox_info_to_yolo_feature(yolo_feature, label_list)

        x_train[i] = img
        y_train[i] = yolo_feature
        
    return x_train, y_train

def read_label_file(label_file):
    bbox_info_list = []

    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            bbox_info = list(map(float, line.split()))
            bbox_info_list.append(bbox_info)

    return bbox_info_list

def resize_image(filename, resize_shape=(224,224)):
    img = cv2.imread(filename).astype('float32') / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=resize_shape)
    return img

# def resize_image(filename, resize_shape=(224,224)):
#     img = cv2.imread(filename, 0).astype('float32') / 255.0
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     img = cv2.resize(img, dsize=resize_shape)
#     return img


def convert_bbox_info_to_yolo_feature(yolo_feature, label_list):
    for label in label_list:
        i_x = int(label[1] * 6)
        i_y = int(label[2] * 6)
        x = label[1] * 6 - i_x
        y = label[2] * 6- i_y
        w = label[3]
        h = label[4]
        class_id = label[0]
        bbox = [x, y, w, h]
        # bbox info
        yolo_feature[i_y,i_x, 4] = 1.0
        yolo_feature[i_y,i_x,:4] = bbox
        # class info 
        yolo_feature[i_y,i_x, 10 + int(class_id)] = 1.

    return yolo_feature
