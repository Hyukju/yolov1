from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, LeakyReLU, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from data_loader import load_dataset
from loss import yolo_loss
import pandas as pd
import os 
import utils
import numpy as np 
import cv2 

class YOLOV1():

    def __init__(self, num_classes) -> None:            
        self.width = 448
        self.height = 448
        self.channel = 3
        self.learning_rate = 1e-4
        self.S = 7
        self.B = 2
        self.C = num_classes
        self.LENGTH_BBOX_INFO = self.B * 5 + self.C
        # 
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.weight_dir = os.path.join(self.current_dir, 'weights')
        os.makedirs(self.weight_dir, exist_ok=True)
        #
        self.epoch = 0
    
    def build_model_vgg16(self):
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(self.width, self.height, self.channel))
        base_model.trainable = False 
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(496))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.5))
        model.add(Dense(self.S*self.S*(self.LENGTH_BBOX_INFO)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Reshape((self.S,self.S,self.LENGTH_BBOX_INFO)))
        base_model.summary()
        model.summary()
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=yolo_loss, metrics=['acc'])
        return model

    def build_model(self):
        inputs = Input(shape=(self.width, self.height, self.channel))
        x = Conv2D(64, (7,7), strides=(2,2), padding='same')(inputs)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2,2), strides=2)(x)
        
        x = Conv2D(192, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2,2), strides=2)(x)

        x = Conv2D(128, (1,1), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(256, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(256, (1,1), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(512, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2,2), strides=2)(x)

        for _ in range(4):
            x = Conv2D(256, (1,1), padding='same')(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Conv2D(512, (3,3), padding='same')(x)        
            x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(512, (1,1), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(1024, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2,2), strides=2, padding='same')(x)

        for _ in range(2):
            x = Conv2D(512, (1,1), padding='same')(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Conv2D(1024, (3,3), padding='same')(x)        
            x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(1024, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(1024, (3,3), strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x) 

        x = Conv2D(1024, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(1024, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.5)(x)

        x = Conv2D(256, (1,1), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.5)(x)

        x = Conv2D(30, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.5)(x)

        outputs = Conv2D(self.LENGTH_BBOX_INFO, (3,3), padding='same')(x)
         
        model = Model(inputs, outputs)
        model.summary()

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=yolo_loss, metrics=['acc'])

        return model

    def train(self, train_img_dir, valid_img_dir=None, batch_size=20, epochs=100, weight_name='yolov1'):
        
        save_weight_dir = os.path.join(self.weight_dir, weight_name)
        os.makedirs(save_weight_dir)

        #load datasets
        x_train, y_train, _ = load_dataset(train_img_dir, (self.width, self.height), yolo_feature_size=self.S, num_classes=self.C)
        if valid_img_dir == None:
            validation_data = None
        else:
            x_valid, y_valid, _ = load_dataset(valid_img_dir, (self.width, self.height), yolo_feature_size=self.S, num_classes=self.C)
            validation_data = (x_valid, y_valid)


        print('x_train.shape: ', x_train.shape)
        print('y_train.shape: ', y_train.shape)

        # train 
        model = self.build_model()

        # callback function 
        weight_path = os.path.join(save_weight_dir, f'{weight_name}_*epoch*.h5')

        weight_path = weight_path.replace('*epoch*','{epoch:04d}_{val_loss:0.2f}')

        callbacks_list = [ModelCheckpoint(filepath=weight_path,
                        monitor='val_loss',
                        save_best_only=True,
                        save_weight_only=True,
                        ), 
                        ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.1,
                            patience=30,                            
                        ),
                        ]

        history = model.fit(x_train, y_train, 
                            initial_epoch=self.epoch,
                            validation_data=validation_data,
                            batch_size=batch_size, 
                            epochs=epochs, 
                            callbacks=callbacks_list) 

        # convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(history.history) 
        # or save to csv: 
        hist_csv_file = os.path.join(save_weight_dir, f'{weight_name}_history.csv')
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)


    def test(self, weight_file, test_img_dir, save_dir, labels=None, nms_iou_th=0.5, benchmark_iou_th=0.5):
        # resize image 
        test_images, _, img_flie_list = load_dataset(test_img_dir, (self.width, self.height), yolo_feature_size=self.S, num_classes=self.C)
        model = self.build_model()
        model.load_weights(weight_file)

        os.makedirs(save_dir, exist_ok=True)

        # 
        total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0 

        for test_image, file_path in zip(test_images, img_flie_list):
            _, file_name = os.path.split(file_path)

            # resized image
            predict = model.predict(np.expand_dims(test_image, axis=0))
            bbox_info_list = utils.get_bbox_info_list_from_predict_single_image(predict_result=predict[0], labels=labels)
            nms = utils.do_nms(bbox_info_list=bbox_info_list, iou_th=nms_iou_th)

            # original szie image
            img = cv2.imread(file_path)
            out = utils.draw_rectangle(img, nms)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            save_file_name = f'result_{file_name}'
            save_path = os.path.join(save_dir, save_file_name)

            cv2.imwrite(save_path, out)


            # benchmark
            label_file = file_name[:-4] + '.txt'
            label_file_path = os.path.join(test_img_dir, label_file)
            
            # read true bbox info list from label file
            true_bbox_info_list = utils.get_bbox_info_list_from_label_file(label_file_path)
            TP, FP, TN, FN  = utils.benchmark(true_bbox_info_list=true_bbox_info_list, pred_bbox_info_list=nms, labels=labels, iou_th=benchmark_iou_th)

            total_TP += TP
            total_FP += FP
            total_TN += TN
            total_FN += FN

            print(f'{file_path}: TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}')

        precison = total_TP / (total_TP + total_FP)
        recall =  total_TP / (total_TP + total_FN)
        accuracy = (total_TP + total_TN) / (total_TP + total_FP + total_TN + total_FN)

        print('precision: ', precison)
        print('recall: ', recall)
        print('accuracy: ', accuracy)

if __name__=='__main__':
    
  
    mode = 'test'

    model = YOLOV1(num_classes=3)

    if mode == 'train':
        train_img_dir = '.\\datasets\\dog_cat_duck\\train'
        valid_img_dir = '.\\datasets\\dog_cat_duck\\valid'  
        model.train(train_img_dir, valid_img_dir, 10,10, 'dog_cat_duck_2')
    elif mode == 'test':
        weight = '.\\weights\\dog_cat_duck_2.h5'
        test_img_dir = '.\\datasets\\dog_cat_duck\\test' 
        save_dir = '.\\result\\'
        labels = ['dog', 'cat', 'duck']
        model.test(weight, test_img_dir, save_dir, labels)

    elif mode == 'benchmark':
        weight = '.\\weights\\dog_cat_duck_2.h5'
        test_img_dir = '.\\datasets\\dog_cat_duck\\test' 
        model.benchmark(weight, test_img_dir)


