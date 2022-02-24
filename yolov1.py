from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, LeakyReLU, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import VGG16, EfficientNetB1
from tensorflow.keras.optimizers import Adam
from data_loader import load_dataset
from loss import yolo_loss

import pandas as pd
import os 

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

    def train(self, train_img_dir, valid_img_dir, batch_size, epochs, weight_filename):
       
       # load data
        x_train, y_train = load_dataset(train_img_dir, (self.width, self.height), yolo_feature_size=self.S, num_classes=self.C)
        x_valid, y_valid = load_dataset(valid_img_dir, (self.width, self.height), yolo_feature_size=self.S, num_classes=self.C)

        print('x_train.shape: ', x_train.shape)
        print('y_train.shape: ', y_train.shape)

        # train 
        model = self.build_model()

         # callback function 
        callbacks_list = [ModelCheckpoint(filepath=f'./weights/{weight_filename}.h5',
                        monitor='val_loss',
                        save_best_only=True,
                        save_weight_only=True,
                        ), 
                        ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.1,
                            patience=30,                            
                        )]

        history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=epochs, callbacks=callbacks_list) 
        
        os.makedirs('./weights/', exist_ok=True)

        # convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(history.history) 

        # or save to csv: 
        hist_csv_file = f'./weights/{weight_filename}_history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        model.save_weights( f'./weights/{weight_filename}.h5')

    def test(self, weight_file, test_img_dir, treshold=0.5):
        test_images, _ = load_dataset(test_img_dir, (self.width, self.height), yolo_feature_size=self.S, num_classes=self.C)
        model = self.build_model()
        model.load_weights(weight_file)

        predict = model.predict(test_images)

        # nms 
        pass 



if __name__=='__main__':
    train_img_dir = 'D:\\projects_test\\yolov1\\datasets\\dog_cat_duck\\train'
    valid_img_dir = 'D:\\projects_test\\yolov1\\datasets\\dog_cat_duck\\valid'

    model = YOLOV1(num_classes=3)
    model.train(train_img_dir, valid_img_dir, 64,100, '111')