from keras import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, LeakyReLU
from keras.models import Model, Sequential
from keras.applications import VGG16

def load_data():
    pass 

class YOLOV1():

    def __init__(self) -> None:            
        self.width = 448
        self.height = 448
        self.channel = 3

    
    def build_model_vgg16(self):
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(self.width, self.height, self.channel))
        base_model.trainable = False 
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(7*7*30))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Reshape((7,7,30)))
        base_model.summary()
        model.summary()
        model.compile(optimizer='adam', loss=self.yolo_loss, metirc=['acc'])
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

        x = Flatten()(x)
        x = Dense(4096)(x)
        x = LeakyReLU(alpha=0.1)(x) 
        x = Dense(7*7*30)(x)
        x = LeakyReLU(alpha=0.1)(x)

        outputs = Reshape((7,7,30))(x)

        model = Model(inputs, outputs)
        model.summary()

        model.compile(optimizer='adam', loss=self.yolo_loss)

        return model

         

    def yolo_loss(self, y_pred, y_true):
        print(y_pred)
        pass 

    def train(self):
        # load data
        # train 
        pass 

if __name__=='__main__':
    model = YOLOV1()
    model.build_model_vgg16()