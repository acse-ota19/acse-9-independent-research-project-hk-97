from keras.layers import Input, Dense, Reshape, Dropout,Convolution2D, Conv2DTranspose, UpSampling2D
from keras.layers import BatchNormalization, Activation, Flatten, MaxPooling1D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils
import keras.backend as K
import numpy as np

class GAN():
    def __init__(self, rows):
        self.gan = None
        
        ### Generator
        self.generator = None
        self.generator_optimizer = Adam(0.0002, 0.9)
        
        ### Discriminator
        self.discriminator = None
        self.discriminator_optimizer = SGD(lr=0.012)
        
        
        self.channels = 3
        self.length = rows
        self.shape = (self.channels, self.length)
        self.latent_dim = 100
        self.disc_loss = []
        self.disc_loss2 = []
        self.gen_loss =[]
        

        ### Build network
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.gan = self.build_gan()

        
    def build_gan(self):
        self.discriminator.trainable = False
        z = Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)
        validity = self.discriminator(generated_seq)

        model = Model(z, validity)
        model.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer)
        
        return model
        

    def build_discriminator(self):
        num_train, height, width, channel = train_data.shape
        
        ## MTCNN
        kernel_size_1 = 5
        kernel_size_2 = 3
        pool_size_1 = 2
        pool_size_2 = 3  
        conv_depth_1 = 50 
        conv_depth_2 = 40 
        conv_depth_3 = 20 
        drop_prob_1 = 0.2 
        drop_prob_2 = 0.4 
        hidden_size = 400 

        model = Sequential()
        model.add(Convolution2D(conv_depth_1, (1 , kernel_size_1), padding='valid', activation='relu', input_shape=(height, width,1)))
        model.add(Convolution2D(conv_depth_1, (1 , kernel_size_2), padding='same', activation='relu'))
        model.add(Dense(int(conv_depth_1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, pool_size_1)))
        model.add(Dropout(drop_prob_1))

        model.add(Convolution2D(conv_depth_2, (1 , kernel_size_1), padding='valid', activation='relu'))
        model.add(Dense(conv_depth_2, activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, pool_size_2)))
        model.add(Dropout(drop_prob_1))

        model.add(Convolution2D(conv_depth_3, (1 , kernel_size_2), padding='valid', activation='relu'))
        model.add(Dropout(drop_prob_1))

        model.add(Flatten())
        model.add(Dense((hidden_size), activation='relu'))
        model.add(Dropout(drop_prob_2))

        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer, metrics=['accuracy'])
  

        return model
      
    def build_generator(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.latent_dim,))
        model.add(Activation('tanh'))
        model.add(Dense(128*3*20))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((3, 20, 128), input_shape=(128*7*7,)))
        model.add(UpSampling2D(size=(1, 5)))
        model.add(Convolution2D(64, (1, 3), padding='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(1,2)))
        model.add(Convolution2D(1, (1, 5), padding='same'))
        model.add(Activation('tanh'))
        return model

        
    def generate(self):
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        predictions = self.generator.predict(noise)
        return predictions
