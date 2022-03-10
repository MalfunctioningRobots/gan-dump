import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
import sys

import os

class DCGAN(): #api for making dcgans (deep convolutive GANs)
    def __init__(self, img_data): #img_data: keras dataset (from image directory)
        #self.discriminator = define_discriminator()
        #self.generator = define_generator()
        
        self.generator_optimizer = Adam(learning_rate=0.0001)
        self.discrim_optimizer = Adam(learning_rate=0.0001)
        
        self.img_data = img_data
        
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        
        self.x_rec_discrim_loss = []
        self.y_rec_discrim_loss = []
        
        self.x_rec_generator_loss = []
        self.y_rec_generator_loss = []
    
    #TODO: MORE FLEXIBILITY IN MODELS
    
    def define_discriminator(self, in_shape=(256,256,3)): #3 color channels
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=in_shape, data_format="channels_last"))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))

        self.discriminator = model
        self.discriminator.compile(optimizer=self.discrim_optimizer)
        #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

   # discriminator = define_discriminator()

    def define_generator(self, in_shape=(500,)):
        model = tf.keras.Sequential()
        model.add(layers.Dense(64*64*3, use_bias=False, input_shape=(500,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((64, 64, 3)))
        #assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', use_bias=False))
        #assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', use_bias=False))
        #assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (3,3), strides=(1,1), padding='same', use_bias=False))
        #assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (3,3), strides=(1,1), padding='same', use_bias=False, activation='sigmoid'))
        #assert model.output_shape == (None, 28, 28, 1)

        self.generator = model
        self.generator.compile(optimizer=self.generator_optimizer)
    
    def discrim_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_img_output):
        return self.cross_entropy(tf.ones_like(fake_img_output), fake_img_output)

    def train_step(self, real_img_batch, iteration, showimg):
        #plt.imshow(real_img_batch[0].numpy()[0]/255)
        #plt.show()
        #print(iteration)
        noise = np.random.normal(size=[len(real_img_batch), 500])
        #print(noise.shape)

        with tf.GradientTape() as discrim_tape1, tf.GradientTape() as discrim_tape2, tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)

            real_img_output = self.discriminator(real_img_batch.reshape((len(real_img_batch), 256, 256, 3)), training=True)
            fake_img_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_img_output)

            discrim_loss_1 = self.discrim_loss(real_img_output, fake_img_output)

        self.x_rec_discrim_loss.append(discrim_loss_1)
        self.y_rec_discrim_loss.append(iteration)
            #discrim_loss_2 = discrim_loss(, np.ones(fake_img_output.shape))

        #if iteration %10 == 0:
        #showimg.set_data(generated_images[0])
        #plt.draw()
        print("generator loss:", gen_loss.numpy(), " discriminator loss:", discrim_loss_1.numpy(), end='\r')
        sys.stdout.flush()

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        discrim_gradients_1 = discrim_tape1.gradient(discrim_loss_1, self.discriminator.trainable_variables)
        #discrim_gradients_2 = discrim_tape2.gradient(discrim_loss_2, discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discrim_optimizer.apply_gradients(zip(discrim_gradients_1, self.discriminator.trainable_variables))
        #discrim_optimizer.apply_gradients(zip(discrim_gradients_2, discriminator.trainable_variables))

    def train(self, epochs):
        j = 0
        for i in range(epochs):
            print("epoch", i+1)
            for batch in self.img_data:
                j += 1
                self.train_step(batch[0].numpy()/255, j, None)
           # generator = define_generator()
    #print(generator.summary())
    
