import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam

class DCGAN():
    def __init__(self, img_data, gen_lr=0.0003, discrim_lr = 0.0001): #img_data should be a keras dataset from image directory
        self.generator_optimizer = Adam(learning_rate=gen_lr)
        self.discrim_optimizer = Adam(learning_rate=discrim_lr)
        
        self.img_data = img_data
        
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        
        self.x_rec_discrim_loss = []
        self.y_rec_discrim_loss = []
        
        self.x_rec_generator_loss = []
        self.y_rec_generator_loss = []
    
    def define_discriminator(self, model=None, in_shape=(256,256,3)): #automatically defaults to default discriminator if model is not inputted
        if model != None:
            self.discriminator = model
            self.discriminator.compile(optimizer=self.discrim_optimizer)
            return
        
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

    def define_generator(self, model=None): #automatically defaults to default generator if a model is not provided
        if model != None:
            self.generator = model
            self.generator.compile(optimizer=self.generator_optimizer)
            return
        
        model = tf.keras.Sequential()
        model.add(layers.Dense(64*64*3, use_bias=False, input_shape=(500,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((64, 64, 3)))

        model.add(layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (3,3), strides=(1,1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (3,3), strides=(1,1), padding='same', use_bias=False, activation='sigmoid'))

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
        try:
            if self.discriminator:
                pass
        except:
            raise ValueError("Discriminator not defined.")
            
        try:
            if self.generator:
                pass
        except:
            raise ValueError("Generator not defined.")
        
        noise = np.random.normal(size=[len(real_img_batch), 64,64,3])

        with tf.GradientTape() as discrim_tape1, tf.GradientTape() as discrim_tape2, tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)

            real_img_output = self.discriminator(real_img_batch.reshape((len(real_img_batch), 256, 256, 3)), training=True)
            fake_img_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_img_output)

            discrim_loss_1 = self.discrim_loss(real_img_output, fake_img_output)

        self.x_rec_discrim_loss.append(discrim_loss_1)
        self.y_rec_discrim_loss.append(iteration)

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
