
# author = "sanglinwei"

from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import sys
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os


# get file path from the
def get_file_paths(directory):
    file_path = []
    file_name = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_path.append(filepath)
            file_name.append(filename)
    return file_path, file_name


class DCGAN():
    def __init__(self):
        # input shape
        self.img_rows = 7
        self.img_cols = 96
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # build generator
        self.generator = self.build_generator()

        # build the combined the model while not train the discriminator
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # false the discriminator
        self.discriminator.trainable = False

        # the combined model
        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        # transposed convolution
        model.add(Dense(128 * 7 * 24, activation='relu', input_dim=self.latent_dim))
        model.add(Reshape((7, 24, 128)))
        model.add(UpSampling2D(size=(1, 2)))
        model.add(Conv2D(128, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(UpSampling2D(size=(1, 2)))
        model.add(Conv2D(64, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation('sigmoid'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        Model(noise, img).summary()

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(256, kernel_size=4, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        model.add(LeakyReLU())

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        Model(img, validity).summary()

        return Model(img, validity)

    def train(self):

        # read data
        root_path = 'sample_data/Jiangxi_data/'
        # save_root_path = 'processed_data/'
        sample_path, sample_name = get_file_paths(root_path)
        sample_path.pop(2)
        sample_name.pop(2)

        # choose training data
        dataset_id = 0
        df1 = pd.read_csv(sample_path[dataset_id])
        idx = int(df1.shape[0] / (7 * 96)) * 7 * 96
        np1 = df1["PAP_R"][0:idx].to_numpy().reshape((-1, 7, 96))
        load_data = np.expand_dims(np1, axis=3)
        print(load_data.shape)

        # scale to -1 - 1
        scale = df1["PAP_R"].max()
        scaled_load_data = load_data / scale * 2 - 1
        print('scaled_load_data')
        # print(scaled_load_data)

        # training
        epochs = 2
        batch_size = 16
        save_interval = 50

        # ground truth
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # -----------------------
            # Train discriminator
            # -----------------------

            # select the random images
            idx = np.random.randint(0, scaled_load_data.shape[0], batch_size)
            imgs = scaled_load_data[idx]

            # sample noise and generate fake images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            print('generate images')
            print(gen_imgs)
            print(gen_imgs.shape)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------------
            # Train generator
            # -----------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # plot progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # save generated images samples
            if epoch % save_interval == 0:
                r, c = 1, 1
                noise = np.random.normal(0, 1, (r * c, self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                gen_load = gen_imgs.reshape((-1, 7 * 96))

                fig, axs = plt.subplots(r, c)
                plt.plot(gen_load)
                # cnt = 0
                # for i in range(r):
                #     for j in range(c):
                #         axs[i, j] = plt.plot(gen_load[cnt, :])
                #         cnt = cnt + 1
                #         print(cnt)

                fig.savefig("image/generate_load_%d.png" % epoch)
                plt.close()


if __name__ == '__main__':
    # start
    dcgan = DCGAN()
    dcgan.train()












