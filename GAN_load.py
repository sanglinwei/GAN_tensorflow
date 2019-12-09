
# author = "sanglinwei"

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
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


def build_generator(latent_dim=100, channels=1):

    model = Sequential()

    # transposed convolution
    # model.add(Dense(128 * 7 * 24, activation='relu', input_dim=latent_dim))
    # model.add(Reshape((7, 24, 128)))
    # model.add(UpSampling2D(size=(1, 2)))
    # model.add(Conv2D(128, kernel_size=3, padding='same'))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation('relu'))
    # model.add(UpSampling2D(size=(1, 2)))
    # model.add(Conv2D(64, kernel_size=3, padding='same'))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation('relu'))
    # model.add(Conv2D(channels, kernel_size=3, padding='same'))
    # model.add(Activation('sigmoid'))

    # GAN papers
    model.add(Dense(2 * 24 * 80, activation='relu', input_dim=latent_dim))
    model.add(Reshape((2, 24, 80)))
    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2,  padding='same', output_padding=(0, 1), name='strange_padding'))
    model.add(Activation('sigmoid'))

    model.summary()

    _noise = Input(shape=(latent_dim,))
    _img = model(_noise)

    Model(_noise, _img).summary()

    return Model(_noise, _img)


def build_discriminator(img_shape=(7, 96, 1)):

    model = Sequential()

    # model.add(Conv2D(256, kernel_size=4, strides=2, input_shape=img_shape, padding='same'))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    # model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))

    # GAN paper
    model.add(Conv2D(256, kernel_size=4, strides=2, input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(3072))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    _img = Input(shape=img_shape)
    validity = model(_img)

    Model(_img, validity).summary()

    return Model(_img, validity)


if __name__ == '__main__':

    # build GAN
    img_rows = 7
    img_cols = 96
    channels = 1
    img_shape = (img_rows, img_cols, channels)
    latent_dim = 100

    optimizer = Adam(0.00002, 0.5)

    discriminator = build_discriminator(img_shape)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    generator = build_generator(latent_dim, channels)

    z = Input(shape=(latent_dim,))
    img = generator(z)

    # false the discriminator
    discriminator.trainable = False

    # the combined model
    valid = discriminator(img)

    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # training
    # read data
    root_path = 'dataport/'
    # save_root_path = 'processed_data/'
    sample_path, sample_name = get_file_paths(root_path)

    # choose training data
    dataset_id = 0
    df1 = pd.read_csv(sample_path[dataset_id])
    idx = int(df1.shape[0] / (7 * 96)) * 7 * 96
    np1 = df1[0:idx].to_numpy()[:, 1]
    for i in range(df1.shape[1]-2):
        np1 = np.concatenate((np1, df1[0:idx].to_numpy()[:, i+2]), axis=0)
    np2 = np1.reshape((-1, 7, 96))
    load_data = np.expand_dims(np2, axis=3)

    # scale to -1 - 1
    scale = np1.max()-np1.min()
    scaled_load_data = (load_data-np1.min()) / scale * 2 - 1
    print('scaled_load_data')
    # print(scaled_load_data)
    # training
    epochs = 20
    batch_size = 32
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
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)
        print('generate images')
        print(gen_imgs.shape)

        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # -----------------------
        # Train generator
        # -----------------------

        g_loss = combined.train_on_batch(noise, valid)

        # plot progress
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        # save generated images samples
        if epoch % save_interval == 0:
            r, c = 1, 1
            noise = np.random.normal(0, 1, (r * c, latent_dim))
            gen_imgs = generator.predict(noise)
            gen_load = gen_imgs.reshape((7 * 96, -1))

            fig, axs = plt.subplots(r, c)
            plt.plot(gen_load)
            plt.show()
            # cnt = 0
            # for i in range(r):
            #     for j in range(c):
            #         axs[i, j] = plt.plot(gen_load[cnt, :])
            #         cnt = cnt + 1
            #         print(cnt)

            fig.savefig("image/generate_load_%d.png" % epoch)
            plt.close()


# for plotting

# import seaborn as sns; sns.set()
# df1 = pd.read_csv(sample_path[1])
# idx = int(df1.shape[0] / (7 * 96)) * 7 * 96
# np1 = df1["PAP_R"][0:idx].to_numpy().reshape((-1, 7, 96))
# ax = sns.heatmap(np1[3])
#
# plt.show()


# import seaborn as sns; sns.set()
# df1 = pd.read_csv(sample_path[4])
# idx = int(df1.shape[0] / (7 * 96)) * 7 * 96
# np1 = df1["PAP_R"][0:idx].to_numpy().reshape((-1, 7, 96))
# plt.plot(np1[1].flatten())
# plt.plot(np1[2].flatten())
# plt.plot(np1[3].flatten())
# plt.plot(np1[4].flatten())
# plt.plot(np1[5].flatten())
# plt.plot(np1[6].flatten())
# plt.plot(np1[7].flatten())
# plt.show()
