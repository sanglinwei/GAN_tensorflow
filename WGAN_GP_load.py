
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt

import sys
import pandas as pd

import numpy as np


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


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
    model.add(Dense(2 * 24 * 80, activation='relu', use_bias=False, input_dim=latent_dim))
    # model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Reshape((2, 24, 80)))
    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))
    model.add(
        Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', output_padding=(0, 1), name='strange_padding'))
    model.add(Activation('tanh'))

    model.summary()

    _noise = Input(shape=(latent_dim,))
    _img = model(_noise)

    return Model(_noise, _img)


def build_discriminator(img_shape=(7, 96, 1)):

    model = Sequential()

    # GAN paper
    model.add(Conv2D(32, kernel_size=4, strides=2, input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=4, strides=1, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, kernel_size=4, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    # model.add(Dense(100))
    # model.add(LeakyReLU(0.2))
    # model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    _img = Input(shape=img_shape)
    validity = model(_img)

    return Model(_img, validity)


def gradient_penalty_loss(y_true, y_pred, average_samples):
    """
    compute the gradients penalty
    """
    gradients = K.gradients(y_pred, average_samples)[0]
    gradients_sqr = K.square(gradients)


    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))

    gradient_l2_norm = K.square(gradients_sqr_sum)

    gradient_penalty = K.square((1-gradient_l2_norm))

    return K.mean(gradient_penalty)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


if __name__ == '__main__':
    # build GAN
    img_rows = 7
    img_cols = 96
    channels = 1
    img_shape = (img_rows, img_cols, channels)
    latent_dim = 100

    # wasserstein parameters
    n_critic = 5
    clip_value = 0.01
    optimizer = RMSprop(lr=0.000005)

    # choose training data
    dataset_id = 0
    df1 = pd.read_csv('./dataport/load_profile.csv')
    idx = int(df1.shape[0] / (7 * 96)) * 7 * 96
    np1 = df1[0:idx].to_numpy()[:, 1]
    for i in range(df1.shape[1] - 2):
        np1 = np.concatenate((np1, df1[0:idx].to_numpy()[:, i + 2]), axis=0)
    np2 = np1.reshape((-1, 7, 96))
    load_data = np.expand_dims(np2, axis=3)

    # scale to -1 - 1
    scale = np1.max() - np1.min()
    scaled_load_data = (load_data - np1.min()) / scale * 2 - 1
    scaled_load_data.astype(np.float32)

    # bulid the WGAN-GP
    generator = build_generator()
    discriminator = build_discriminator()

    generator.trainable = False

    real_img = Input(shape=img_shape)

    z_disc = Input(shape=(latent_dim,))

    fake_img = generator(z_disc)

    fake = discriminator(fake_img)
    valid = discriminator(real_img)

    interpolated_img = RandomWeightedAverage()([real_img, fake_img])

    valid_interpolated = discriminator(interpolated_img)

    partial_gp_loss = partial(gradient_penalty_loss, average_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty'

    discriminator_model = Model(inputs=[real_img, z_disc], outputs=[valid, fake, valid_interpolated])
    discriminator_model.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss], optimizer=optimizer,
                                loss_weights=[1, 1, 10])

    discriminator.trainable = False
    generator.trainable = True

    z_gen = Input(shape=(latent_dim,))
    img = generator(z_gen)
    valid = discriminator(img)
    generator_model = Model(z_gen, valid)
    generator_model.compile(loss=wasserstein_loss, optimizer=optimizer)

    # trainning
    batch_size = 32
    epochs = 10

    valid = -np.ones((batch_size, 1))
    fake = np.ones((batch_size, 1))
    dummy = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # train discriminator
        for _ in range(n_critic):

            idx = np.random.randint(0, scaled_load_data.shape[0], batch_size)
            imgs = scaled_load_data[idx]

            noise = np.random.normal(0, 1, [batch_size, latent_dim])
            d_loss = discriminator_model.train_on_batch([imgs, noise], [valid, fake, dummy])

        # train Generator

        g_loss = generator_model.train_on_batch(noise, valid)
        print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))






