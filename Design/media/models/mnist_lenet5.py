# python3.6
# -*- coding: utf-8 -*-
'''
mnist模型训练程序
val_loss: 0.02655
val_acc: 0.9914
'''

from keras import Model, Input
import sys
import time
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
# import pandas as pd
import numpy as np
import load_data




def model_mnist():
    # (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28
    (X_train, Y_train), (X_test, Y_test) = load_data.load_mnist_data('./mnist.npz')
    X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
    X_train /= 255
    X_test /= 255
    print('Train:{},Test:{}'.format(len(X_train), len(X_test)))

    nb_classes = 10

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    print('data success')

    input_tensor = Input((28, 28, 1))
    temp = Conv2D(filters=6, kernel_size=(5, 5),
                  padding='valid', use_bias=False)(input_tensor)
    temp = Activation('relu')(temp)
    temp = MaxPooling2D(pool_size=(2, 2))(temp)
    temp = Conv2D(filters=16,
                  kernel_size=(5, 5),
                  padding='valid',
                  use_bias=False)(temp)
    temp = Activation('relu')(temp)
    temp = MaxPooling2D(pool_size=(2, 2))(temp)
    temp = Flatten()(temp)
    temp = Dense(120, activation='relu')(temp)
    temp = Dense(84, activation='relu')(temp)
    output = Dense(nb_classes, activation='softmax')(temp)

    model = Model(inputs=input_tensor, outputs=output)
    model.summary()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath='./mnist_lenet5.h5', monitor='val_accuracy', mode='auto',
                                 save_best_only='TRUE')
    model.fit(X_train, Y_train, batch_size=64, epochs=15, validation_data=(X_test, Y_test), callbacks=[checkpoint])
    # model = load_model(r'C:\Users\86195\Desktop\deep\mnist_data/mnist_lenet5.h5')
    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print(score)


if __name__ == '__main__':
    model_mnist()
