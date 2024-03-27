import numpy as np
from keras import Model
from tqdm import tqdm
from .load_data import gen_data
from scipy import stats
from functools import reduce
# k-多节神经元覆盖率
import tensorflow as tf

from .mnist_pic_transform_npy import get_mnist_test_npy


class kmnc(object):
    def __init__(self, train, test, input, layers, k_bins=1000):
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train = train
        self.input = input
        self.layers = layers
        self.k_bins = k_bins
        self.lst = []
        self.upper = []
        self.lower = []
        self.test = test
        index_lst = []
        for index, l in layers:
            self.lst.append(tf.keras.models.Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = tf.keras.models.Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = i.predict(train).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.upper.append(np.max(temp, axis=0))
            self.lower.append(np.min(temp, axis=0))
        self.upper = np.concatenate(self.upper, axis=0)
        self.lower = np.concatenate(self.lower, axis=0)
        self.neuron_num = self.upper.shape[0]
        self.lst = list(zip(index_lst, self.lst))

    def fit(self):
        '''
        test:测试集数据
        输出测试集的覆盖率
        '''
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = l.predict(self.test).reshape(len(self.test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(self.test).reshape(len(self.test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)
        act_num = 0
        for index in range(len(self.upper)):
            bins = np.linspace(self.lower[index], self.upper[index], self.k_bins)
            act_num += len(np.unique(np.digitize(self.neuron_activate[:, index], bins)))
        return act_num / float(self.k_bins * self.neuron_num)  # 所有神经元占的总节数/所有节数


# 神经元边界覆盖率
class nbc(object):
    def __init__(self, train, test, input, layers, std=0):
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train = train
        self.input = input
        self.layers = layers
        self.std = std
        self.lst = []
        self.upper = []
        self.lower = []
        self.test = test
        index_lst = []
        for index, l in layers:
            self.lst.append(tf.keras.models.Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = tf.keras.models.Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = i.predict(train).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.upper.append(np.max(temp, axis=0) + std * np.std(temp, axis=0))
            self.lower.append(np.min(temp, axis=0) - std * np.std(temp, axis=0))
        self.upper = np.concatenate(self.upper, axis=0)
        self.lower = np.concatenate(self.lower, axis=0)
        self.neuron_num = self.upper.shape[0]
        self.lst = list(zip(index_lst, self.lst))

    def fit(self, use_lower=False):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = l.predict(self.test).reshape(len(self.test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(self.test).reshape(len(self.test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)
        act_num = 0
        act_num += (np.sum(self.neuron_activate > self.upper, axis=0) > 0).sum()
        print(use_lower, ',', act_num)
        # print(self.upper)
        if use_lower:
            act_num += (np.sum(self.neuron_activate < self.lower, axis=0) > 0).sum()
        if use_lower:
            return act_num / (2 * float(self.neuron_num))
        else:
            return act_num / float(self.neuron_num)


# top-k神经元覆盖率
class tknc(object):
    def __init__(self, test, input, layers, k=2):
        self.train = test
        self.input = input
        self.layers = layers
        self.k = k
        self.lst = []
        self.neuron_activate = []
        self.test = test
        index_lst = []

        for index, l in layers:
            self.lst.append(tf.keras.models.Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = tf.keras.models.Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = i.predict(self.test).reshape(len(self.test), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(self.test).reshape(len(self.test), l.shape[-1])
            self.neuron_activate.append(temp)
        self.neuron_num = np.concatenate(self.neuron_activate, axis=1).shape[-1]
        self.lst = list(zip(index_lst, self.lst))

    def fit(self, choice_index):
        neuron_activate = 0
        for neu in self.neuron_activate:
            temp = neu[choice_index]
            neuron_activate += len(np.unique(np.argsort(temp, axis=1)[:, -self.k:]))
        return neuron_activate / float(self.neuron_num)


# if __name__ == '__main__':
#     input, layers, test, train, label, pred_test, true_test, pred_test_prob = gen_data('lenet5',
#                                                                                                  './mnist_lenet5.h5')
#     test = []
#     for i in range(0, 10):
#         # path =r"./mnist_data/size_200/dnn_difference/update_1/"+ str(i) + "/"
#         path = "./mnist_data/size_200/" + str(i) + "/"
#         test_temp, label_temp, name =get_mnist_test_npy(path)
#         test.append(test_temp)
#     test = np.concatenate(np.array(test), axis=0)
#     kmnc1 = kmnc(train, test, input, layers, 1000)
#     print('kmnc:', kmnc1.fit())
#     nbc1 = nbc(train, test, input, layers, 0)
#     print('snac:', nbc1.fit(False))  # 强神经元覆盖率
#     print('nbc:', nbc1.fit(True))  #
#     tknc1 = tknc(test, input, layers, 2)
#     print('tknc:', tknc1.fit(True))
