import os

import numpy as np
from tensorflow.keras.models import load_model

# from tensorflow.python.keras._impl.keras.datasets.cifar import load_batch
from tensorflow.python.keras import backend as K
# from tensorflow.python.keras._impl.keras.utils import get_file


def load_mnist_data(path):  # 加载模型
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def divide_mnist_data(test, true_test):  # 划分模型
    Test_data = []
    Test_label = []
    for i in range(0, 10):
        temp = []
        for j in range(len(test)):
            if (true_test[j] == i):
                temp.append(test[j])
        Test_data.append(temp)
        Test_label.append(i)
        # print('子集',i,'：创建完成。')
    test_data = np.array(Test_data)
    test_label = np.array(Test_label)
    return Test_data, Test_label


def screen_data(test_data, test_label, size):  # 展示数据
    for i in range(len(test_data)):
        if (len(test_data[i]) < size):
            print('数据子集', i, '数据个数不够')
            break
        del test_data[i][size:len(test_data[i])]
        # print(len(test_data[i]))
    return test_data, test_label


def gen_data(name, path):  # 加载数据
    (X_train, Y_train), (X_test, Y_test) = load_mnist_data('C:/Users/86195/Desktop/newDesign/Design/deepgauge/mnist.npz')  # 28*28 获取数据   keras库的函数怎么运行
   # (X_train, Y_train), (X_test, Y_test) = load_mnist_data('../deepgauge/mnist.npz')
    X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
    X_train /= 255
    X_test /= 255
    model_path = path
    test = X_test
    true_test = Y_test
    train = X_train
    label = Y_train
    model = load_model(model_path)  # 加载模型
    pred_test_prob = model.predict(test)
    pred_test = np.argmax(pred_test_prob, axis=1)
    input = model.layers[0].output
    if (name == 'lenet5'):
        layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output, model.layers[6].output,
                  model.layers[8].output, model.layers[9].output, model.layers[10].output]
        layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))
    else:
        layers = [model.layers[1].output, model.layers[2].output, model.layers[3].output, model.layers[4].output,
                  model.layers[7].output]
        layers = list(zip(4 * ['conv'] + 1 * ['dense'], layers))
    return input, layers, test, train, label, pred_test, true_test, pred_test_prob


if __name__ == '__main__':
    pass
    #(X_train, Y_train), (X_test, Y_test) = load_mnist_data('./mnist.npz')
    # print('读取数据完成')
    # test_data,test_label = divide_mnist_data(X_test, Y_test)
    # for i in range(0, 10):
    #     print('test_data',np.array(test_data[i]))
