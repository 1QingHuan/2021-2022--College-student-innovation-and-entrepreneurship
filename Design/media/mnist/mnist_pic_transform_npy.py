import os
import numpy as np
from PIL import Image


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]


def get_mnist_test_npy(path):
    data = np.load('./mnist.npz')
    print(type(data['x_test']))
    print((data['x_test']).dtype)
    print((data['x_test']).size)
    print((data['x_test']).shape)
    # print((['x_test']).ndim)
    print(np.array("x_test"))
    print(type(data['y_test']))
    print((data['y_test']).dtype)
    print((data['y_test']).size)
    print((data['y_test']).shape)
    # print((data['y_test']).ndim)
    print(np.array("y_test"))
    print(data['y_test'])
    filename = get_imlist(r"./mnist_data/size_200/0/")
    class_name = path[-2:-1]
    filename = get_imlist(path)
    d = len(filename)
    test_x = np.empty((d, 28, 28), dtype=np.uint8)
    test_y = np.empty((d,), dtype=np.uint8)
    path_name = []
    i = 0
    while i < d:
        img = Image.open(filename[i])
        img = img.convert('L')
        img_ndarray = np.array(img)
        # print(img_ndarray.shape)
        test_x[i] = img_ndarray
        # print(img_ndarray.shape)
        test_y[i] = class_name
        path_name.append(filename[i])
        i += 1

        # im = Image.fromarray(np.uint8(img_attack)).convert('RGB')

    # print ("test_x.shape:",test_x.shape)
    # print(test_x)
    # print("test_y.shape:",test_y.shape)
    # print(test_y)
    # print(test_x.shape)
    test_x = test_x.astype('float32').reshape(-1, 28, 28, 1)
    test_x /= 255
    # print(test_x.shape)
    return test_x, test_y, path_name


def get_data(path):
    test_x = []
    test_y = []
    pathname = []
    for i in range(0, 10):
        temp_x, temp_y, temp_name = get_mnist_test_npy(path + '/' + str(i) + "/")
        # temp_x, temp_y, temp_name = get_mnist_test_npy(r"./mnist_data/size_200/origin/" + str(i) + "/")
        test_x.append(temp_x)
        test_y.append(temp_y)
        pathname.append(temp_name)
    return test_x, test_y, pathname


if __name__ == '__main__':
    pass
    # root = './mnist_data/size_200'  # 保存地址
    # path = r"C:\Users\86195\Desktop\deep\mnist.npz"
    # data = np.load(path)
    # x_train = data["x_train"]
    # y_train = data["y_train"]
    #
    # n = 0
    # for i in range(60000):
    #     print(y_train[i])
    #     im = Image.fromarray(x_train[i])
    #     # im.show()
    #     if y_train[i] == 9:
    #         im.save('./mnist_data/size_200/9' + os.sep + '9_' + str(n) + '.png')
    #         n += 1
    #
    # test_x, test_y, pathname = get_data(r"./mnist_data/size_200/images")
    # print(pathname[6])
    # print(test_x[1].shape)
    # print(pathname[5][3])
    # img = Image.fromarray(test_x[5][3].reshape(28, 28) * 255);
    # img.show()
