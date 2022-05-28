from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.datasets import cifar10
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.getLogger('tensorflow').disabled = True


def get_data():
    return fashion_mnist.load_data()


def get_data2():
    return mnist.load_data()


def get_data3():
    return cifar10.load_data()


if __name__ == '__main__':
    path1 = "/Users/alain/Desktop/winter2022/comp551/as3/code/Task1/output.csv"
    path2 = "/Users/alain/Desktop/winter2022/comp551/as3/code/Task1/output2.csv"
    path3 = "/Users/alain/Desktop/output.csv"
    path4 = "/Users/alain/Desktop/output2.csv"
    x = []
    training_accuracy0 = []
    training_loss0 = []
    with open(path3, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            x.append(int(line.split(',')[0][6:]))
            training_accuracy0.append(float(line.split(',')[1][20:]))
            training_loss0.append(float(line.split(',')[2][17:-1]))
    f.close()

    training_accuracy = []
    training_loss = []
    with open(path4, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            training_accuracy.append(float(line.split(',')[1][20:]))
            training_loss.append(float(line.split(',')[2][17:-1]))
    f.close()

    # plt.title("CNN_softmax training accuracy on Fashion_MNIST")
    # plt.title("CNN_softmax training loss on Fashion_MNIST")
    # plt.title("CNN_SVM training accuracy on Fashion_MNIST")
    # plt.title("CNN_SVM training loss on Fashion_MNIST")
    plt.title("Fashion_MNIST")
    # plt.title("training loss on MNIST")
    plt.xlabel("epochs")
    plt.ylabel("training accuracy")
    plt.plot(x,training_accuracy0,label="CNN_softmax")
    plt.plot(x,training_accuracy,label="CNN_SVM")
    plt.legend()
    plt.show()

    (x_train, y_train), (x_test, y_test) = get_data()
    # (x_train, y_train), (x_test, y_test) = get_data2()
    # (x_train, y_train), (x_test, y_test) = get_data3()
    # print(x_train.shape)
    # plt.title("Class distribution of training set")
    # _, _, patches = plt.hist(y_train, bins=10)
    # for pp in patches:
    #     x = pp._x0
    #     y = pp._height + 50
    #     plt.text(x, y, int(pp._height))
    # plt.show()

