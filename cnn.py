from keras.datasets import fashion_mnist
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import logging

logging.getLogger('tensorflow').disabled = True


def evaluate_acc(y, y_hat):
    num = 0
    for i in range(len(y)):
        if y[i] == y_hat[i].argmax():
            num += 1
    return num / len(y) * 100


def get_data():
    return fashion_mnist.load_data()

def get_data2():
    return mnist.load_data()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = get_data()
    # (x_train, y_train), (x_test, y_test) = get_data2()
    # print(x_test.shape)


    model = tf.keras.models.Sequential()
    # two convolutional layers
    model.add(tf.keras.layers.Conv2D(filters=30, kernel_size=(3, 3), strides=(1, 1),
                                     activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(filters=15, kernel_size=(3, 3), strides=(1, 1),
                                     activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # change dim
    model.add(tf.keras.layers.Flatten())

    # two fully connected layers
    # model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))

    # output layer
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    model.summary()

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Train the CNN on the training data
    history = model.fit(x_train, y_train, batch_size=300, epochs=2, validation_split=0.1, verbose=1)

    y_bar = model.predict(x_test)
    print(evaluate_acc(y_test, y_bar))
