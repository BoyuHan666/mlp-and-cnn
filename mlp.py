import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import random

# np.random.seed(123)
# random.seed(123)


def get_batch(x, y, bs):
    num_instances = len(x)
    inds = random.sample(range(0, num_instances), bs)
    x_select, y_select = x[inds], y[inds]
    return x_select, y_select


def softmax_y(y):
    k = []
    for i in y:
        l = []
        for j in range(len(set(y.tolist()))):
            if i == j:
                l.append(1)
            else:
                l.append(0)
        k.append(l)
    return np.array(k)


def normalization_x(x):
    np.seterr(divide='ignore', invalid='ignore')
    nor_matrix = []
    for data in x:
        features = data - np.mean(data)
        features /= np.std(features)
        nor_matrix.append(features)
    nor_matrix = np.array(nor_matrix)
    return nor_matrix

def nor_hidden_for_relu(x):
    x -= np.min(x)
    x /= np.max(x) - np.min(x)
    return x

def ReLU(x):
    x[x < 0] = 0
    return x


def dReLU(x):
    x[x < 0] = 0
    x[x >= 1] = 1
    return x


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return 1.0 - np.power(x, 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return np.multiply(x, (1.0 - x))


def softmax(xs):
    l = []
    for x in xs:
        l.append(np.exp(x) / np.sum(np.exp(x)))
    return np.array(l)


def cost_function2(y, y_pre):
    loss = 1 / 2 * np.sum((y - y_pre) ** 2)
    return loss / float(y_pre.shape[0])


class MLP:

    def __init__(self, num_input, num_hidden, hidden_layer, num_output, activation_function, derivative, nor_factor, dp, dp_prob):
        self.loss_list = []
        self.af = activation_function
        self.daf = derivative
        self.nor_factor = nor_factor
        self.dp = dp
        self.dp_prob = dp_prob
        self.layer = [num_input] + hidden_layer + [num_output]
        self.num_layer = len(self.layer)
        self.hidden_layer = hidden_layer
        self.num_hidden = num_hidden
        weights = []
        bias = []
        weights_derivatives = []
        bias_derivatives = []
        activations = []
        pdw = []
        pdb = []
        sdw = []
        sdb = []
        for i in range(self.num_layer - 1):
            w = np.random.rand(self.layer[i], self.layer[i + 1])
            b = np.random.rand(self.layer[i + 1])
            wd = np.zeros((self.layer[i], self.layer[i + 1]))
            bd = np.zeros((self.layer[i + 1]))
            activation = np.ones(self.layer[i])
            pw = np.zeros((self.layer[i], self.layer[i + 1]))
            pb = np.zeros((self.layer[i + 1]))
            sw = np.zeros((self.layer[i], self.layer[i + 1]))
            sb = np.zeros((self.layer[i + 1]))
            weights.append(w)
            bias.append(b)
            weights_derivatives.append(wd)
            bias_derivatives.append(bd)
            activations.append(activation)
            pdw.append(pw)
            pdb.append(pb)
            sdw.append(sw)
            sdb.append(sb)
        self.weights = weights
        self.bias = bias
        self.weights_derivatives = weights_derivatives
        self.bias_derivatives = bias_derivatives
        self.activations = activations
        self.pdw = pdw
        self.pdb = pdb
        self.sdw = sdw
        self.sdb = sdb


    def feed_forward(self, inputs):
        if self.dp:
            activation = self.dropout(inputs, self.dp_prob)
            # self.activations[0] = self.dropout(inputs, self.dp_prob)
            # activation = inputs
            self.activations[0] = inputs
            for i in range(self.num_layer - 1):
                if i == self.num_layer - 2:
                    output = self.activations[i] @ self.weights[i] + self.bias[i]
                    predictions = np.array(softmax(output))
                    self.predictions = predictions
                else:
                    activation = self.af(activation @ self.weights[i] + self.bias[i])
                    if self.af == ReLU or self.af == leaky_ReLU:
                        self.activations[i + 1] = self.dropout(nor_hidden_for_relu(activation) * self.nor_factor, self.dp_prob)
                        # self.activations[i + 1] = self.dropout(activation / np.max(activation) * self.nor_factor, self.dp_prob)
                    else:
                        self.activations[i + 1] = self.dropout(activation * self.nor_factor, self.dp_prob)
        else:
            activation = inputs
            self.activations[0] = inputs
            for i in range(self.num_layer - 1):
                if i == self.num_layer - 2:
                    output = self.activations[i] @ self.weights[i] + self.bias[i]
                    predictions = np.array(softmax(output))
                    self.predictions = predictions
                else:
                    activation = self.af(activation @ self.weights[i] + self.bias[i])
                    if self.af == ReLU or self.af == leaky_ReLU:
                        self.activations[i + 1] = nor_hidden_for_relu(activation) * self.nor_factor
                        # self.activations[i + 1] = activation / np.max(activation) * self.nor_factor
                    else:
                        self.activations[i + 1] = activation * self.nor_factor
        return self

    def back_propagation(self, y):
        loss = 0
        for i in range(self.num_layer - 2, -1, -1):
            if i == self.num_layer - 2:
                loss = y - self.predictions
            else:
                loss = (loss @ self.weights[i + 1].T) * self.daf(self.activations[i + 1])
            delta_w = (1 / len(y)) * (self.activations[i]).T @ loss
            delta_v = (1 / len(y)) * np.sum(loss, axis=0)
            self.weights_derivatives[i] = delta_w
            self.bias_derivatives[i] = delta_v
        return

    def gradient_descent(self, lr1, lr2, beta1, beta2, adams):
        for i in range(len(self.weights)):
            if adams:
                self.pdw[i] = beta1 * self.pdw[i] + (1 - beta1) * self.weights_derivatives[i]
                self.pdb[i] = beta1 * self.pdb[i] + (1 - beta1) * self.bias_derivatives[i]
                self.sdw[i] = beta2 * self.sdw[i] + (1 - beta2) * np.square(self.weights_derivatives[i])
                self.sdb[i] = beta2 * self.sdb[i] + (1 - beta2) * np.square(self.bias_derivatives[i])
                self.weights[i] += lr1 / np.sqrt(self.sdw[i] + 1e-8) * self.pdw[i]
                self.bias[i] += (lr2 / np.sqrt(self.sdb[i] + 1e-8)) * self.pdb[i]
            else:
                self.weights[i] += lr1 * self.weights_derivatives[i]
                self.bias[i] += lr2 * self.bias_derivatives[i]

    def fit(self, x_train, y_train, lr1, lr2, beta1, beta2, epochs, batch_size, adams):
        num = 0
        for i in range(epochs):
            num = num + 1
            if num % 100 == 0:
                print("iterate times:" + str(num))
            x, y = get_batch(x_train, y_train, batch_size)
            self.feed_forward(x)
            loss = cost_function2(y, self.predictions)
            self.loss_list.append(loss)
            self.back_propagation(y)
            self.gradient_descent(lr1, lr2, beta1, beta2, adams)

        plt.plot(range(epochs), self.loss_list)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.title('training performance over epochs')
        plt.show()

    def predict(self, x_test):
        self.feed_forward(x_test)
        return self.predictions


    def dropout(self, activation, p):
        # p *= random.random()
        n = activation.shape[1]
        k = np.random.binomial(1, 1-p, size=n) / (1-p)
        activation *= k
        return activation


def evaluate_acc(y, y_hat):
    num = 0
    for i in range(len(y)):
        if y[i] == y_hat[i].argmax():
            num += 1
    return num / len(y) * 100


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train[:60000]
    y_train = y_train[:60000]

    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    print("start normalization...")
    x_train = normalization_x(x_train)
    x_test = normalization_x(x_test)
    print("finish normalization...")
    y_train = softmax_y(y_train)

    # predict by relu with 0 hidden layer
    # mlp = MLP(num_input=784, num_hidden=0, hidden_layer=[], num_output=10,
    #                activation_function=ReLU, derivative=dReLU, nor_factor=6, dp=False, dp_prob=0.05)
    # print("start fitting...")
    # mlp.fit(x_train=x_train, y_train=y_train, lr1=0.01, lr2=0.01,
    #              beta1=0.99, beta2=0.99, epochs=200, batch_size=200, adams=False)

    # # predict by relu with one hidden layer
    # mlp = MLP(num_input=784, num_hidden=1, hidden_layer=[200], num_output=10,
    #           activation_function=ReLU, derivative=dReLU, nor_factor=6, dp=False, dp_prob=0.05)
    # print("start fitting...")
    # mlp.fit(x_train=x_train, y_train=y_train, lr1=0.01, lr2=0.01,
    #         beta1=0.9, beta2=0.9, epochs=2000, batch_size=1000, adams=True)
    #
    # predict by relu with two hidden layer
    # mlp = MLP(num_input=784, num_hidden=2, hidden_layer=[128, 128], num_output=10,
    #           activation_function=ReLU, derivative=dReLU, nor_factor=1, dp=False, dp_prob=0.05)
    # print("start fitting...")
    # mlp.fit(x_train=x_train, y_train=y_train, lr1=0.01, lr2=0.01,
    #         beta1=0.9, beta2=0.9, epochs=100, batch_size=200, adams=False)

    # predict by relu with two layer with dropout
    # mlp = MLP(num_input=784, num_hidden=2, hidden_layer=[128, 128], num_output=10,
    #           activation_function=ReLU, derivative=dReLU, nor_factor=5, dp=True, dp_prob=0.05)
    # print("start fitting...")
    # mlp.fit(x_train=x_train, y_train=y_train, lr1=0.01, lr2=0.01,
    #         beta1=0.9, beta2=0.9, epochs=2000, batch_size=500, adams=True)

    # predict by leaky_relu with two layer
    a_factor = 0.001


    def leaky_ReLU(x, a=a_factor):
        x[x < 0] *= a
        return x


    def dleaky_ReLU(x, a=a_factor):
        x[x < 0] = a
        x[x >= 1] = 1
        return x


    mlp = MLP(num_input=784, num_hidden=1, hidden_layer=[200], num_output=10,
              activation_function=leaky_ReLU, derivative=dleaky_ReLU, nor_factor=6, dp=False, dp_prob=0.1)
    print("start fitting...")
    mlp.fit(x_train=x_train, y_train=y_train, lr1=0.01, lr2=0.01,
            beta1=0.9, beta2=0.9, epochs=2000, batch_size=1000, adams=True)

    # predict by tanh
    # mlp = MLP(num_input=784, num_hidden=2, hidden_layer=[128,128], num_output=10,
    #                activation_function=tanh, derivative=dtanh, nor_factor=6, dp=False, dp_prob=0.1)
    # print("start fitting...")
    # mlp.fit(x_train=x_train, y_train=y_train, lr1=0.008, lr2=0.008,
    #              beta1=0.98, beta2=0.98, epochs=550, batch_size=1000, adams=True)

    y_bar = mlp.predict(x_test)
    print(mlp.loss_list)
    print(mlp.predictions[0])
    print(evaluate_acc(y_test, y_bar))
