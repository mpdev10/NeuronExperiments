from Neuron import Neuron
import numpy as np


class Adaline(Neuron):

    def __init__(self, input_size, init='rand', error_th=0.26, weight_range=None):
        super().__init__(input_size, init, weight_range)
        self.loss = 0
        self.error_th = error_th

    def train(self, train_x, train_y, learning_rate=0.05):

        while True:
            for j in range(len(train_y)):
                x = np.reshape(train_x[j], (1, 2))
                y = train_y[j]
                predicted_label = self._forward(x)
                error = y - predicted_label
                self.weights = self.weights + (learning_rate * error * np.transpose(x))
                self.b = self.b + learning_rate * error

            y_predict = self._forward(train_x)
            global_error = np.reshape(train_y, np.shape(y_predict)) - y_predict
            loss = np.mean(global_error ** 2)
            if loss <= self.error_th:
                break

