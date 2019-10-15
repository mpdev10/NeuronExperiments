from Neuron import Neuron
import numpy as np


class Perceptron(Neuron):

    def __init__(self, input_size, init='rand', weight_range=[0, 0], activation='unipolar'):
        super().__init__(input_size, init, weight_range)
        self.activation = activation

    def _forward(self, x):
        return super()._forward(x) + self.b

    def train(self, train_x, train_y, learning_rate=0.05):
        while True:
            loss = 0
            for j in range(len(train_y)):
                x = np.reshape(train_x[j], (1, 2))
                y = train_y[j]
                predicted_label = self.predict(x, activation=self.activation)
                error = y - predicted_label
                self.weights = self.weights + (learning_rate * error * np.transpose(x))
                self.b = self.b + learning_rate * error
                loss = loss + abs(error)
            if loss == 0:
                break
            self.epoch_num = self.epoch_num + 1
