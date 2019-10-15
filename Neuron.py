import numpy as np


class Neuron:

    def __init__(self, input_size, init='rand', weight_range=[0, 0]):
        if init == 'rand':
            self.weights = np.random.random((input_size, 1))
        elif init == 'zeros':
            self.weights = np.zeros((input_size, 1))
        elif init == 'range':
            self.weights = np.random.uniform(low=weight_range[0], high=weight_range[1], size=(input_size, 1))

        self._unipolar = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        self._bipolar = np.vectorize(lambda x: 1 if x >= 0 else -1)
        self.epoch_num = 0
        self.b = np.random.uniform(low=weight_range[0], high=weight_range[1], size=(1, 1))


    def predict(self, x, activation='unipolar'):
        forward = self._forward(x)
        if activation == 'unipolar':
            return self._unipolar(forward)
        elif activation == 'bipolar':
            return self._bipolar(forward)
        elif activation == 'none':
            return forward

    def _forward(self, x):
        return x @ self.weights + self.b
