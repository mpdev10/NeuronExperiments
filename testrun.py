import numpy as np

from Adaline import Adaline
from Perceptron import Perceptron

train_x_uni = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_x_bi = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])


def test_run(neuron_t, train_x, train_y, test_label, activation, lr, init='range', weight_range=None, run_num=1):
    epoch_nums = np.array([])
    print(test_label)
    x = 0
    while x < run_num:
        if neuron_t == 'perceptron':
            neuron = Perceptron(2, init=init, weight_range=weight_range, activation=activation)
        else:
            neuron = Adaline(2, init=init, weight_range=weight_range, error_th=0.3)
        neuron.train(train_x, train_y, learning_rate=lr)
        epoch_nums = np.append(epoch_nums, neuron.epoch_num)
        x += 1
    average = np.average(epoch_nums)
    std_deviation = np.sqrt(np.average((epoch_nums - average) ** 2))

    print("AVG EPOCH NUM:", average)
    print("STD DEVIATION:", std_deviation)


def test_activation(train_y, neuron, activation, lr, weight_range, run_num):
    if activation == 'bipolar':
        test_run(neuron, train_x_bi, train_y, "BIPOLAR ACTIVATION", activation, lr,
                 init='range', weight_range=weight_range, run_num=run_num)
    else:
        test_run(neuron, train_x_uni, train_y, "UNIPOLAR ACTIVATION", activation, lr,
                 init='range', weight_range=weight_range, run_num=run_num)


def test_and_func(neuron, activation, lr, weight_range, run_num):
    print("AND FUNCTION")
    if activation == 'bipolar':
        train_y = np.array([-1, -1, -1, 1])
    else:
        train_y = np.array([0, 0, 0, 1])
    test_activation(train_y, neuron, activation, lr, weight_range, run_num)


def test_or_func(neuron, activation, lr, weight_range, run_num):
    print("OR FUNCTION")
    if activation == 'bipolar':
        train_y = np.array([-1, 1, 1, 1])
    else:
        train_y = np.array([0, 1, 1, 1])
    test_activation(train_y, neuron, activation, lr, weight_range, run_num)


def test_func(func, neuron, activation, lr, weight_range, run_num):
    if func == 'and':
        test_and_func(neuron, activation, lr, weight_range, run_num)
    else:
        test_or_func(neuron, activation, lr, weight_range, run_num)



if __name__ == '__main__':

    run_number = 150
    functions = ['and', 'or']

    for neuron in ('perceptron', 'adaline'):

        learning_rate = 0.1
        range = [-0.2, 0.2]
        print("NEURON:", neuron)
        actv = 'unipolar' if neuron == 'perceptron' else 'bipolar'

        # Weight range experiment
        print("---WEIGHT RANGE EXPERIMENT---")
        for fun in functions:
            initial_range = 1.0
            while initial_range >= 0:
                current_range = [-initial_range, initial_range]
                print("RANGE:", current_range)
                print("")
                test_func(fun, neuron, actv, learning_rate, current_range, run_number)
                initial_range = np.around(initial_range - 0.2, decimals=1)
                print("")

        # Learning rate experiment
        print("---LEARNING RATE EXPERIMENT---")
        for fun in functions:
            for lr in [0.5, 0.25, 0.1, 0.05, 0.01, 0.001]:
                print("LEARNING RATE:", lr)
                print("")
                if fun == 'and' and lr == 0.5:
                    continue
                if fun == 'or' and lr == 0.5 and neuron == 'adaline':
                    continue
                test_func(fun, neuron, actv, lr, range, run_number)
                print("")

        # Activation function experiment
        if neuron == 'perceptron':
            print("---ACTIVATION FUNCTION EXPERIMENT---")
            test_and_func(neuron, actv, learning_rate, range, run_number)
            print("")
            test_or_func(neuron, actv, learning_rate, range, run_number)
            print("")