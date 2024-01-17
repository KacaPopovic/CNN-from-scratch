import unittest
try:
    LSTM_TEST = True
    from Layers import *
except BaseException as e:
    if str(e)[-6:] == "'LSTM'":
        LSTM_TEST = False
    else:
        raise e
from Optimization import *
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
import NeuralNetwork
import matplotlib.pyplot as plt
import os
import argparse
import tabulate

ID = 3  # identifier for dispatcher

class L2Loss:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)


class TestFullyConnected(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 4
        self.output_size = 3
        self.input_tensor = np.random.rand(self.batch_size, self.input_size)

        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    class TestInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros(shape)
            weights[0] = 1
            weights[1] = 2
            return weights

    def test_trainable(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        self.assertTrue(layer.trainable)

    def test_weights_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        self.assertTrue(
            (layer.weights.shape) in ((self.input_size + 1, self.output_size), (self.output_size, self.input_size + 1)))

    def test_forward_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size)
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_backward_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_update(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_update_bias(self):
        input_tensor = np.zeros([self.batch_size, self.input_size])
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(input_tensor)
            error_tensor = np.zeros([self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = FullyConnected.FullyConnected(100000, 1)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0)

    def test_initialization(self):
        input_size = 4
        categories = 10
        layer = FullyConnected.FullyConnected(input_size, categories)
        init = TestFullyConnected.TestInitializer()
        layer.initialize(init, Initializers.Constant(0.5))
        self.assertEqual(init.fan_in, input_size)
        self.assertEqual(init.fan_out, categories)
        if layer.weights.shape[0] > layer.weights.shape[1]:
            self.assertLessEqual(np.sum(layer.weights) - 17, 1e-5)
        else:
            self.assertLessEqual(np.sum(layer.weights) - 35, 1e-5)


class TestRNN(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 13
        self.output_size = 5
        self.hidden_size = 7
        self.input_tensor = np.random.rand(self.input_size, self.batch_size).T

        self.categories = 4
        self.label_tensor = np.zeros([self.categories, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_initialization(self):
        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)
        init = TestFullyConnected.TestInitializer()
        layer.initialize(init, Initializers.Constant(0.0))
        if layer.weights.shape[0] > layer.weights.shape[1]:
            self.assertEqual(np.sum(layer.weights), 21.0, msg="Make sure to provide a property named 'weights' that allows"
                                                          " to access the weights of the first FC layer.")
        else:
            self.assertEqual(np.sum(layer.weights), 60.0, msg="Make sure to provide a property named 'weights' that allows"
                                                          " to access the weights of the first FC layer.")

    def test_trainable(self):
        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)
        self.assertTrue(layer.trainable, "Error: trainable flag set to false.")

    def test_forward_size(self):
        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size, msg="Possible error: wrong weights' shapes in "
                                                                       "one of the FC layers.")
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_forward_stateful(self):
        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)

        input_vector = np.random.rand(self.input_size, 1).T
        input_tensor = np.tile(input_vector, (2, 1))

        output_tensor = layer.forward(input_tensor)

        self.assertNotEqual(np.sum(np.square(output_tensor[0, :] - output_tensor[1, :])), 0,
                            msg="Possible error: hidden state is not updated.")

    def test_forward_stateful_TBPTT(self):
        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)
        layer.memorize = True
        output_tensor_first = layer.forward(self.input_tensor)
        output_tensor_second = layer.forward(self.input_tensor)
        self.assertNotEqual(np.sum(np.square(output_tensor_first - output_tensor_second)), 0,
                            msg="Possible error: hidden state is not updated.")

    def test_backward_size(self):
        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_update(self):
        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([self.output_size, self.batch_size]).T
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        layers = list()
        layer = RNN.RNN(self.input_size, self.hidden_size, self.categories)
        layer.initialize(Initializers.He(), Initializers.He())
        layers.append(layer)
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4,
                             msg="To compute the gradient we recommend using the 'backward' methods"
                                 "of each layer. In order to use the backward method, we need to restore the correct"
                                 " 'state' of each layer, that is we need to set the activations of that layer to the "
                                 "correct ones (at each time step t these activations were different). Once all "
                                 "activations are set correctly we can proceed and compute the gradient by going back"
                                 " through time: starting from time step T going back to time step T-1, T-2, ...,  0."
                                 "In this way we will compute the gradient at each time step. The result should be then"
                                 " reversed such that we obtain the gradient starting from time step 0.")

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        layers = list()
        layer = RNN.RNN(self.input_size, self.hidden_size, self.categories)
        layer.initialize(Initializers.He(), Initializers.He())
        layers.append(layer)
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-4, msg="If you have implemented the structure correctly, "
                                                           "the gradient weights will be computed automatically by the "
                                                           "FC layers when you call the 'backward' method. In doing so "
                                                           "we have a different value of the gradient_weights for each "
                                                           "time step. The gradient_weights used to compute the update "
                                                           "for the FC layers is the sum of all gradients at each step."
                                                           "Thus, you should accumulate the gradient_weights when going"
                                                           " back through time and use the sum to compute the update."
                                                           " Remember that you can access these gradient using the "
                                                           "property named 'gradient_weights'. ")

    def test_weights_shape(self):
        layer = RNN.RNN(self.input_size, self.hidden_size, self.categories)
        layer.initialize(Initializers.He(), Initializers.He())
        self.assertTrue(hasattr(layer, 'weights'), msg='your RNN layer does not provide a weights attribute')
        fc_layer = FullyConnected.FullyConnected(20, 7)
        self.assertIn(layer.weights.shape, [fc_layer.weights.shape, fc_layer.weights.T.shape])

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = RNN.RNN(100000, 100, 1)
        layer.initialize(Initializers.UniformRandom(), Initializers.UniformRandom())
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0, "Possible error: bias in the FC layers is not implemented.")


if __name__ == "__main__":
    unittest.main()
