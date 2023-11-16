import copy

import numpy as np


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.current_loss = None

    def forward(self):

        # todo delete unnecessary
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)
        #loss = self.layers[-1].forward(input_tensor, label_tensor)
        output_tensor = input_tensor
        #predicted_class = np.argmax(output_tensor, axis=1)
        self.current_loss = self.loss_layer.forward(self.label_tensor, output_tensor)
        return self.current_loss

    def backward(self):
        error_tensor = self.layers[-1].backward(self.label_tensor)
        for i in range(len(self.layers)-1, -1, -1):
            error_tensor = self.layers[i].backward(error_tensor)
        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            optimizer_copy = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer_copy
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            #loss = self.forward()
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        for i in range(len(self.layers)-1):
            input_tensor = self.layers[i].forward(input_tensor)
        probabilities = input_tensor
        return probabilities

