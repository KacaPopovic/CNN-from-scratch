import copy

import numpy as np


class NeuralNetwork:
    """
    Neural Network Skeleton.

    Attributes:
        optimizer(Sgd): Optimizer for weights tensor updating. It is a protected member of the class.
        loss(list): List containing loss values for every training iteration.
        layers(list): List containing all the layers in our neural network.
        data_layer(Data): Layer containing input data and labels.
        loss_layer(CrossEntropyLoss): Layer for calculating the loss.
        label_tensor(np.ndarray): Tensor with true labels.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.current_loss = None

    def forward(self):
        """
        Passes input data forward through all the layers of our neural network.

        Returns:
            float: Loss accumulated in the batch.
        """
        # We load the data from data layer.
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        # We pass forward through all the layers
        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)
        output_tensor = input_tensor
        self.current_loss = self.loss_layer.forward(output_tensor, self.label_tensor)
        return self.current_loss

    def backward(self):
        """
        Passes error tensor backward through all the layers of our neural network,
        and updates weight tensor, if we have an optimizer.

        Returns:
            np.ndarray: Tensor with errors.
        """
        # First to compute the error is loss layer, which is why it receives the label tensor.
        error_tensor = self.loss_layer.backward(self.label_tensor)
        # We go backward through all the layers and propagate the error.
        for i in range(len(self.layers)-1, -1, -1):
            error_tensor = self.layers[i].backward(error_tensor)
        return error_tensor

    def append_layer(self, layer):
        """
        Adding new layer to the neural network.

        Args:
        later(BaseLayer): New layer to append.

        Returns:
            None
        """
        if layer.trainable:
            optimizer_copy = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer_copy
        self.layers.append(layer)

    def train(self, iterations):
        """
        Trains the neural network by performing forward and backward pass, iterations number of times.

        Args:
            iterations(int): Number of times we should perform forward and backward pass on the batch.

        Returns:
            None
        """
        # We save the loss calculated in each iteration.
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        """
        Trains the neural network by performing forward and backward pass, iterations number of times.

        Args:
            input_tensor(np.ndarray): Tensor to test the neural network on.

        Returns:
            np.ndarray: Tensor with predicted probabilities.
        """
        # We propagate the input tensor through all the layers.
        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)
        output_tensor = input_tensor
        return output_tensor
