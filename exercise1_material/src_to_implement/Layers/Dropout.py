import numpy as np

from . Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.dropout_mask = None

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            dropout_mask = np.random.binomial(1/self.probability, self.probability, size=input_tensor.shape)
            self.dropout_mask = dropout_mask
            return dropout_mask*input_tensor
    def backward(self, error_tensor):
        return error_tensor * self.dropout_mask