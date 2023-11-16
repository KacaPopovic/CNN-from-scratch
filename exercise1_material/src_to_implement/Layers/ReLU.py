from . Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    def _init_(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = np.maximum(0, input_tensor)
        return output_tensor

    def backward(self, error_tensor):
        previous_error_tensor = np.zeros_like(error_tensor)
        previous_error_tensor[self.input_tensor > 0] = error_tensor[self.input_tensor > 0]
        return previous_error_tensor