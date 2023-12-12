from .Base import BaseLayer
import numpy as np


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.max_locations = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channel_number, y, x = self.input_tensor.shape

        y_out = (y - (y % self.pooling_shape[0])) // self.stride[0]
        x_out = (x - (x % self.pooling_shape[1])) // self.stride[1]

        output_tensor = np.zeros(shape=(batch_size, channel_number, y_out, x_out))
        self.max_locations = np.empty(output_tensor.shape, dtype=object)

        for b in range(batch_size):
            for c in range(channel_number):
                for i in range(0, y_out*self.stride[0], self.stride[0]):
                    for j in range(0, x_out*self.stride[1], self.stride[1]):
                        window = self.input_tensor[b, c, i:i + self.pooling_shape[0], j:j + self.pooling_shape[1]]
                        max_value = np.max(window)
                        output_tensor[b, c, i // self.stride[0], j // self.stride[1]] = max_value
                        max_indices = np.unravel_index(np.argmax(window), window.shape)
                        self.max_locations[b][c][i // self.stride[0]][j // self.stride[1]] = [i + max_indices[0], j + max_indices[1]]

        return output_tensor

    def backward(self, error_tensor):
        previous_error_tensor = np.zeros_like(self.input_tensor)
        batch_size, channel_number, y, x = error_tensor.shape

        for b in range(batch_size):
            for c in range(channel_number):
                for i in range(y):
                    for j in range(x):
                        previous_error_tensor[b, c, self.max_locations[b, c, i, j][0], self.max_locations[b, c, i, j][1]] += error_tensor[b, c, i, j]

        return previous_error_tensor
