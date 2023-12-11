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
        self.max_locations = np.zeros_like(self.input_tensor)
        batch_size, channel_number, y, x = self.input_tensor.shape

        y_out = (y - self.pooling_shape[0]) // self.stride[0] + 1
        x_out = (x - self.pooling_shape[1]) // self.stride[1] + 1

        output_tensor = np.zeros(shape=(batch_size, channel_number, y_out, x_out))

        for b in range(batch_size):
            for c in range(channel_number):
                for i in range(0, y_out*self.stride[0], self.stride[0]):
                    for j in range(0, x_out*self.stride[1], self.stride[1]):
                        window = self.input_tensor[b, c, i:i + self.pooling_shape[0], j:j + self.pooling_shape[1]]
                        max_value = np.max(window)
                        output_tensor[b, c, i // self.stride[0], j // self.stride[1]] = max_value
                        max_indices = np.unravel_index(np.argmax(window), window.shape)
                        self.max_locations[b, c, i + max_indices[0], j + max_indices[1]] = 1

        return output_tensor

    def backward1(self, error_tensor):
        batch, channel, y, x = self.input_tensor.shape
        prev_error_tensor = np.zeros_like(self.input_tensor)

        for b in range(batch):
            for c in range(channel):
                for i in range(error_tensor.shape[2]):
                    for j in range(error_tensor.shape[3]):
                        start_y = i * self.stride[0]
                        end_y = min(start_y + self.pooling_shape[0], y)
                        start_x = j * self.stride[1]
                        end_x = min(start_x + self.pooling_shape[1], x)

                        # Find the location of the max value in the original pooling region
                        max_pos = np.where(self.max_locations[b, c, start_y:end_y, start_x:end_x])

                        # Propagate the error to the location of the max value
                        for k in range(len(max_pos[0])):
                            y_max = start_y + max_pos[0][k]
                            x_max = start_x + max_pos[1][k]
                            prev_error_tensor[b, c, y_max, x_max] += error_tensor[b, c, i, j]

        return prev_error_tensor

    def backward(self, error_tensor):
        batch, channel, y, x = self.input_tensor.shape
        prev_error_tensor = np.zeros_like(self.input_tensor)

        for b in range(batch):
            for c in range(channel):
                for i in range(error_tensor.shape[2]):
                    for j in range(error_tensor.shape[3]):
                        start_y = i * self.stride[0]
                        end_y = start_y + self.pooling_shape[0]
                        start_x = j * self.stride[1]
                        end_x = start_x + self.pooling_shape[1]

                        # Ensure we don't go past the edge of the input tensor
                        end_y = min(end_y, y)
                        end_x = min(end_x, x)

                        # Find the location of the max value in the original pooling region
                        max_pos = np.where(self.max_locations[b, c, start_y:end_y, start_x:end_x])

                        # Check if there is at least one max location
                        if max_pos[0].size > 0:
                            # Take the first occurrence in row-wise order
                            y_max = start_y + max_pos[0][0]
                            x_max = start_x + max_pos[1][0]

                            # Propagate the error to the location of the max value
                            prev_error_tensor[b, c, y_max, x_max] += error_tensor[b, c, i, j]

        return prev_error_tensor



