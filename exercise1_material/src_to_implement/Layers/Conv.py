from . Base import BaseLayer
from scipy.signal import correlate, convolve
from scipy.ndimage import zoom
import numpy as np
import copy


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.stride = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)
        self.input_tensor = None
        self._gradient_weights = None
        self._gradient_bias = None
        self.padding_shape = None
        self._weight_optimizer = None
        self._bias_optimizer = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._weight_optimizer, self._bias_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._weight_optimizer = copy.deepcopy(optimizer)
        self._bias_optimizer = copy.deepcopy(optimizer)

    def initialize(self, weight_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weight_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, self.convolution_shape[0], self.num_kernels)

    def preprocess(self, error_tensor=None, mode=0):
        # if mode = 0 -> forward
        # if mode = 1 -> backward

        if len(self.input_tensor.shape) == 3:
            input_tensor = np.expand_dims(self.input_tensor, axis=3)
            expanded = True
        else:
            input_tensor = self.input_tensor
            expanded = False

        if len(self.stride) == 1:
            stride = self.stride * 2
        else:
            stride = self.stride

        if len(self.convolution_shape) == 2:
            convolution_shape = self.convolution_shape + (1, )
        else:
            convolution_shape = self.convolution_shape

        if self.weights.ndim == 3:
            weights = np.expand_dims(self.weights, axis=-1)
        else:
            weights = self.weights

        if mode == 1:
            if len(error_tensor.shape) == 3:
                error_tensor = np.expand_dims(error_tensor, axis=3)

        return input_tensor, stride, convolution_shape, weights, error_tensor, expanded

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        input_tensor, stride, _, weights, _, expanded = self.preprocess(0)
        b, c, y, x = input_tensor.shape

        output_tensor = np.zeros([b, self.num_kernels, y, x])

        for i in range(b):
            for k in range(self.num_kernels):
                for channel in range(c):
                    output_tensor[i, k, :, :] += correlate(input_tensor[i, channel, :, :], weights[k, channel, :, :], mode="same", method="direct")
                output_tensor[i, k, :, :] += self.bias[k]
        output_tensor = output_tensor[:, :, ::stride[0], ::stride[1]]

        if expanded:
            output_tensor = np.squeeze(output_tensor, axis=-1)

        return output_tensor

    def upsample_tensor(self, error_tensor, input_tensor):
        stride = self.stride
        if len(self.stride) == 1:
            stride = self.stride * 2
        _, _, height, width = input_tensor.shape
        batch_size, channels, reduced_height, reduced_width = error_tensor.shape

        scale_factor_y = stride[0]
        scale_factor_x = stride[1]

        error_tensor_upsampled = np.zeros(shape=(batch_size, channels, height, width))

        # Fill the upscaled tensor with the original error values at the correct positions
        for b in range(batch_size):
            for c in range(channels):
                for y in range(reduced_height):
                    for x in range(reduced_width):
                        new_y = y * scale_factor_y
                        new_x = x * scale_factor_x
                        error_tensor_upsampled[b, c, new_y, new_x] = error_tensor[b, c, y, x]

        return error_tensor_upsampled

    # def upsample_tensor(self, error_tensor, input_tensor):
    #     _, _, height, width = input_tensor.shape
    #     batch_size, channels, reduced_height, reduced_width = error_tensor.shape
    #
    #     scale_factor_y = height / reduced_height
    #     scale_factor_x = width / reduced_width
    #
    #     error_tensor_upsampled = np.zeros(shape=(batch_size, channels, height, width))
    #
    #     # Upsample each batch and channel independently
    #     for b in range(batch_size):
    #         for c in range(channels):
    #             error_tensor_upsampled[b, c, :, :] = zoom(error_tensor[b, c, :, :], (scale_factor_y, scale_factor_x),
    #                                                       order=1)  # bilinear interpolation
    #
    #     return error_tensor_upsampled

    def calculate_gradients(self, error_tensor, input_tensor, convolution_shape, weights):

        error_tensor = self.upsample_tensor(error_tensor, input_tensor)

        weights_gradient_tensor = np.zeros_like(weights)
        bias_gradient_tensor = np.zeros(shape=error_tensor.shape[1])

        padding_y = (convolution_shape[1] - 1) // 2, convolution_shape[1] // 2
        padding_x = (convolution_shape[2] - 1) // 2, convolution_shape[2] // 2
        padded_input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), padding_y, padding_x), mode='constant')

        for i in range(error_tensor.shape[1]):
            for j in range(input_tensor.shape[1]):
                for b in range(input_tensor.shape[0]):
                    weights_gradient_tensor[i, j, :, :] += correlate(padded_input_tensor[b, j, :, :], error_tensor[b, i, :, :], mode='valid')
            bias_gradient_tensor[i] = np.sum(error_tensor[:, i, :, :])

        # error_spatial_sum = np.sum(error_tensor, axis=(2, 3))
        # bias_gradient_tensor = np.sum(error_spatial_sum, axis=0)

        return weights_gradient_tensor, bias_gradient_tensor

    def calculate_previous_error_tensor(self, error_tensor, input_tensor, convolution_shape, weights):
        flipped_weights = np.swapaxes(weights, 0, 1)
        previous_error_tensor = np.zeros_like(input_tensor)
        error_tensor = self.upsample_tensor(error_tensor, input_tensor)
        for b in range(input_tensor.shape[0]):
            for k in range(flipped_weights.shape[0]):
                for channel in range(flipped_weights.shape[1]):
                    previous_error_tensor[b, k, :, :] += convolve(error_tensor[b, channel, :, :], flipped_weights[k, channel, :, :], mode='same')

        return previous_error_tensor

    def backward(self, error_tensor):

        input_tensor, _, convolution_shape, weights, error_tensor, expanded = self.preprocess(error_tensor, 1)

        # Calculate the gradient
        weights_gradient_tensor, bias_gradient_tensor = self.calculate_gradients(error_tensor, input_tensor, convolution_shape, weights)
        self._gradient_weights = weights_gradient_tensor
        self._gradient_bias = bias_gradient_tensor

        # Calculate previous error tensor
        previous_error_tensor = self.calculate_previous_error_tensor(error_tensor, input_tensor, convolution_shape, weights)

        if expanded:
            weights_gradient_tensor = np.squeeze(weights_gradient_tensor, axis=-1)
            previous_error_tensor = np.squeeze(previous_error_tensor, axis=-1)

        # Perform the update
        if self._weight_optimizer:
            self.weights = self._weight_optimizer.calculate_update(self.weights, weights_gradient_tensor)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, bias_gradient_tensor)

        return previous_error_tensor
