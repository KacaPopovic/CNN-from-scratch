from .Base import BaseLayer
import numpy as np
from scipy.signal import correlate, convolve
import copy
from scipy.ndimage import zoom

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None
        self.padding_shape = None
        self._weights_optimizer = None
        self._bias_optimizer = None
        self.input_tensor = None

    @property
    def optimizer(self):
        return self._weights_optimizer, self._bias_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._weights_optimizer = copy.deepcopy(optimizer)
        self._bias_optimizer = copy.deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=3)
            expanded = True
        else:
            expanded = False
        b, c, y, x = input_tensor.shape
        if len(self.convolution_shape) == 2:
            convolution_shape = self.convolution_shape + (1,)
        else:
            convolution_shape = self.convolution_shape
        if len(self.stride_shape) == 1:
            stride_shape = self.stride_shape * 2
        else:
            stride_shape = self.stride_shape

        if self.weights.ndim == 3:
            weights = np.expand_dims(self.weights, axis=-1)
        else:
            weights = self.weights

        output_tensor = np.zeros([b, self.num_kernels, y, x])

        for i in range(b):
            for k in range(self.num_kernels):
                for channel in range(c):
                    output_tensor[i, k, :, :] += correlate(input_tensor[i, channel, :, :], weights[k, channel, :, :], mode="same", method="direct")
                output_tensor[i,k,:,:] += self.bias[k]

        output_tensor = output_tensor[:,:,::stride_shape[0],::stride_shape[1]]

        if expanded:
            output_tensor = np.squeeze(output_tensor, axis=-1)
        return output_tensor

    def initialize(self, weight_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weight_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    def calculate_gradients(self, error_tensor, input_tensor, convolution_shape, weights):

        error_tensor = self.upsample_error_tensor(error_tensor, input_tensor)
        weights_gradient_tensor = np.zeros_like(weights)
        bias_gradient_tensor = np.zeros(shape =error_tensor.shape[1])

        # TODO dodaj even
        padding_y = (convolution_shape[1] - 1) // 2, convolution_shape[1] // 2
        padding_x = (convolution_shape[2] - 1) // 2, convolution_shape[2] // 2
        padded_input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), padding_y, padding_x), mode='constant')
        for i in range(error_tensor.shape[1]):
            for j in range(input_tensor.shape[1]):
                for b in range(input_tensor.shape[0]):
                    temp = convolve(padded_input_tensor[b,j,:,:], error_tensor[b,i,:,:], mode="valid")
                    weights_gradient_tensor[i, j, : , :] += convolve(padded_input_tensor[b,j,:,:], error_tensor[b,i,:,:], mode="valid")
            bias_gradient_tensor[i] = np.sum(error_tensor[:,i,:,:])
        return weights_gradient_tensor, bias_gradient_tensor

    def upsample_error_tensor(self, error_tensor, input_tensor):
        _, _, height, width = input_tensor.shape
        batch_size, channels, reduced_height, reduced_width = error_tensor.shape

        scale_factor_y = height / reduced_height
        scale_factor_x = width / reduced_width

        error_tensor_upsampled = np.zeros(shape=(batch_size,channels,height,width))

        # Upsample each batch and channel independently
        for b in range(batch_size):
            for c in range(channels):
                error_tensor_upsampled[b, c, :, :] = zoom(error_tensor[b, c, :, :], (scale_factor_y, scale_factor_x),
                                                          order=1)  # bilinear interpolation

        return error_tensor_upsampled

    def calculate_prev_error_tensor(self, error_tensor, input_tensor, convolution_shape, weights):
        flipped_weights = np.swapaxes(weights, 0, 1)
        previous_error_tensor = np.zeros_like(input_tensor)
        error_tensor = self.upsample_error_tensor(error_tensor, input_tensor)
        for b in range(input_tensor.shape[0]):
            for k in range(flipped_weights.shape[0]):
                for channel in range(flipped_weights.shape[1]):
                    previous_error_tensor[b, k, :, :] += convolve(error_tensor[b, channel, :, :],
                                                                  flipped_weights[k, channel, :, :], mode="same")

        return previous_error_tensor

    def backward(self, error_tensor):
        if len(self.input_tensor.shape) == 3:
            input_tensor = np.expand_dims(self.input_tensor, axis=3)
            expanded = True
        else:
            input_tensor = self.input_tensor
            expanded = False
        if len(error_tensor.shape) == 3:
            error_tensor = np.expand_dims(error_tensor, axis=3)
            expanded = True
        else:
            expanded = False
        b, c, y, x = error_tensor.shape
        if len(self.stride_shape) == 1:
            stride_shape = self.stride_shape * 2
        else:
            stride_shape = self.stride_shape
        if len(self.convolution_shape) == 2:
            convolution_shape = self.convolution_shape + (1,)
        else:
            convolution_shape = self.convolution_shape
        if self.weights.ndim == 3:
            weights = np.expand_dims(self.weights, axis=-1)
        else:
            weights = self.weights

        previous_error_tensor = self.calculate_prev_error_tensor(error_tensor, input_tensor, convolution_shape, weights)
        weights_gradient_tensor, bias_gradient_tensor = self.calculate_gradients(error_tensor, input_tensor, convolution_shape, weights)

        if expanded:
            weights_gradient_tensor = np.squeeze(weights_gradient_tensor, axis=-1)
            previous_error_tensor = np.squeeze(previous_error_tensor, axis=-1)

        if self._weights_optimizer:
            self.weights = self._weights_optimizer.calculate_update(self.weights, weights_gradient_tensor)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, bias_gradient_tensor)

        return previous_error_tensor
