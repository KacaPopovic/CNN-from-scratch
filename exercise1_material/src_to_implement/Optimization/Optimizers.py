import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    """
    Optimizer that updates the weight tensor using stochastic gradient descent.

    Attributes:
        learning_rate(float): Step with which we calculate the update.
    """
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Updates weight tensor.

        Args:
            weight_tensor(np.ndarray): Tensor with weights of the layer.
            gradient_tensor(np.ndarray): Tensor with calculated gradient values for the layer.

        Returns:
            np.ndarray: Tensor with updated weight for the layer.
        """
        if self.regularizer:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        updated_weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weight_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.momentum = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        if self.momentum is None:
            self.momentum = np.zeros_like(weight_tensor)
        self.momentum = self.momentum_rate * self.momentum - self.learning_rate * gradient_tensor
        updated_weight_tensor = weight_tensor + self.momentum
        return updated_weight_tensor


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.momentum = None
        self.second_order_momentum = None
        self.iteration = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.iteration += 1
        eps = np.finfo(float).eps
        if self.regularizer:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        if self.momentum is None:
            self.momentum = np.zeros_like(weight_tensor)
        if self.second_order_momentum is None:
            self.second_order_momentum = np.zeros_like(weight_tensor)
        self.momentum = self.mu * self.momentum + (1 - self.mu) * gradient_tensor
        self.second_order_momentum = self.rho * self.second_order_momentum + (1 - self.rho) * gradient_tensor * gradient_tensor
        momentum_corrected = self.momentum / (1 - self.mu**self.iteration)
        second_order_momentum_corrected = self.second_order_momentum / (1 - self.rho**self.iteration)
        updated_weight_tensor = weight_tensor - self.learning_rate * momentum_corrected / (np.sqrt(second_order_momentum_corrected) + eps)
        return updated_weight_tensor
