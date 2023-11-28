class Sgd:
    """
    Optimizer that updates the weight tensor using stochastic gradient descent.

    Attributes:
        learning_rate(float): Step with which we calculate the update.
    """
    def __init__(self, learning_rate: float):
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
        updated_weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weight_tensor
