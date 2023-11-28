import numpy as np


class CrossEntropyLoss:
    """
    Loss class which computes the loss value according to cross entropy, accumulated over the batch.

    Attributes:
        prediction_tensor(np.ndarray): Tensor with predictions our Neural Network made.
        loss(float): Value of the loss that our batch generates.
    """
    def __init__(self):
        self.prediction_tensor = None
        self.loss = None

    def forward(self, prediction_tensor, label_tensor):
        """
        Calculates the loss in the forward pass.

        Args:
            prediction_tensor(np.ndarray): Tensor with predicted probabilities.
            label_tensor(np.ndarray): Tensor with true labels.

        Returns:
            float: Loss accumulated in the batch.
        """
        self.prediction_tensor = prediction_tensor
        # We generate the smallest representable number, which we use to increase stability (no log(0)).
        eps = np.finfo(float).eps
        neg_log_probs = -np.log(prediction_tensor + eps)
        loss = np.sum(neg_log_probs * label_tensor)
        self.loss = loss
        return loss

    def backward(self, label_tensor):
        """
        Calculates the error for the previous layer, in the backward pass.

        Args:
            label_tensor(np.ndarray): Tensor with true labels.

        Returns:
            np.ndarray: Tensor with errors.
        """
        eps = np.finfo(float).eps
        prev_error_tensor = -label_tensor/(self.prediction_tensor+eps)
        return prev_error_tensor
