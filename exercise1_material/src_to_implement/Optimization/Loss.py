import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.loss = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        eps = np.finfo(float).eps
        neg_log_probs = -np.log(prediction_tensor + eps)
        loss = np.sum(neg_log_probs * label_tensor)
        self.loss = loss

        # todo check if you need to return it
        return loss


    def backward(self, label_tensor):
        eps = np.finfo(float).eps
        prev_error_tensor = -label_tensor/(self.prediction_tensor+eps)
        return prev_error_tensor
