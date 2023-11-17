
import numpy as np

# Example arrays
array1 = np.random.randint(1, 6, size=(9, 5))  # Random integers between 1 and 5
array2 = np.random.randint(1, 6, size=(9, 5))

prediction_tensor = np.random.randint(1, 10, size=(4,4))/10
print("Pred tensor:")
print(prediction_tensor)
label_tensor = np.eye(4,4)
eps = np.finfo(float).eps

print("Label tensor: ")
print(label_tensor)
eps = np.finfo(float).eps
neg_log_probs = -np.log(prediction_tensor + eps)
loss = np.sum(neg_log_probs * label_tensor)
print("loss")
print(loss)