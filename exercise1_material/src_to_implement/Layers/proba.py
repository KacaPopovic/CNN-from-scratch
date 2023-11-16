import numpy as np

import numpy as np

# Example arrays
array1 = np.random.randint(1, 6, size=(9, 5))  # Random integers between 1 and 5
array2 = np.random.randint(1, 6, size=(9, 5))
print(array1)
# Element-wise multiplication and sum along axis 1
result_vector = np.sum(array1 * array2, axis=1, keepdims=True)

# Print the result
print(result_vector)

subst = array1 - result_vector
print(subst)