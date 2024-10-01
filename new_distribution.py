import numpy as np
import matplotlib.pyplot as plt

def custom_distribution(array, a):
    normalized_array = (array - np.min(array))/(np.max(array) - np.min(array))

    normalized_array = normalized_array**(a) / (normalized_array**(a) + (1-normalized_array)**a)

    new_array = normalized_array * (np.max(array) - np.min(array)) + np.min(array)
    return new_array

random_array = np.random.uniform(15, 27, 50)
linear_array = np.linspace(15, 27, 50)

import numpy as np
import matplotlib.pyplot as plt

# Define the custom function for the desired distribution
def custom_distribution(array, a):
    normalized_array = (array - np.min(array))/(np.max(array) - np.min(array))
    normalized_array = normalized_array**(a) / (normalized_array**(a) + (1 - normalized_array)**a)
    new_array = normalized_array * (np.max(array) - np.min(array)) + np.min(array)
    return new_array

# Create arrays
random_array = np.random.uniform(15, 27, 50)
linear_array = np.linspace(15, 27, 50)
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Subplot for linear_array
axs[0].plot(np.sort(random_array), '.', label='Random')
axs[0].plot(linear_array, 'r', label='Trend')
axs[1].plot(linear_array, '.', label='Linear')
axs[0].legend()
axs[1].legend()

plt.tight_layout()
plt.show()
