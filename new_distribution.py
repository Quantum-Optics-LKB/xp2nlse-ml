import numpy as np
import matplotlib.pyplot as plt

# Define the custom function for the desired distribution
def custom_distribution(array, a):
    normalized_array = (array - np.min(array))/(np.max(array) - np.min(array))

    normalized_array = normalized_array**(a) / (normalized_array**(a) + (1-normalized_array)**a)

    new_array = normalized_array * (np.max(array) - np.min(array)) + np.min(array)
    return new_array

# Create a linearly spaced array from 0 to 1
linear_array = np.linspace(15, 27, 50)
# Apply the custom transformation function
a = 1.5
transformed_array = custom_distribution(linear_array, a)

print(transformed_array)
print(linear_array)
# Plotting the results
plt.figure(figsize=(50, 2))
plt.scatter(linear_array, np.zeros_like(transformed_array), c='red', marker='o')
plt.scatter(transformed_array, np.zeros_like(transformed_array), c='blue', marker='+')
plt.yticks([])
plt.xlabel('Value')
plt.title('1D Scatter Plot of Custom Distribution')
plt.grid(True)
plt.show()
