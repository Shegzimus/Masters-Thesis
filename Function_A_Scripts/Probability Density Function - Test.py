import numpy as np
import matplotlib.pyplot as plt

def function_A(a, r):
    return 2 * a * r - a

# Set the range and step for 'a' values
a_values = np.linspace(0, 2, 50)  # Varying 'a' from 0 to 2
inner_iterations = 1000  # Number of iterations for each 'a'

# List to store probabilities for each 'a'
probabilities = []

for a in a_values:
    below_abs_1_count = 0

    for _ in range(inner_iterations):
        random_vector = np.random.rand()
        result = function_A(a, random_vector)

        if abs(result) < 1:
            below_abs_1_count += 1

    # Calculate the probability for this 'a'
    probability = below_abs_1_count / inner_iterations
    probabilities.append(probability)

# Plotting the Probability Density Function
plt.figure(figsize=(10, 6))
plt.plot(a_values, probabilities, marker='o', linestyle='-', color='b')
plt.title('Probability Density Function')
plt.xlabel('Value of a')
plt.ylabel('Probability of Result Being Below abs(1)')
plt.grid(True)
plt.show()
