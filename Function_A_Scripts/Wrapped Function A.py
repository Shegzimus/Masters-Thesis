import numpy as np
import matplotlib.pyplot as plt
from Function_A import function_A



# Set the number of total_iterations
total_iterations = 20
inner_iterations = 500

# Initialize lists to store counts and probabilities for each iteration
above_abs_1_counts = []
below_abs_1_counts = []
below_abs_1_probabilities = []
above_abs_1_probabilities = []

for iteration in range(total_iterations):
    # Linearly decreasing real number from 2 to 0
    x_values = np.linspace(2, 0, inner_iterations)

    # Initialize count variables for points above abs(1) and below abs(1)
    above_abs_1_count = 0
    below_abs_1_count = 0

    # Loop over 1000 iterations
    for i in range(inner_iterations):
        # Generate a random vector 'r' in the range [0, 1]
        random_vector = np.random.rand()

        # Calculate the output of function A for the current x and random vector
        result = function_A(x_values[i], random_vector)

        # Check if the result is above abs(1)
        if abs(result) > 1:
            above_abs_1_count += 1

        # Check if the result is below abs(1)
        if abs(result) < 1:
            below_abs_1_count += 1

    # Append counts to the respective lists
    above_abs_1_counts.append(above_abs_1_count)
    below_abs_1_counts.append(below_abs_1_count)

    # Calculate the probability for points being below abs(1)
    below_abs_1_probability = below_abs_1_count / inner_iterations
    below_abs_1_probabilities.append(below_abs_1_probability)

    #Calculate the probability for points being above abs(1)
    above_abs_1_probability = 1 - below_abs_1_probability
    above_abs_1_probabilities.append(above_abs_1_probability)

# Plot the counts as a function of the nth total_iterations
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, total_iterations + 1), above_abs_1_counts, marker='o', linestyle='-', color='r', label='Above abs|1|')
plt.plot(range(1, total_iterations + 1), below_abs_1_counts, marker='o', linestyle='-', color='b', label='Below abs|1|')
plt.title('Counts Above and Below |1|')
plt.xlabel('Iteration Number')
plt.ylabel('Count')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, total_iterations + 1), above_abs_1_probabilities, marker='o', linestyle='-', color='g', label='Probability (Above abs|1|)')
plt.title('Probability of Points Being Above abs|1|')
plt.xlabel('Iteration Number')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
