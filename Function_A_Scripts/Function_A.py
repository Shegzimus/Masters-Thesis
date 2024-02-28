import numpy as np
import matplotlib.pyplot as plt

# Define the function A
def function_A(a, r):
    return (2 * a * r) - a

# Set the number of iterations
iterations = 500

# Initialize an array to store the results
results = []

# Initialize a count variable for points above abs(1)
above_abs_1_count = 0

# Initialize a count variable for iterations between A = 1 and A = -1
between_1_and_minus_1_count = 0

# Loop over the number of iterations
for i in range(iterations):
    
    #a = 2 - 2 * i / (iterations - 1) # Linearly decreasing real number from 2 to 0
 
 
    #a = 2 - (2 * i / (iterations - 1)) ** 0.5   # Slower decrease of 'a'
  
    #a = 2 - np.log(1 + i) / np.log(iterations) # Logarithmic decrease

    #a = 2 / (1 + i / (iterations - 1))  # Inverse relationship

    #a = max(0.5, 2 - 2 * i / (iterations - 1))  # 'a' does not go below 0.5

    a = np.exp(-i / (iterations ))*2.5  # exponential

    #a = (1.4 - i / (iterations - 1))**2   # 'a' decreases quadratically

    #a = 1.5 + np.sin(i * np.pi / iterations)  # Oscillatory variation

    #a = 2 + np.sin(2 * np.pi * i / iterations) # Sinusoidal variation                     

    #a = 2 - 1 * i / (iterations - 1)  # a linearly decreases from 1 to 0

     

    # Generate a random vector 'r' in the range [0, 1]
    random_vector = np.random.rand()

    # Calculate the output of function A for the current a and random vector
    result = function_A(a, random_vector)

    # Append the result to the list
    results.append(result)

    # Check if the result is above abs(1)
    if abs(result) > 1:
        above_abs_1_count += 1

    # Check if the result is between A = 1 and A = -1
    if abs(result) < 1:
        between_1_and_minus_1_count += 1

# Display the count of points above abs(1)
print(f"A > |1|: {above_abs_1_count}")

# Display the count of iterations between A = 1 and A = -1
print(f"A < |1|: {between_1_and_minus_1_count}")

print('')

# x_values for the plot
x_values = np.linspace(0, iterations - 1, iterations)

# Plot the results
plt.plot(x_values, results, marker='o', linestyle='-', color='r')
#plt.title('Function A Results for '+ str(iterations) + ' iterations')
plt.xlabel('Iteration')
plt.ylabel('A')
plt.grid(True)

# Add dashed lines at A = 1 and A = -1
plt.axhline(y=1, color='black', linestyle='--', label='A = 1')
plt.axhline(y=-1, color='black', linestyle='--', label='A = -1')

plt.legend()
plt.show()
