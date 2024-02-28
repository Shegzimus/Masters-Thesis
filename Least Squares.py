import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import fsolve

def triple_diode_model(V, Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3, T=330, q=1.602e-19, k=1.380e-23):
    """
    Solve for I using the fsolve method for a given voltage V.
    """
    def current_equation(I, V, Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3, T, q, k):
        return Iph - Isd1 * (np.exp(q * (V + Rs * I) / (n1 * k * T)) - 1) \
                   - Isd2 * (np.exp(q * (V + Rs * I) / (n2 * k * T)) - 1) \
                   - Isd3 * (np.exp(q * (V + Rs * I) / (n3 * k * T)) - 1) \
                   - (V + Rs * I) / Rsh - I
    
    I_initial_guess = 0
    I_solution, = fsolve(current_equation, I_initial_guess, args=(V, Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3, T, q, k))
    return I_solution

def objective_function(params, V_measured, I_measured):
    Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3 = params
    I_predicted = [triple_diode_model(V, Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3) for V in V_measured]
    return np.array(I_predicted) - np.array(I_measured)

# Initial guess for the parameters
initial_guess = [0.1, 50, 0.5, 1e-4, 1e-4, 1e-4, 1.5, 1.5, 1.5]


V_measured = [39.614016, 39.867812, 39.956701, 47.278408, 43.490203, 39.735026]
I_measured = [5.235561, 4.190781, 3.144837, 2.043319, 2.070523, 2.097727]


# Perform the optimization
result = least_squares(objective_function, initial_guess, args=(V_measured, I_measured), bounds=([0, 0, 0, 0, 0, 0, 1, 1, 1], [0.5, 100, 1, 1e-3, 1e-3, 1e-3, 2, 2, 2]))

print("Optimized Parameters:", result.x)
