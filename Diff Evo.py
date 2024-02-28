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

def rmse_objective_function(params, V_measured, I_measured):
    Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3 = params
    I_predicted = [triple_diode_model(V, Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3) for V in V_measured]
    mse = np.mean((np.array(I_predicted) - np.array(I_measured))**2)
    rmse = np.sqrt(mse)
    return rmse

# Initial guess for the parameters
initial_guess = [0.1, 50, 0.5, 1e-4, 1e-4, 1e-4, 1.5, 1.5, 1.5]

# Since least_squares expects the residuals (and not a single value like RMSE),
# we should use a general optimizer for this case. Let's use differential_evolution.
from scipy.optimize import differential_evolution

bounds = [
    (0, 0.5),  # Bounds for Rs
    (0, 100),  # Bounds for Rsh
    (0, 1),    # Bounds for Iph
    (0, 1e-3), # Bounds for Isd1
    (0, 1e-3), # Bounds for Isd2
    (0, 1e-3), # Bounds for Isd3
    (1, 2),    # Bounds for n1
    (1, 2),    # Bounds for n2
    (1, 2)     # Bounds for n3
]


V_measured = [39.614016, 39.867812, 39.956701, 47.278408, 43.490203, 39.735026]
I_measured = [5.235561, 4.190781, 3.144837, 2.043319, 2.070523, 2.097727]


# Perform the optimization
result = differential_evolution(rmse_objective_function, bounds, args=(V_measured, I_measured))

print("Optimized Parameters:", result.x)
