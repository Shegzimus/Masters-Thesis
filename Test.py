import numpy as np
from IWOA import BaoWOA
from scipy.optimize import fsolve



def objective_function(params, V_measured, I_measured, T):
    Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3 = params
    I_calculated = np.array([triple_diode_model(v, Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3, T) for v in V_measured])
    rmse = np.sqrt(np.mean((I_measured - I_calculated) ** 2))
    return rmse


def triple_diode_model(V, Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3, T=330, q=1.602e-19, k=1.380e-23):
    """
    Solve for I using the fsolve method for a given voltage V.
    """
    def current_equation(I, V, Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3, T, q, k):
        return Iph - Isd1 * (np.exp(q * (V + Rs * I) / (n1 * k * T)) - 1) \
                   - Isd2 * (np.exp(q * (V + Rs * I) / (n2 * k * T)) - 1) \
                   - Isd3 * (np.exp(q * (V + Rs * I) / (n3 * k * T)) - 1) \
                   - (V + Rs * I) / Rsh - I
    # Assume initial guess for I
    I_initial_guess = 0
    I_solution, = fsolve(current_equation, I_initial_guess, args=(V, Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3, T, q, k))
    return I_solution

'''
V_measured = [39.614016, 39.867812, 39.956701, 47.278408, 43.490203, 39.735026]
I_measured = [5.235561, 4.190781, 3.144837, 2.043319, 2.070523, 2.097727]
'''
V_measured = 39.614016
I_measured = 5.235561

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





## Setting parameters
root_paras = {
    "problem_size": 9,
    "domain_range": bounds,
    "print_train": True,
    "objective_func": objective_function
}
woa_paras = {
    "epoch": 50,
    "pop_size": 6
}


optimizer = BaoWOA(objective_function, bounds=bounds)
results = optimizer.optimize(V_measured, I_measured, T)

'''
optimizer = BaoWOA()

# Set additional arguments for the objective function (V_measured, I_measured, T)
additional_args = (V_measured, I_measured, 330)  # Example temperature

# Run the optimization
best_params = optimizer.optimize(objective_function, bounds, additional_args)

print("Optimized Parameters:", best_params)
'''