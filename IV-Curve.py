import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Definition of the triple diode model remains the same
def triple_diode_model(V, Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3, T=330, q=1.602e-19, k=1.380e-23):
    def current_equation(I, V, Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3, T, q, k):
        return Iph - Isd1 * (np.exp(q * (V + Rs * I) / (n1 * k * T)) - 1) \
                   - Isd2 * (np.exp(q * (V + Rs * I) / (n2 * k * T)) - 1) \
                   - Isd3 * (np.exp(q * (V + Rs * I) / (n3 * k * T)) - 1) \
                   - (V + Rs * I) / Rsh - I
    # Adjust initial guess based on expected current values
    I_initial_guess = Iph / 2  # Adjusted initial guess
    I_solution, infodict, ier, mesg = fsolve(current_equation, I_initial_guess, args=(V, Rs, Rsh, Iph, Isd1, Isd2, Isd3, n1, n2, n3, T, q, k), full_output=True)
    if ier != 1:
        print(f"Solution not found for V={V}: {mesg}")
    return I_solution

# Synthetic range of voltage values
V_range = np.linspace(0, 45, 100)  # Adjusted to a more typical range for PV cells


params_example = (0.02, 100, 6, 1e-5, 1e-5, 1e-5, 1.3, 1.3, 1.3)  

# Calculate I for each V in V_range using the triple diode model
I_calculated = np.array([triple_diode_model(v, *params_example, T=330) for v in V_range])

# Plotting the I-V curve
plt.figure(figsize=(10, 6))
plt.plot(V_range, I_calculated, '-b', label='Calculated I-V Curve')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (I)')
plt.title('I-V Curve of the PV Cell')
plt.legend()
plt.grid(True)
plt.show()
