# NJL-HRG-Chiral-Model
Full code
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Constants
m_u = 0.005   # GeV
m_d = 0.005   # GeV
m_s = 0.1407  # GeV
la = 0.6023   # GeV
G = 1.835 / (la**2)  # GeV^-2 
K = 12.36 / (la**5)  # GeV^-5
nc = 3

# Integrands
def integrand_phi_u(p, M_u, muB, muQ, T):
    E = np.sqrt(p**2 + M_u**2)
    f = 1 / (np.exp((E - ((muB / 3)+(2*muQ / 3))) / T) + 1)
    fbar = 1 / (np.exp((E + ((muB / 3)+(2*muQ / 3))) / T) + 1)
    return -((2 * nc * 4 * np.pi * p**2 * M_u) / ((2 * np.pi)**3 * E)) * (1 - f - fbar)

def integrand_phi_d(p, M_d, muB, muQ, T):
    E = np.sqrt(p**2 + M_d**2)
    f = 1 / (np.exp((E - ((muB / 3)-(muQ / 3))) / T) + 1)
    fbar = 1 / (np.exp((E + ((muB / 3)-(muQ / 3))) / T) + 1)
    return -((2 * nc * 4 * np.pi * p**2 * M_d) / ((2 * np.pi)**3 * E)) * (1 - f - fbar)

def integrand_phi_s(p, M_s, muB, muQ, muS, T):
    E = np.sqrt(p**2 + M_s**2)
    f = 1 / (np.exp((E - ((muB / 3)-(muQ / 3) - muS)) / T) + 1)
    fbar = 1 / (np.exp((E + ((muB / 3)-(muQ / 3) - muS)) / T) + 1)
    return -((2 * nc * 4 * np.pi * p**2 * M_s) / ((2 * np.pi)**3 * E)) * (1 - f - fbar)

# Integrals
def integral_u(M_u, muB, muQ, T):
    result, _ = quad(integrand_phi_u, 0, la, args=(M_u, muB, muQ, T))
    return result

def integral_d(M_d, muB, muQ, T):
    result, _ = quad(integrand_phi_d, 0, la, args=(M_d, muB, muQ, T))
    return result

def integral_s(M_s, muB, muQ, muS, T):
    result, _ = quad(integrand_phi_s, 0, la, args=(M_s, muB, muQ, muS, T))
    return result

# Coupled equations
def coupled_eqs(vars, muB, muQ, muS, T):
    M_u, M_d, M_s = vars
    phi_u = integral_u(M_u, muB, muQ, T)
    phi_d = integral_d(M_d, muB, muQ, T)
    phi_s = integral_s(M_s, muB, muQ, muS, T)

    eq1 = M_u - (m_u - 4 * G * phi_u + 2 * K * phi_d * phi_s)
    eq2 = M_d - (m_d - 4 * G * phi_d + 2 * K * phi_u * phi_s)
    eq3 = M_s - (m_s - 4 * G * phi_s + 2 * K * phi_u * phi_d)
    return [eq1, eq2, eq3]

# Solver
def solve_masses(muB, T, muQ=0.0, muS=0.0):
    guess = [0.3, 0.3, 0.5]  # Initial guess for M_u, M_d, M_s
    sol = fsolve(coupled_eqs, guess, args=(muB, muQ, muS, T))
    return sol  # returns [M_u, M_d, M_s]


# Define temperature range and fixed muB
T_range = np.linspace(0.01, 0.4, 1000)
muB_fixed = 0  # GeV

# Store solutions
M_u_vals = []
M_d_vals = []
M_s_vals = []

# Solve at each temperature
for T in T_range:
    M_u, M_d, M_s = solve_masses(muB_fixed, T)
    M_u_vals.append(M_u)
    M_d_vals.append(M_d)
    M_s_vals.append(M_s)

# Plotting

# Convert lists to arrays
T_vals = np.array(T_range)
M_u_vals = np.array(M_u_vals)
M_d_vals = np.array(M_d_vals)
M_s_vals = np.array(M_s_vals)

# Light quark constituent mass
M_q_vals = 0.5 * (M_u_vals + M_d_vals)

# Combine columns: T, M_q, M_s
data = np.column_stack((T_vals, M_q_vals, M_s_vals))

# Save as txt file
np.savetxt(
    "masses_vs_T.txt",
    data,
    header="T(GeV)    M_q(GeV)    M_s(GeV)",
    fmt="%.6f"
)

# Save as csv file
np.savetxt(
    "masses_vs_T.csv",
    data,
    delimiter=",",
    header="T_GeV,M_q_GeV,M_s_GeV",
    comments="",
    fmt="%.6f"
)
plt.figure(figsize=(8, 6))

plt.plot(T_range, M_u_vals, label="$M_u$", linewidth=2)
plt.plot(T_range, M_d_vals, label="$M_d$", linewidth=2)
plt.plot(T_range, M_s_vals, label="$M_s$", linewidth=2)

plt.xlabel("$T$ (GeV)", fontsize=18)
plt.ylabel("Constituent Mass $M$ (GeV)", fontsize=18)
plt.title(f"Three-Flavor NJL Model at $\\mu_B = {muB_fixed}$ GeV", fontsize=16)
plt.legend(fontsize=14)

# Beautify plot
plt.tick_params(axis='both', which='major', right=True, top=True, direction='in', labelsize=14, length=10, width=2)  
plt.tick_params(axis='both', which='minor', right=True, top=True, direction='in', labelsize=12, length=5, width=1.5)   
plt.minorticks_on()
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("three_flavor_masses_vs_T.pdf")
plt.show()
