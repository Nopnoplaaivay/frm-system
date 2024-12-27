import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# PARAMETERS
# ---------------------------------------------
T = 100  # Time steps
lambda_values = np.linspace(0, 1, 100)  # Probability that trader is informed (0 to 1)
V_H, V_L = 105, 95  # High and low asset values

# Simulate the true value of the asset (can be modeled as a random walk)
np.random.seed(42)  # Set random seed for reproducibility
true_values = 100 + np.cumsum(np.random.randn(T))  # Random walk for true asset value

# Public and Private Information Sets
public_expectations = true_values + np.random.randn(T) * 2  # Public has noisy access
private_expectations = true_values + 0.5 * np.random.randn(T)  # Insiders have better info

# ---------------------------------------------
# FUNCTION TO COMPUTE INFOGAP
# ---------------------------------------------
def compute_infogap(public_exp, private_exp):
    """
    Computes the information asymmetry (InfoGap) as the difference between
    private (insider) and public expectations.
    """
    return private_exp - public_exp

# Compute InfoGap for all time steps
info_gap = compute_infogap(public_expectations, private_expectations)

# ---------------------------------------------
# PLOT 1: InfoGap Over Time
# ---------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(true_values, label='True Value $X_t$', linestyle='-', color='tab:blue')
plt.plot(public_expectations, label='Public Expectation $\mathbb{E}[X \mid \mathcal{F}]$', linestyle='--', color='tab:red')
plt.plot(private_expectations, label='Private Expectation $\mathbb{E}[X \mid \mathcal{G}]$', linestyle='--', color='tab:green')
plt.plot(info_gap, label='InfoGap $E[X \mid \mathcal{G}] - E[X \mid \mathcal{F}]$', linestyle='-', color='tab:orange')

plt.xlabel('Time')
plt.ylabel('Asset Value / Expectation')
plt.title('Information Asymmetry (InfoGap) Over Time')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# ---------------------------------------------
# PLOT 2: Jensen Inequality Visualization
# ---------------------------------------------
x = np.linspace(-2, 2, 100)
f = lambda x: x**2  # Example of a convex function
E_X = np.mean(x)
f_E_X = f(E_X)
E_f_X = np.mean(f(x))

plt.figure(figsize=(8, 6))
plt.plot(x, f(x), label='$f(x) = x^2$', color='tab:blue')
plt.scatter(E_X, f_E_X, color='red', label='$f(\mathbb{E}[X])$', s=100)
plt.scatter(E_X, E_f_X, color='green', label='$\mathbb{E}[f(X)]$', s=100)

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Jensen\'s Inequality Visualization')
plt.axvline(x=E_X, linestyle='--', color='gray', label='$\mathbb{E}[X]$')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# ---------------------------------------------
# PLOT 3: Heatmap of InfoGap as a function of λ
# ---------------------------------------------
spread_heatmap = np.zeros((len(lambda_values), T))

for i, lam in enumerate(lambda_values):
    for t in range(T):
        # InfoGap depends on lambda (more lambda, more private info)
        spread_heatmap[i, t] = lam * (private_expectations[t] - public_expectations[t])

plt.figure(figsize=(10, 8))
plt.imshow(spread_heatmap, aspect='auto', origin='lower', 
           extent=[0, T, 0, 1], cmap='viridis')
plt.colorbar(label='InfoGap')
plt.xlabel('Time')
plt.ylabel('Informed Trader Probability $\lambda$')
plt.title('Heatmap of InfoGap as a Function of $\lambda$ and Time')
plt.show()
