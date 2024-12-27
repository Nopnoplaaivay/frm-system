import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# PARAMETERS
# ---------------------------------------------
N = 50  # Number of agents
T = 100  # Number of time steps
kappa = 1  # Interaction strength
p = 2  # Power in interaction potential
V_H = 105  # High value of the asset
V_L = 95  # Low value of the asset

# Initial state of agents (W: wealth, Q: position, I: private info)
agents = np.random.rand(N, 3)  # Random initial states for all agents

# ---------------------------------------------
# FUNCTIONS
# ---------------------------------------------
def pairwise_potential(X_i, X_j):
    """Computes the potential between two agents."""
    distance = np.linalg.norm(X_i - X_j)
    return kappa / (distance ** p + 1e-6)  # Small epsilon to avoid division by zero

def total_potential(agents):
    """Computes total potential energy of the system."""
    total_energy = 0
    for i in range(N):
        for j in range(i+1, N):
            total_energy += pairwise_potential(agents[i], agents[j])
    return total_energy

def compute_gradient(U_func, X_i, X_j, epsilon=1e-5):
    """Compute the gradient of the potential U with respect to X_i using finite differences."""
    gradient = np.zeros_like(X_i)
    for k in range(len(X_i)):  # Loop over dimensions (W, Q, I)
        X_i_perturbed = np.copy(X_i)
        X_i_perturbed[k] += epsilon  # Add small change to one component of X_i
        gradient[k] = (U_func(X_i_perturbed, X_j) - U_func(X_i, X_j)) / epsilon
    return gradient

def update_agents(agents, dt=0.01, noise_level=0.1):
    """Update agent positions using gradient descent on the potential surface."""
    new_agents = np.copy(agents)
    for i in range(N):
        gradient = np.zeros_like(agents[i])
        for j in range(N):
            if i != j:
                gradient += compute_gradient(pairwise_potential, agents[i], agents[j])
        noise = noise_level * np.random.randn(3)  # Add stochastic noise
        new_agents[i] += -gradient * dt + noise  # Update agent's state
    return new_agents

# ---------------------------------------------
# SIMULATION
# ---------------------------------------------
potential_history = []
spread_history = []

# Initial True Value of the Asset (can be constant or slowly varying)
true_value = 100  

for t in range(T):
    agents = update_agents(agents)
    total_energy = total_potential(agents)
    potential_history.append(total_energy)
    
    # Calculate spread as an emergent property
    lambda_eff = np.mean(agents[:, 2])  # Use information level (I) to compute lambda_eff
    S_t = 2 * lambda_eff * (V_H - V_L)  # V_H - V_L = 10
    spread_history.append(S_t)

# ---------------------------------------------
# PLOT 1: Potential Energy and Spread Over Time
# ---------------------------------------------
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(potential_history, label='Market Potential $\mathcal{U}(t)$', color='tab:blue')
plt.xlabel('Time')
plt.ylabel('Potential Energy')
plt.title('Market Potential Over Time')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(spread_history, label='Spread $S_t$', color='tab:orange')
plt.xlabel('Time')
plt.ylabel('Spread')
plt.title('Emergent Bid-Ask Spread Over Time')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()

# ---------------------------------------------
# PLOT 2: Bid, Ask, and True Value (for fixed λ)
# ---------------------------------------------
lambda_example = 0.5  # Fixed λ for visualization
spread_example = 2 * lambda_example * (V_H - V_L)
bid_prices = [true_value - spread_example / 2 for _ in range(T)]
ask_prices = [true_value + spread_example / 2 for _ in range(T)]
true_values = [true_value for _ in range(T)]  # Assume constant true value

# Plot Bid, Ask, and True Value
plt.figure(figsize=(12, 6))
plt.plot(bid_prices, label='Bid $b_t$', linestyle='--', color='tab:red')
plt.plot(ask_prices, label='Ask $a_t$', linestyle='--', color='tab:green')
plt.plot(true_values, label='True Value $\mu$', linestyle='-', color='tab:blue')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'Bid, Ask, and True Value ($\lambda = {lambda_example}$)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# ---------------------------------------------
# PLOT 3: Heatmap of Spread as a function of λ
# ---------------------------------------------
lambda_values = np.linspace(0, 1, 100)  # Range of lambda from 0 to 1
time_values = np.arange(0, T)  # Time steps from 0 to T
spread_heatmap = np.zeros((len(lambda_values), T))

for i, lam in enumerate(lambda_values):
    for t in range(T):
        spread_heatmap[i, t] = 2 * lam * (V_H - V_L)  # Spread S_t = 2λ(V_H - V_L)

# Plot the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(spread_heatmap, aspect='auto', origin='lower', 
           extent=[0, T, 0, 1], cmap='viridis')
plt.colorbar(label='Spread $S_t$')
plt.xlabel('Time')
plt.ylabel('Informed Trader Probability $\lambda$')
plt.title('Heatmap of Spread $S_t$ as a Function of $\lambda$ and Time')
plt.show()
