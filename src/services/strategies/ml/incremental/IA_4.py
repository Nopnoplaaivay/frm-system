import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
N = 100  # Number of traders
J = 1.0  # Interaction strength between traders
lambda_val = 0.5  # Probability that a trader is informed
h = 1.0  # External field (representing market news or public info)
steps = 1000  # Number of simulation steps

# Initialize the system (traders' decisions as spins)
# 1 means the trader is an informed buyer (+1), -1 means uninformed seller (-1)
spins = np.random.choice([-1, 1], size=N)

# Function to compute Hamiltonian for Transverse Ising model
def hamiltonian(spins, J, h, lambda_val):
    interaction_term = -J * np.sum(spins[:-1] * spins[1:])  # Interaction between neighboring traders
    external_field_term = -lambda_val * h * np.sum(np.sign(spins))  # External field based on information asymmetry
    return interaction_term + external_field_term

# Function to update the system using Metropolis algorithm
def metropolis_step(spins, J, h, lambda_val):
    i = np.random.randint(N)  # Pick a random trader
    delta_H = -2 * spins[i] * (J * (spins[i-1] + spins[(i+1)%N])) - lambda_val * h  # Change in Hamiltonian
    if delta_H < 0 or np.random.rand() < np.exp(-delta_H):
        spins[i] = -spins[i]  # Flip the spin (buy/sell decision)
    return spins

# Simulate the system over time
spread_history = []
for step in range(steps):
    spins = metropolis_step(spins, J, h, lambda_val)
    
    # Calculate the bid-ask spread S_t based on spin configuration
    # Here, +1 means a trader wants to buy (ask), -1 means a trader wants to sell (bid)
    bid_price = np.sum(spins == -1) / N  # Fraction of sellers (uninformed)
    ask_price = np.sum(spins == 1) / N  # Fraction of buyers (informed)
    
    # Spread at time t (S_t = ask_price - bid_price)
    spread_history.append(ask_price - bid_price)

# --- Visualization ---

# Plot 1: Bid-Ask Spread over Time
plt.figure(figsize=(12, 6))
plt.plot(spread_history, label=f'Bid-Ask Spread (λ={lambda_val})', color='blue')
plt.title(f'Bid-Ask Spread Over Time (λ={lambda_val})')
plt.xlabel('Time Steps')
plt.ylabel('Bid-Ask Spread')
plt.legend()
plt.grid(True)
plt.show()

# --- Heatmap: Spread as a function of lambda ---
lambda_values = np.linspace(0, 1, 20)
spread_matrix = np.zeros((len(lambda_values), steps))

for idx, lambda_val in enumerate(lambda_values):
    spins = np.random.choice([-1, 1], size=N)  # Reset the spins
    for step in range(steps):
        spins = metropolis_step(spins, J, h, lambda_val)
        bid_price = np.sum(spins == -1) / N  # Fraction of sellers
        ask_price = np.sum(spins == 1) / N  # Fraction of buyers
        spread_matrix[idx, step] = ask_price - bid_price

# Plot 2: Heatmap of Spread as a function of λ
plt.figure(figsize=(12, 6))
plt.imshow(spread_matrix, aspect='auto', cmap='hot', extent=[0, steps, 0, 1], origin='lower')
plt.colorbar(label='Spread')
plt.title('Heatmap of Spread as a Function of λ')
plt.xlabel('Time Steps')
plt.ylabel('λ (Information Asymmetry)')
plt.show()
