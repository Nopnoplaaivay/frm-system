import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

# Fetch stock data (e.g., S&P 500 index) for 2019-2022 period
stock_symbol = '^GSPC'  # S&P 500 Index
stock_data = yf.download(stock_symbol, start="2019-01-01", end="2019-12-01")['Adj Close']

# Number of weights and theta parameters per term
num_w = 3  # Number of weights (adjustable)
num_theta = 3  # Number of theta coefficients per term (adjustable)
lambda_reg = 0.1  # Regularization weight (adjustable)
Q = np.eye(1) * 0.1  # Stage cost penalty for state deviation
R = np.eye(1) * 0.01  # Stage cost penalty for control input
P = np.eye(1) * 0.5  # Terminal cost penalty for state deviation

# Nonlinear function g based on derivatives of the state
def g(x_dot_k_minus_1, x_dot_k, w, theta):
    h = 0
    for i in range(len(w)):
        h += w[i] * (theta[i][0] * x_dot_k_minus_1**2 +
                     theta[i][1] * x_dot_k**2 +
                     theta[i][2] * x_dot_k_minus_1 * x_dot_k)
    return h

# Nonlinear state-space function f
def f(x_k, g_value):
    return x_k + g_value  # Adjust as needed for your model

# Function to simulate the state trajectory for given weights and parameters
def simulate_state_trajectory(w, theta, stock_data):
    state_trajectory = [stock_data.iloc[0]]  # Initialize with the first stock value
    for k in range(1, len(stock_data)):
        x_dot_k_minus_1 = stock_data.iloc[k-1] - stock_data.iloc[k-2] if k > 1 else 0
        x_dot_k = stock_data.iloc[k] - stock_data.iloc[k-1]
        g_value = g(x_dot_k_minus_1, x_dot_k, w, theta)
        x_next = f(state_trajectory[-1], g_value)
        state_trajectory.append(x_next)
    return np.array(state_trajectory)

# Objective function with terminal cost and regularization
def objective_function(params, stock_data, lambda_reg, Q, R, P):
    w = params[:num_w]
    theta = params[num_w:].reshape((num_w, num_theta))
    state_trajectory = simulate_state_trajectory(w, theta, stock_data)
    N = len(state_trajectory)
    total_loss = 0.0

    for k in range(N - 1):
        x_dot_k_minus_1 = stock_data.iloc[k-1] - stock_data.iloc[k-2] if k > 1 else 0
        x_dot_k = stock_data.iloc[k] - stock_data.iloc[k-1]
        g_current = g(x_dot_k_minus_1, x_dot_k, w, theta)
        g_prev = g(stock_data.iloc[k-2] - stock_data.iloc[k-3], stock_data.iloc[k-1] - stock_data.iloc[k-2], w, theta) if k > 2 else g_current

        # Stage cost: Adjust for scalar x_diff and matrix Q
        x_diff = state_trajectory[k] - stock_data.iloc[k]
        stage_cost = (x_diff**2) * Q[0, 0] + (g_current**2) * R[0, 0]  # Control effort
        reg_term = lambda_reg * (g_current - g_prev)**2
        total_loss += stage_cost + reg_term

    # Terminal cost: Adjust for scalar x_terminal and matrix P
    x_terminal = state_trajectory[-1]
    x_goal = stock_data.iloc[-1]  # Example: set final stock value as the goal
    terminal_cost = (x_terminal - x_goal)**2 * P[0, 0]
    total_loss += terminal_cost
    return total_loss


# Initial guesses for w and theta parameters
w_initial = np.array([0.5, 0.5, 0.5])
theta_initial = np.array([[1, 0.5, 0.3], [0.7, 0.2, 0.5], [0.4, 0.8, 0.6]]).flatten()
params_initial = np.concatenate((w_initial, theta_initial))

# Optimize weights and theta
result = minimize(objective_function, params_initial, args=(stock_data, lambda_reg, Q, R, P), method='Nelder-Mead')

# Extract optimized parameters
optimal_params = result.x
optimal_w = optimal_params[:num_w]
optimal_theta = optimal_params[num_w:].reshape((num_w, num_theta))

# Simulate optimal trajectory
optimal_state_trajectory = simulate_state_trajectory(optimal_w, optimal_theta, stock_data)
optimal_state_trajectory = pd.Series(optimal_state_trajectory, index=stock_data.index)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(stock_data, label="Original Stock Index", linestyle="--", color="blue")
plt.plot(optimal_state_trajectory, label="Optimized Modeled Dynamics", color="red")
plt.title(f"Stock Index Dynamics Modeled by Nonlinear State-Space Model (Optimized) - {stock_symbol}")
plt.xlabel("Time")
plt.ylabel("Index Value")
plt.legend()
plt.grid(True)
plt.show()
