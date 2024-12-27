import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

# Fetch stock data (e.g., S&P 500 index) for the 2019-2022 period
stock_symbol = '^GSPC'  # S&P 500 Index
stock_data = yf.download(stock_symbol, start="2019-01-01", end="2019-02-01")['Adj Close']

# Convert stock data into differences between consecutive timesteps
stock_data_diff = stock_data.diff().dropna()  # Drop NaN created by differencing

# Number of weights and theta parameters per term
num_w = 3  # Number of weights (adjustable)
num_theta = 3  # Number of theta coefficients per term (adjustable)
lambda_reg = 0.1  # Regularization weight (adjustable)
rho = 0.2  # Smoothness penalty weight

# Define Q, R, P matrices
Q = np.array([[1.0]])  # State tracking weight
R = np.array([[0.5]])  # Control effort weight
P = np.array([[0.5]])  # Terminal state weight

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
        x_dot_k_minus_1 = stock_data.iloc[k-1] if k > 1 else 0
        x_dot_k = stock_data.iloc[k]
        g_value = g(x_dot_k_minus_1, x_dot_k, w, theta)
        x_next = f(state_trajectory[-1], g_value)
        state_trajectory.append(x_next)
    return np.array(state_trajectory)

# Objective function with terminal cost and regularization
def objective_function(params, stock_data, lambda_reg, Q, R, P, rho):
    w = params[:num_w]
    theta = params[num_w:].reshape((num_w, num_theta))
    state_trajectory = simulate_state_trajectory(w, theta, stock_data)
    N = len(state_trajectory)
    total_loss = 0.0

    for k in range(1, N - 1):
        x_k = state_trajectory[k]
        x_ref = stock_data.iloc[k]
        x_dot_k_minus_1 = stock_data.iloc[k - 1] if k > 1 else 0
        x_dot_k = stock_data.iloc[k]
        g_current = g(x_dot_k_minus_1, x_dot_k, w, theta)
        g_prev = g(stock_data.iloc[k - 2], stock_data.iloc[k - 1], w, theta) if k > 2 else g_current

        tracking_error = (x_k - x_ref)**2
        control_effort = g_current**2
        stage_cost = tracking_error * Q[0, 0] + control_effort * R[0, 0]
        smoothness_penalty = (g_current - g_prev)**2
        total_loss += stage_cost + rho * smoothness_penalty

    x_terminal = state_trajectory[-1]
    x_goal = stock_data.iloc[-1]
    terminal_cost = (x_terminal - x_goal)**2 * P[0, 0]
    total_loss += terminal_cost + lambda_reg * np.sum(theta**2)
    return total_loss

# Initial parameters for optimization
w_initial = np.array([0.5, 0.5, 0.5])  # Initial weights
theta_initial = np.array([[1, 0.5, 0.3], [0.7, 0.2, 0.5], [0.4, 0.8, 0.6]]).flatten()  # Flatten for optimization
params_initial = np.concatenate((w_initial, theta_initial))

# Split stock data differences into train and test sets
train_size = 0.5
split_index = int(len(stock_data_diff) * train_size)
train_data = stock_data_diff[:split_index]
test_data = stock_data_diff[split_index-1:]

# Optimize on the training data
result_train = minimize(
    objective_function,
    params_initial,
    args=(train_data, lambda_reg, Q, R, P, rho),
    method='Nelder-Mead'
)

# Retrieve optimized weights and theta parameters
optimal_params_train = result_train.x
optimal_w_train = optimal_params_train[:num_w]
optimal_theta_train = optimal_params_train[num_w:].reshape((num_w, num_theta))

# Simulate state trajectories for train and test data
train_trajectory = simulate_state_trajectory(optimal_w_train, optimal_theta_train, train_data)
test_trajectory = simulate_state_trajectory(optimal_w_train, optimal_theta_train, test_data)

# Convert trajectories to Pandas Series for plotting
train_trajectory = pd.Series(train_trajectory, index=train_data.index)
test_trajectory = pd.Series(test_trajectory, index=test_data.index)

# Plot original stock data differences and trajectories
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

# Plot train data and trajectory
axs[0].plot(train_data, label="Train Data (Differences)", linestyle="--", marker = 'o', color="blue")
axs[0].plot(train_trajectory, label="Train Model Trajectory", color="red", marker = 'x')
axs[0].set_title("Training Data and Model Trajectory (Differences)")
axs[0].set_ylabel("Difference Value")
axs[0].legend()
axs[0].grid(True)

# Plot test data and trajectory
axs[1].plot(test_data, label="Test Data (Differences)", linestyle="--", color="green",marker = 'o')
axs[1].plot(test_trajectory, label="Test Model Trajectory", color="orange", marker = 'x')
axs[1].set_title("Testing Data and Model Trajectory (Differences)")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Difference Value")
axs[1].legend()
axs[1].grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()
