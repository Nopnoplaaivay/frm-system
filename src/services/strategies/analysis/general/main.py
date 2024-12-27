import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from sklearn.model_selection import train_test_split

# Fetch stock data (e.g., S&P 500 index) for 2019-2022 period
stock_symbol = '^GSPC'  # S&P 500 Index
stock_data = yf.download(stock_symbol, start="2019-01-01", end="2019-12-31")['Adj Close']

# Number of weights and theta parameters per term
num_w = 3  # Number of weights (adjustable)
num_theta = 3  # Number of theta coefficients per term (adjustable)
lambda_reg = 0.1  # Regularization weight (adjustable)
Q = np.eye(1) * 0.1  # Stage cost penalty for state deviation
R = np.eye(1) * 0.01  # Stage cost penalty for control input
P = np.eye(1) * 0.5  # Terminal cost penalty for state deviation
rho = 0.2  # Smoothness penalty weight


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
# Define the modified objective function
def objective_function(params, stock_data, lambda_reg, Q, R, P, rho):
    w = params[:num_w]
    theta = params[num_w:].reshape((num_w, num_theta))
    state_trajectory = simulate_state_trajectory(w, theta, stock_data)
    N = len(state_trajectory)
    total_loss = 0.0

    # Loop through each step in the trajectory to accumulate costs
    for k in range(1, N - 1):
        x_k = state_trajectory[k]
        x_ref = stock_data.iloc[k]
        x_dot_k_minus_1 = stock_data.iloc[k - 1] - stock_data.iloc[k - 2] if k > 1 else 0
        x_dot_k = stock_data.iloc[k] - stock_data.iloc[k - 1]
        
        # Calculate current and previous g values
        g_current = g(x_dot_k_minus_1, x_dot_k, w, theta)
        g_prev = g(stock_data.iloc[k - 2] - stock_data.iloc[k - 3], stock_data.iloc[k - 1] - stock_data.iloc[k - 2], w, theta) if k > 2 else g_current
        
        # Stage cost: tracking error and control effort
        tracking_error = (x_k - x_ref)**2
        control_effort = g_current**2
        stage_cost = tracking_error * Q[0, 0] + control_effort * R[0, 0]
        
        # Smoothness penalty using ψ term
        smoothness_penalty = (g_current - g_prev)**2
        total_loss += stage_cost + rho * smoothness_penalty

    # Terminal cost at final state
    x_terminal = state_trajectory[-1]
    x_goal = stock_data.iloc[-1]  # Goal: last stock value or predefined target
    terminal_cost = (x_terminal - x_goal)**2 * P[0, 0]
    
    # Total cost: include terminal cost and regularization on θ
    total_loss += terminal_cost + lambda_reg * np.sum(theta**2)
    return total_loss

# Parameters for optimization (same setup as before)
w_initial = np.array([0.5, 0.5, 0.5])  # Initial weights
theta_initial = np.array([[1, 0.5, 0.3], [0.7, 0.2, 0.5], [0.4, 0.8, 0.6]]).flatten()  # Flatten for optimization
params_initial = np.concatenate((w_initial, theta_initial))

# Define Q, R, P matrices as 1x1 for simplicity (example values)
Q = np.array([[1.0]])  # State tracking weight
R = np.array([[0.1]])  # Control effort weight
P = np.array([[0.5]])  # Terminal state weight
rho = 0.2  # Smoothness penalty weight
# Split stock data into train and test sets (e.g., 80% train, 20% test)
train_size = 0.8
split_index = int(len(stock_data) * train_size)
train_data = stock_data[:split_index]
test_data = stock_data[split_index-1:]

# Optimize on the training data
result_train = minimize(
    objective_function,
    params_initial,
    args=(train_data, lambda_reg, Q, R, P, rho),
    method='Nelder-Mead'
)

# Retrieve optimized weights and theta parameters from training
optimal_params_train = result_train.x
optimal_w_train = optimal_params_train[:num_w]
optimal_theta_train = optimal_params_train[num_w:].reshape((num_w, num_theta))

# Simulate state trajectories for both train and test sets
train_trajectory = simulate_state_trajectory(optimal_w_train, optimal_theta_train, train_data)
test_trajectory = simulate_state_trajectory(optimal_w_train, optimal_theta_train, test_data)

# Convert trajectories to Pandas Series for plotting
train_trajectory = pd.Series(train_trajectory, index=train_data.index)
test_trajectory = pd.Series(test_trajectory, index=test_data.index)

# Plot original stock data and trajectories
# Create subplots for train and test data
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot train data and trajectory
axs[0].plot(train_data, label="Train Data (Original)", linestyle="--", color="blue")
axs[0].plot(train_trajectory, label="Train Model Trajectory", color="red")
axs[0].set_title("Training Data and Model Trajectory")
axs[0].set_ylabel("Index Value")
axs[0].legend()
axs[0].grid(True)

# Plot test data and trajectory
axs[1].plot(test_data, label="Test Data (Original)", linestyle="--", color="green")
axs[1].plot(test_trajectory, label="Test Model Trajectory", color="orange")
axs[1].set_title("Testing Data and Model Trajectory")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Index Value")
axs[1].legend()
axs[1].grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()
