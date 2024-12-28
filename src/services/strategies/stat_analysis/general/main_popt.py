import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
import tqdm


# Fetch stock data for multiple assets (portfolio)
#symbols = ['TSLA', 'AAPL', 'GOOG', 'MSFT', 'AMZN']  # Example portfolio of 5 stocks
symbols = ['TSLA', 'AAPL']
start_date = "2019-01-01"
end_date = "2019-02-01"
stock_data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns for each asset in the portfolio
stock_returns = stock_data.pct_change().dropna()

# Number of assets in the portfolio
num_assets = len(symbols)

# Portfolio optimization parameters
num_w = 3  # Number of weights (adjustable)
num_theta = 3  # Number of theta coefficients per term (adjustable)
lambda_reg = 0.1  # Regularization weight (adjustable)
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
def simulate_state_trajectory(w, theta, returns_data):
    state_trajectory = [returns_data.iloc[0, :]]  # Initialize with the first return value for each asset
    for k in range(1, len(returns_data)):
        x_dot_k_minus_1 = returns_data.iloc[k-1, :] if k > 1 else np.zeros(num_assets)
        x_dot_k = returns_data.iloc[k, :]
        g_value = g(x_dot_k_minus_1, x_dot_k, w, theta)
        x_next = f(state_trajectory[-1], g_value)
        state_trajectory.append(x_next)
    return np.array(state_trajectory)

# Objective function for portfolio optimization
def objective_function(params, returns_data, lambda_reg, rho):
    w = params[:num_w]
    theta = params[num_w:num_w + num_w * num_theta].reshape((num_w, num_theta))
    Q_val, R_val, P_val = params[-3:]
    Q = np.array([[Q_val]])
    R = np.array([[R_val]])
    P = np.array([[P_val]])
    
    state_trajectory = simulate_state_trajectory(w, theta, returns_data)
    N = len(state_trajectory)
    total_loss = 0.0

    for k in range(1, N - 1):
        x_k = state_trajectory[k]
        x_ref = returns_data.iloc[k, :]
        x_dot_k_minus_1 = returns_data.iloc[k - 1, :] if k > 1 else np.zeros(num_assets)
        x_dot_k = returns_data.iloc[k, :]
        g_current = g(x_dot_k_minus_1, x_dot_k, w, theta)
        g_prev = g(returns_data.iloc[k - 2, :], returns_data.iloc[k - 1, :], w, theta) if k > 2 else g_current

        tracking_error = np.sum((x_k - x_ref)**2)
        control_effort = np.sum(g_current**2)
        stage_cost = tracking_error * Q[0, 0] + control_effort * R[0, 0]
        smoothness_penalty = np.sum((g_current - g_prev)**2)
        total_loss += stage_cost + rho * smoothness_penalty

    x_terminal = state_trajectory[-1]
    x_goal = returns_data.iloc[-1, :]
    terminal_cost = np.sum((x_terminal - x_goal)**2) * P[0, 0]
    total_loss += terminal_cost + lambda_reg * np.sum(theta**2)
    
    return total_loss

# Initial parameters for optimization
w_initial = np.array([0.5, 0.5, 0.5])  # Initial weights
theta_initial = np.array([[1, 0.5, 0.3], [0.7, 0.2, 0.5], [0.4, 0.8, 0.6]]).flatten()  # Flatten for optimization
Q_initial = 1.0
R_initial = 0.5
P_initial = 0.5
params_initial = np.concatenate((w_initial, theta_initial, [Q_initial, R_initial, P_initial]))

# Bounds for optimization to ensure positive Q, R, P
bounds = [(None, None)] * (num_w + num_w * num_theta) + [(0.1, None)] * 3

# Split stock data into train and test sets
train_size = 0.8
split_index = int(len(stock_returns) * train_size)
train_data = stock_returns[:split_index]
test_data = stock_returns[split_index-1:]

# Optimize on the training data
result_train = minimize(
    objective_function,
    params_initial,
    args=(train_data, lambda_reg, rho),
    method='L-BFGS-B',
    bounds=bounds,
    options={'disp': True}
)

# Retrieve optimized weights, theta parameters, and matrices
optimal_params_train = result_train.x
optimal_w_train = optimal_params_train[:num_w]
optimal_theta_train = optimal_params_train[num_w:num_w + num_w * num_theta].reshape((num_w, num_theta))
optimal_Q_train = optimal_params_train[-3]
optimal_R_train = optimal_params_train[-2]
optimal_P_train = optimal_params_train[-1]

# Simulate state trajectories for train and test data
train_trajectory = simulate_state_trajectory(optimal_w_train, optimal_theta_train, train_data)
test_trajectory = simulate_state_trajectory(optimal_w_train, optimal_theta_train, test_data)

# Convert trajectories to Pandas Series for plotting
train_trajectory = pd.DataFrame(train_trajectory, index=train_data.index)
test_trajectory = pd.DataFrame(test_trajectory, index=test_data.index)

# Calculate R-squared for train and test data
def calculate_r_squared(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)  # Residual sum of squares
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)

r_squared_train = calculate_r_squared(train_data.values, train_trajectory.values)
r_squared_test = calculate_r_squared(test_data.values, test_trajectory.values)

# Plotting the results
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

# Plot train data and model trajectory
axs[0].plot(train_data, label="Train Data (Returns)", linestyle="--", color="blue")
axs[0].plot(train_trajectory, label="Train Model Trajectory", color="red")
#axs[0].scatter(train_data, train_trajectory, label="Scatter (Train)", alpha=0.5)
axs[0].set_title(f"Train Data and Model Trajectory\nR^2 = {r_squared_train:.4f}")
axs[0].set_ylabel("Returns")
axs[0].legend()
axs[0].grid(True)

# Plot test data and model trajectory
axs[1].plot(test_data, label="Test Data (Returns)", linestyle="--", color="green")
axs[1].plot(test_trajectory, label="Test Model Trajectory", color="orange")
#axs[1].scatter(test_data, test_trajectory, label="Scatter (Test)", alpha=0.5)
axs[1].set_title(f"Test Data and Model Trajectory\nR^2 = {r_squared_test:.4f}")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Returns")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
