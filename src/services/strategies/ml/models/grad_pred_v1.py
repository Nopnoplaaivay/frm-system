import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

# Fetch stock data (e.g., S&P 500 index) for the 2019-2022 period
stock_symbol = 'TSLA'  # S&P 500 Index
stock_data_diff = yf.download(stock_symbol, start="2019-01-01", end="2019-12-31")['Adj Close']

# Convert stock data into differences between consecutive timesteps
#stock_data_diff = stock_data.diff().dropna()  # Drop NaN created by differencing
# Normalize the differences
#mean_diff = stock_data_diff.mean()
#std_diff = stock_data_diff.std()
#stock_data_diff = (stock_data_diff - mean_diff) / std_diff



# Number of weights and theta parameters per term
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
def objective_function(params, stock_data, lambda_reg, rho):
    # Extract parameters
    w = params[:num_w]
    theta = params[num_w:num_w + num_w * num_theta].reshape((num_w, num_theta))
    Q_val, R_val, P_val = params[-3:]
    Q = np.array([[Q_val]])
    R = np.array([[R_val]])
    P = np.array([[P_val]])
    
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
Q_initial = 1.0
R_initial = 0.5
P_initial = 0.5
params_initial = np.concatenate((w_initial, theta_initial, [Q_initial, R_initial, P_initial]))

# Bounds for optimization to ensure positive Q, R, P
bounds = [(None, None)] * (num_w + num_w * num_theta) + [(0.1, None)] * 3

# Split stock data differences into train and test sets
train_size = 0.8
split_index = int(len(stock_data_diff) * train_size)
train_data = stock_data_diff[:split_index]
test_data = stock_data_diff[split_index-1:]

# Optimize on the training data
result_train = minimize(
    objective_function,
    params_initial,
    args=(train_data, lambda_reg, rho),
    method='L-BFGS-B',
    bounds=bounds
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
train_trajectory = pd.Series(train_trajectory, index=train_data.index)
test_trajectory = pd.Series(test_trajectory, index=test_data.index)

# Calculate Simple Moving Average (SMA) for comparison (using a 10-day window)
sma_window = 10  # Window size for SMA
sma_train = stock_data_diff.rolling(window=sma_window).mean().iloc[1:split_index]  # Skip NaN generated by rolling window
sma_test = stock_data_diff.rolling(window=sma_window).mean().iloc[split_index-1:]


# Calculate R-squared
def calculate_r_squared(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)  # Residual sum of squares
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared
# Calculate R-squared for train and test data
r_squared_train = calculate_r_squared(train_data, train_trajectory)
r_squared_test = calculate_r_squared(test_data, test_trajectory)

print(r_squared_train)
print(r_squared_test)

# Plotting original stock data differences, SMA, scatter plots, and model trajectories
fig, axs = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

# Plot train data, SMA, model trajectory, and scatter plot
axs[0].plot(train_data, label="Train Data (Differences)", linestyle="--", marker='o', color="blue")
axs[0].plot(train_trajectory, label="Train Model Trajectory", color="red", marker='x')
axs[0].plot(sma_train, label="Train SMA (10-day)", color="green", linestyle="-.")
#axs[0,1].scatter(train_data, train_trajectory, color='black', label="Scatter (Train)", alpha=0.5)
axs[0].set_title(f"Training Data, SMA, Model Trajectory (Differences)\nR^2 = {r_squared_train:.4f}")
axs[0].set_ylabel("Difference Value")
axs[0].legend()
axs[0].grid(True)

# Plot test data, SMA, model trajectory, and scatter plot
axs[1].plot(test_data, label="Test Data (Differences)", linestyle="--", color="green", marker='o')
axs[1].plot(test_trajectory, label="Test Model Trajectory", color="orange", marker='x')
axs[1].plot(sma_test, label="Test SMA (10-day)", color="purple", linestyle="-.")
#axs[1,1].scatter(test_data, test_trajectory, color='black', label="Scatter (Test)", alpha=0.5)
axs[1].set_title(f"Testing Data, SMA, Model Trajectory (Differences)\nR^2 = {r_squared_test:.4f}")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Difference Value")
axs[1].legend()
axs[1].grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()