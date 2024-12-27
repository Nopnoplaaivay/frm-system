import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

# Fetch stock data (e.g., S&P 500 index) for 2019-2022 period
stock_symbol = '^GSPC'  # S&P 500 Index
stock_data = yf.download(stock_symbol, start="2019-01-01", end="2019-12-31")['Adj Close']

# Resample to daily data if needed (e.g., just using 'Adj Close' for adjusted close prices)
# stock_data = stock_data.resample('D').last()  # Uncomment if resampling is required

# Number of weights and theta parameters per term
num_w = 3  # Number of weights (adjustable)
num_theta = 3  # Number of theta coefficients per term (adjustable)
lambda_reg = 0.1  # Regularization weight (adjustable)

# Nonlinear function g based on derivatives of the state
def g(x_dot_k_minus_1, x_dot_k, w, theta):
    # Define h as a sum of nonlinear transformations with weights and parameters
    h = 0
    for i in range(len(w)):
        # Construct polynomial basis functions as per the description
        h += w[i] * (theta[i][0] * x_dot_k_minus_1**2 + 
                     theta[i][1] * x_dot_k**2 + 
                     theta[i][2] * x_dot_k_minus_1 * x_dot_k)
    return h

# Nonlinear state-space function f
def f(x_k, g_value):
    # Define the state evolution; adjust as needed based on your model's specifics
    return x_k + g_value  # A basic state propagation with control input g

# Function to simulate the state trajectory for given weights and parameters
def simulate_state_trajectory(w, theta, stock_data):
    state_trajectory = [stock_data.iloc[0]]  # Initialize with the first stock value
    
    for k in range(1, len(stock_data)):
        # Calculate differences to feed into function g
        x_dot_k_minus_1 = stock_data.iloc[k-1] - stock_data.iloc[k-2] if k > 1 else 0
        x_dot_k = stock_data.iloc[k] - stock_data.iloc[k-1]
        
        # Calculate g and next state
        g_value = g(x_dot_k_minus_1, x_dot_k, w, theta)
        x_next = f(state_trajectory[-1], g_value)
        
        # Append the calculated state to the trajectory
        state_trajectory.append(x_next)
        
    return np.array(state_trajectory)

# Objective function with regularization term to minimize
def objective_function(params, stock_data, lambda_reg):
    # Extract weights and theta parameters from the flattened params array
    w = params[:num_w]
    theta = params[num_w:].reshape((num_w, num_theta))  # Reshape to match theta structure
    
    # Generate simulated trajectory
    state_trajectory = simulate_state_trajectory(w, theta, stock_data)
    
    # Mean squared error (original loss function)
    mse_loss = np.mean((state_trajectory - stock_data.values) ** 2)
    
    # Regularization term: squared L2 norm of theta parameters
    regularization_term = np.sum(theta**2)  # Equivalent to ||theta||^2
    
    # Objective function: original loss + regularization term
    total_loss = mse_loss + lambda_reg * regularization_term
    return total_loss




# Initial guesses for w and theta parameters
w_initial = np.array([0.5, 0.5, 0.5])  # Initial weights
theta_initial = np.array([[1, 0.5, 0.3], [0.7, 0.2, 0.5], [0.4, 0.8, 0.6]]).flatten()  # Flatten for optimization
params_initial = np.concatenate((w_initial, theta_initial))

# Optimize weights and theta using minimize from scipy.optimize
result = minimize(objective_function, params_initial, args=(stock_data, lambda_reg), method='Nelder-Mead')

# Retrieve optimized weights and theta parameters
optimal_params = result.x
optimal_w = optimal_params[:num_w]
optimal_theta = optimal_params[num_w:].reshape((num_w, num_theta))
print("Optimal weights (w):", optimal_w)
print("Optimal theta parameters:", optimal_theta)

# Simulate state trajectory using the optimized parameters
optimal_state_trajectory = simulate_state_trajectory(optimal_w, optimal_theta, stock_data)
optimal_state_trajectory = pd.Series(optimal_state_trajectory, index=stock_data.index)

# Plot the stock data vs. optimized state trajectory
plt.figure(figsize=(12, 6))
plt.plot(stock_data, label="Original Stock Index", linestyle="--", color="blue")
plt.plot(optimal_state_trajectory, label="Optimized Modeled Dynamics", color="red")
plt.title(f"Stock Index Dynamics Modeled by Nonlinear State-Space Model (Optimized) - {stock_symbol}")
plt.xlabel("Time")
plt.ylabel("Index Value")
plt.legend()
plt.grid(True)
plt.show()
