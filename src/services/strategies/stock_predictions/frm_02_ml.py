import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torchviz import make_dot


# Load and preprocess the data
indices = ['IonQ', 'Rigetti Computing', 'Quantum Computing Inc.',
           'D-Wave Quantum', 'Alphabet', 'IBM', 
           'Microsoft', 'Nvidia', 'Defiance Quantum ETF', 'Global X Future Analytics Tech']

symbols = ['Date', 'IONQ', 'RGTI', 'QUBT', 'QBTS', 'GOOGL', 'IBM', 'MSFT', 'NVDA', 'QTUM', 'AIQ']

df = pd.read_csv('quantum_technology_indices_prices.csv')
df = df.dropna(axis=0)
df.columns = symbols
df = df.reset_index(drop=True)

symbols = ['IONQ', 'RGTI', 'QUBT', 'QBTS', 'GOOGL', 'IBM', 'MSFT', 'NVDA', 'QTUM', 'AIQ']
for symbol in symbols:
# Select one stock (e.g., MSFT) and normalize the data
    #symbol = symbol
    # data = df[symbol].values.reshape(-1, 1)
    # scaler = MinMaxScaler()
    # data_normalized = scaler.fit_transform(data)

    # # Create time series dataset
    # sequence_length = 63  # Use 60 days of data to predict the next value
    # X, y = [], []

    # for i in range(len(data_normalized) - sequence_length):
    #     X.append(data_normalized[i:i+sequence_length])
    #     y.append(data_normalized[i+sequence_length])

    # X = np.array(X)
    # y = np.array(y) 

    # # Split into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Convert to PyTorch tensors
    # X_train = torch.tensor(X_train, dtype=torch.float32)
    # y_train = torch.tensor(y_train, dtype=torch.float32)
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # y_test = torch.tensor(y_test, dtype=torch.float32)

    # Define the RNN model
    class StockRNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(StockRNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
            out = self.fc(out[:, -1, :])  # Take the last output
            return out



    # Initialize the model
    # input_size = 1
    # hidden_size = 32
    # num_layers = 4
    # output_size = 1

    model = StockRNN(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    '''# Create a dummy input tensor (batch size = 1, sequence length = 60, input size = 1)
    dummy_input = torch.randn(1, 60, 1)

    # Get the computation graph using torchviz
    output = model(dummy_input)
    dot = make_dot(output, params=dict(model.named_parameters()))

    # Visualize the model graph
    dot.render("model_architecture", format="png")
    '''
    # Train the model
    epochs = 100
    batch_size = 32
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}')




    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
        y_test_np = y_test.numpy()
        predictions = scaler.inverse_transform(predictions)
        y_test_np = scaler.inverse_transform(y_test_np)

    # Plot the results
    fix, axes = plt.subplots(1,2,figsize=(8, 4))

    # Subplot 1: Actual vs Predicted Prices on Test Set
    axes[0].plot(y_test_np, label='Actual Prices', color = 'blue')
    axes[0].plot(predictions, label='Predicted Prices', marker = 'o', alpha = 0.5, color = 'red')
    axes[0].legend()
    axes[0].set_title(f'{symbol} Price Prediction (Test Set)', weight = 'bold')
    axes[0].set_xlabel('Time [days]', fontsize = 12, weight = 'bold')
    axes[0].set_ylabel('Value [USD]', fontsize = 12, weight = 'bold')
    axes[0].grid(True)
    # Predict the next 63 days (3 months)
    last_sequence = X_test[-2].reshape(1, sequence_length, 1)
    future_predictions = []

    # Generate future predictions (next 3 months ~ 63 days)
    for _ in range(63):
        with torch.no_grad():
            future_pred = model(torch.tensor(last_sequence, dtype=torch.float32)).numpy()
            future_predictions.append(future_pred[0, 0])
            # Update the sequence with the new prediction
            last_sequence = np.append(last_sequence[:, 1:, :], future_pred.reshape(1, 1, 1), axis=1)

    # Plot the future predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    axes[1].plot(future_predictions, label='Predicted Future Prices', color = 'red', alpha = 0.8)
    axes[1].legend()
    axes[1].set_title(f'{symbol} 3-Month Price Forecast', weight = 'bold')
    axes[1].set_xlabel('Time [days]', fontsize = 12, weight = 'bold')
    axes[1].set_ylabel('Value [USD]', fontsize = 12, weight = 'bold')
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(f'{symbol}.jpg', dpi = 600)
    #plt.show()
