import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ModelNN(torch.nn.Module):
    """A simple feedforward neural network model for regression tasks."""

    def __init__(self, input_size:int, hidden_size:list[int], output_size:int, dropout_rate:float=0.2, use_batch_norm:bool=False):
        """Initialize the neural network model.
        Args:
            input_size (int): Number of input features.
            hidden_size (list[int]): List of integers representing the number of neurons in each hidden layer
            output_size (int): Number of output features.
            dropout_rate (float): Dropout rate for regularization.
            use_batch_norm (bool): Whether to use batch normalization after each hidden layer.
        """
        super(ModelNN, self).__init__()
        # Define the layers of the neural network
        layers = []
        # Input layer
        layers.append(torch.nn.Linear(input_size, hidden_size[0]))
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(hidden_size[0]))
        layers.append(torch.nn.ReLU())
        if dropout_rate > 0:
            layers.append(torch.nn.Dropout(dropout_rate))
        # Hidden layers
        for i in range(len(hidden_size) - 1):
            layers.append(torch.nn.Linear(hidden_size[i], hidden_size[i+1]))
            layers.append(torch.nn.ReLU())
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(hidden_size[i+1]))
            # Dropout layer for regularization
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout(dropout_rate))  
        # Output layer
        layers.append(torch.nn.Linear(hidden_size[-1], output_size))
        self.layers = torch.nn.Sequential(*layers)

    def _to_tensor(self, x, dtype=torch.float32, reshape=False):
        if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
            x = torch.tensor(np.array(x), dtype=dtype)
        if reshape:
            x = x.view(-1, 1)
        return x

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
                torch.Tensor: Output tensor.
        """
        return self.layers(x)
    
    def predict(self, x:pd.DataFrame) -> torch.Tensor:
        """Make predictions using the neural network.
        Args:
            x (pd.DataFrame): Input features as a DataFrame.
        Returns:
            torch.Tensor: Output tensor.
        """
        # Convert DataFrame to PyTorch tensor
        x = self._to_tensor(x)
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def fit(self, x:pd.DataFrame, y:pd.DataFrame, eval_set:tuple[pd.DataFrame, pd.DataFrame]=None, epochs:int=100, lr:float=0.001, early_stopping:bool=True, patience:int=10, plot_graphs:bool=False, verbose_freq:int=100) -> tuple[list, list]:
        """Train the neural network."""

        # Convert DataFrame to PyTorch tensors
        x = self._to_tensor(x)
        y = self._to_tensor(y, reshape=True)
        if eval_set is not None:
            x_val, y_val = eval_set
            x_val = self._to_tensor(x_val)
            y_val = self._to_tensor(y_val, reshape=True)

        # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # Initialize variables for early stopping
        best_val_loss = float('inf')
        best_model_state = None
        current_patience = 0
        # Lists to store training and validation losses
        train_losses = []
        val_losses = []

        # Initial validation loss
        if x_val is not None and y_val is not None:
            self.eval()
            with torch.no_grad():
                val_loss = criterion(self.forward(x_val), y_val)
                print(f'Initial Validation Loss: {val_loss.item():.4f}')
            self.train()

        # Training loop
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            output = self.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            if epoch % verbose_freq == 0:
                train_losses.append(loss.item())
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
            if x_val is not None and y_val is not None:
                val_loss = criterion(self.forward(x_val), y_val)
                if epoch % verbose_freq == 0:
                    val_losses.append(val_loss.item())
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.state_dict()
                    current_patience = 0
                else:
                    current_patience += 1
                    if early_stopping and current_patience >= patience:
                        print('Early stopping triggered.')
                        break
        print('Training complete.')
        # Load the best model state if early stopping was triggered
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
        self.eval()
        if plot_graphs:
            self._plot_graphs(train_losses, val_losses, verbose_freq=verbose_freq)
        return self, train_losses, val_losses
    
    def save_model(self, path:str):
        """Save the model to a file."""
        torch.save(self.state_dict(), path)
        print(f'Model saved to {path}')
    
    def _plot_graphs(self, train_losses:list, val_losses:list, verbose_freq:int=100):
        # Plot training and validation losses
        plt.figure(figsize=(10, 6))
        epochs = np.arange(0, len(train_losses) * verbose_freq, verbose_freq)
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def score(self, x:pd.DataFrame, y:pd.DataFrame) -> float:
        """Evaluate the model on the test data."""
        x = self._to_tensor(x)
        y = self._to_tensor(y, reshape=True)
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            loss = torch.nn.functional.mse_loss(output, y)
        return loss.item()