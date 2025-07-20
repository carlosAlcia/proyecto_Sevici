import torch 

class ModelNN(torch.nn.Module):
    def __init__(self, input_size:int, hidden_size:list[int], output_size:int):
        super(ModelNN, self).__init__()
        # Define the layers of the neural network
        layers = []
        # Input layer
        layers.append(torch.nn.Linear(input_size, hidden_size[0]))
        layers.append(torch.nn.ReLU())
        # Hidden layers
        for i in range(len(hidden_size) - 1):
            layers.append(torch.nn.Linear(hidden_size[i], hidden_size[i+1]))
            layers.append(torch.nn.ReLU())
            # Dropout layer for regularization
            layers.append(torch.nn.Dropout(0.2))  
        # Output layer
        layers.append(torch.nn.Linear(hidden_size[-1], output_size))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network."""
        return self.layers(x)
    
    def predict(self, x:torch.Tensor) -> torch.Tensor:
        """Make predictions using the neural network."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        self.train()
        return self.forward(x)

    def fit(self, x:torch.Tensor, y:torch.Tensor, x_val:torch.Tensor=None, y_val:torch.Tensor=None, epochs:int=100, lr:float=0.001, early_stopping:bool=True, patience:int=10) -> tuple[list, list]:
        """Train the neural network."""
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_val_loss = float('inf')
        current_patience = 0
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
            train_losses.append(loss.item())
            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
            if x_val is not None and y_val is not None:
                val_loss = criterion(self.forward(x_val), y_val)
                val_losses.append(val_loss.item())
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    current_patience = 0
                else:
                    current_patience += 1
                    if early_stopping and current_patience >= patience:
                        print('Early stopping triggered.')
                        break
        print('Training complete.')
        self.eval()
        return self, train_losses, val_losses
    
    def score(self, x:torch.Tensor, y:torch.Tensor) -> float:
        """Evaluate the model on the test data."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            loss = torch.nn.functional.mse_loss(output, y)
        return loss.item()