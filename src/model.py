import torch
from torch import nn
seed = 10
torch.manual_seed(seed)
import os


# Helper function to convert numpy arrays to tensors
def t(x):
    """Convert a numpy array to a PyTorch tensor with float type."""
    return torch.from_numpy(x).float()

# Actor module for a neural network that outputs categorical actions
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        """
        Initializes the Actor network.

        Parameters:
        - state_dim: Integer representing the dimensionality of the input state space.
        - n_actions: Integer representing the number of possible actions (output classes).

        The network consists of three fully connected layers:
        - The first layer maps the input state to 128 hidden units.
        - The second layer maps from 128 hidden units to 64 hidden units.
        - The third layer maps from 64 hidden units to 32 hidden units.
        - Finally, a layer maps to n_actions, and Softmax is applied to produce a probability distribution over actions.
        """
        super().__init__()  # Call the initializer of nn.Module
        # Define a sequential model for the Actor network
        self.model = nn.Sequential(
            # First linear layer maps input state to 64-dimensional hidden layer
            nn.Linear(state_dim, 64),
            # Apply Tanh activation function
            nn.Tanh(),
            # Second linear layer maps 64-dimensional hidden layer to 32-dimensional hidden layer
            nn.Linear(64, 32),
            # Apply Tanh activation function
            nn.Tanh(),
            # Third linear layer maps 32-dimensional hidden layer to the output layer of size n_actions
            nn.Linear(32, n_actions),
            # Apply Softmax activation function to get a probability distribution over actions
            nn.Softmax(dim=-1)  # Ensure the softmax is applied over the correct dimension
        )

    def forward(self, X):
        """
        Forward pass through the network.

        Parameters:
        - X: A tensor of input states, with shape (batch_size, state_dim).

        Returns:
        - A tensor representing the probability distribution over actions for each input state,
          with shape (batch_size, n_actions).
        """
        # Pass the input through the sequential model to get action probabilities
        return self.model(X)

    def save(self, file_name='actormodel.pth'):
        """
        Save the model's state dictionary to a file.

        Parameters:
        - file_name: String representing the name of the file to save the model weights.
        """
        model_folder_path = './model'
        # Create the model directory if it doesn't exist
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        # Save the model's state dictionary
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='actormodel.pth'):
        """
        Load the model's state dictionary from a file.

        Parameters:
        - file_name: String representing the name of the file to load the model weights from.
        """
        file_name = os.path.join('./model', file_name)
        # Load the model's state dictionary
        self.load_state_dict(torch.load(file_name))



# Critic module for a neural network that outputs a single value estimation (e.g., state value)
class Critic(nn.Module):
    def __init__(self, state_dim):
        """
        Initializes the Critic network.

        Parameters:
        - state_dim: Integer representing the dimensionality of the input state space.

        The network consists of three fully connected layers:
        - The first layer maps the input state to 128 hidden units.
        - The second layer maps from 128 hidden units to 64 hidden units.
        - The third layer maps from 64 hidden units to a single output, representing the value estimation.
        - ReLU activations are used in the hidden layers.
        """
        super().__init__()  # Call the initializer of nn.Module
        # Define a sequential model for the Critic network
        self.model = nn.Sequential(
            # First linear layer maps input state to 64-dimensional hidden layer
            nn.Linear(state_dim, 64),
            # Apply ReLU activation function to introduce non-linearity
            nn.ReLU(),
            # Second linear layer maps 64-dimensional hidden layer to 32-dimensional hidden layer
            nn.Linear(64, 32),
            # Apply ReLU activation function to introduce non-linearity
            nn.ReLU(),
            # Third linear layer maps 32-dimensional hidden layer to a single output value
            nn.Linear(32, 1)  # Output is a single scalar value representing the state value
        )

    def forward(self, X):
        """
        Forward pass through the network.

        Parameters:
        - X: A tensor of input states, with shape (batch_size, state_dim).

        Returns:
        - A tensor representing the estimated value for each input state,
          with shape (batch_size, 1).
        """
        # Pass the input through the sequential model to get the value estimation
        return self.model(X)

    def save(self, file_name='criticmodel.pth'):
        """
        Save the model's state dictionary to a file.

        Parameters:
        - file_name: String representing the name of the file to save the model weights.
        """
        model_folder_path = './model'
        # Create the model directory if it doesn't exist
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        # Save the model's state dictionary
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='criticmodel.pth'):
        """
        Load the model's state dictionary from a file.

        Parameters:
        - file_name: String representing the name of the file to load the model weights from.
        """
        file_name = os.path.join('./model', file_name)
        # Load the model's state dictionary
        self.load_state_dict(torch.load(file_name))


