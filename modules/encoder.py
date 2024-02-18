import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(encoder, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
            for i in range(len(hidden_sizes)-1)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # Input layer
        x = F.relu(self.input_layer(x))
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        # Output layer
        x = self.output_layer(x)
        
        return x