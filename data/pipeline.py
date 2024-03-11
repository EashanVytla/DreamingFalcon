import torch
import pandas as pd

class Pipeline:
    def __init__(self, csv_path_states, csv_path_actions):
        self.csv_path_states = csv_path_states
        self.csv_path_actions = csv_path_actions
        self.data = None
        self.states_over_time = None
        self.actions_over_time = None

    def read_csv(self):
        # Read the CSV file using pandas
        self.data_states = pd.read_csv(self.csv_path_states, header=None)
        self.data_actions = pd.read_csv(self.csv_path_actions, header=None)

    def prepare_data(self, batch_size):
        assert self.data_states != None
        assert self.data_actions != None
        # Convert the pandas DataFrame to a PyTorch tensor
        self.states_over_time = torch.tensor(self.data_states, dtype=torch.float32)
        self.actions_over_time = torch.tensor(self.data_actions, dtype=torch.float32)

        # Add batch dimension
        self.states_over_time = self.states_over_time.unsqueeze(0)  # Assuming you want batch size of 1 initially
        self.actions_over_time = self.actions_over_time.unsqueeze(0)  # Assuming you want batch size of 1 initially

        # Split into batches
        self.states_over_time = self.states_over_time.chunk(batch_size, dim=1)
        self.actions_over_time = self.actions_over_time.chunk(batch_size, dim=1)

    def get_data(self):
        return {'states': self.states_over_time, 'actions': self.actions_over_time}