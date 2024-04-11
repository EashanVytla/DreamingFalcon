import pandas as pd
import torch
from torch.utils.data import Dataset

class Replay(Dataset):
    def __init__(self, max_size=1000, sequence_length=32, history_size=15):
        self.max_size = max_size
        self.sequence_length = sequence_length
        self.history_size = history_size
        self.data = pd.DataFrame(columns=[f'feature_{i}' for i in range(15)])

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx:idx+self.sequence_length-self.history_size].values)

    def add(self, data):
        """
        Add data to the replay buffer.
        If the buffer exceeds the maximum size, the oldest data is removed.
        """
        new_data = pd.DataFrame(data, columns=self.data.columns)
        self.data = pd.concat([self.data, new_data], ignore_index=True)
        if len(self.data) > self.max_size:
            self.data = self.data.iloc[-self.max_size:]

    def clear(self):
        """
        Clear the replay buffer.
        """
        self.data = pd.DataFrame(columns=[f'feature_{i}' for i in range(15)])