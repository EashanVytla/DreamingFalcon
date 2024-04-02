import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os

class Pipeline:
    def __init__(self, csv_path_states, csv_path_actions):
        self.csv_path_states = csv_path_states
        self.csv_path_actions = csv_path_actions
        self.states_dataloader = None
        self.actions_dataloader = None

    def read_csv(self, sequence_length=25, batch_size=500):
        self.dataloader = DataLoader(SequenceDataset(self.csv_path_states, self.csv_path_actions, sequence_length), num_workers=os.cpu_count(), batch_size=batch_size, shuffle=True, pin_memory=True)

        return self.dataloader
    
class SequenceDataset(Dataset):
    def __init__(self, states_file, actions_file, seq_len=25):
        self.states = pd.read_csv(states_file)
        self.actions = pd.read_csv(actions_file)
        self.seq_len = seq_len

    def __len__(self):
        length = len(self.states) - self.seq_len + 1
        print(length)
        return length

    def __getitem__(self, idx):
        states_seq = self.states.iloc[idx:idx+self.seq_len, 1:].values
        actions_seq = self.actions.iloc[idx:idx+self.seq_len, 1:].values
        return states_seq, actions_seq