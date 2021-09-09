import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RocksDBDataset(Dataset):
    def __init__(self, X, y):
        super(RocksDBDataset, self).__init__()
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

class SingleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleNet, self).__init__()
        self.input_dim = input_dim # 3327
        self.hidden_dim = hidden_dim # 1024
        self.output_dim = output_dim # 148
        self.knob_fc = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU())
#         self.hidden = nn.Sequential(nn.Linear(self.hidden_dim, 64), nn.ReLU())
        self.im_fc = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        self.x_kb = self.knob_fc(x)
#         self.h = self.hidden(self.x_kb)
        self.x_im = self.im_fc(self.x_kb)
        return self.x_im