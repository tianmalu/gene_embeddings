import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class HPOClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1000, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    def forward(self, x): return self.net(x)