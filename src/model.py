import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class HPOClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, dropout_prob=0.2):
        super().__init__()
        
        self.l1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob) 
        self.l2 = nn.Linear(hidden_size, out_size)
    
    def forward(self, x):
        out = F.relu(self.l1(x))
        out = self.dropout(out)
        out = self.l2(out)
        return out