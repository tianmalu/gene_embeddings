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
    
class ImprovedHPOClassifier(nn.Module):
    def __init__(self, input_size, out_size, hidden_size=2048, dropout_prob=0.3):
        super().__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.output = nn.Linear(hidden_size, out_size)
        
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        x = self.bn1(F.relu(self.layer1(x)))
        x = self.dropout(x)
        
        x = self.bn2(F.relu(self.layer2(x)))
        x = self.dropout(x)
        
        return self.output(x)