import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils import split_data
from model import HPOClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

HIDDEN_DIM = 1000
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, Y_batch in data_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_outputs.append(probs)
            all_targets.append(Y_batch.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    try:
        auprc = average_precision_score(all_targets, all_outputs, average='micro')
    except ValueError:
        auprc = np.nan 

    return avg_loss, auprc

class GeneHPODataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data()

    train_dataset = GeneHPODataset(X_train, Y_train)
    val_dataset   = GeneHPODataset(X_val, Y_val)
    test_dataset  = GeneHPODataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nDataLoader Batches: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

    in_dim = X_train.shape[1]
    out_dim = Y_train.shape[1]
    
    model = HPOClassifier(
        input_size=in_dim, 
        hidden_size=HIDDEN_DIM, 
        out_size=out_dim
    )
    model.to(DEVICE) 
    print("\nModel Architecture:\n", model)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Training...")
    best_val_auprc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE) 

            optimizer.zero_grad() 
            
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()
            
        avg_train_loss = total_loss / len(train_loader)
        
        avg_val_loss, val_auprc = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val AUPRC: {val_auprc:.4f}")

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), 'best_hpo_classifier.pth')

    print("\nStarting Final Test...")
    model.load_state_dict(torch.load('best_hpo_classifier.pth'))
    
    test_loss, test_auprc = evaluate(model, test_loader, criterion, DEVICE)

    print("---------------------------------------------")
    print(f"Final Test Loss (Best Model): {test_loss:.4f}")
    print(f"Final Test AUPRC (Best Model): {test_auprc:.4f}")
    print("---------------------------------------------")

    