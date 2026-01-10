import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils import split_data, plot_training_curves, plot_roc_curves, plot_prediction_distribution, plot_top_k_accuracy, plot_per_class_performance
from model import HPOClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

HIDDEN_DIM = 1000
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

    return avg_loss, auprc, all_outputs, all_targets

class GeneHPODataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(use_esm2=True, fusion_mode="concat", esm2_model="8M")


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
    
    train_losses = []
    val_losses = []
    train_auprcs = []
    val_auprcs = []

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
        
        train_loss, train_auprc, _, _ = evaluate(model, train_loader, criterion, DEVICE)
        avg_val_loss, val_auprc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(avg_val_loss)
        train_auprcs.append(train_auprc)
        val_auprcs.append(val_auprc)

        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Train AUPRC: {train_auprc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val AUPRC: {val_auprc:.4f}")

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), 'best_hpo_classifier.pth')

    print("\nStarting Final Test...")
    print("Loading best model from: best_hpo_classifier.pth")
    model.load_state_dict(torch.load('best_hpo_classifier.pth'))
    
    test_loss, test_auprc, test_outputs, test_targets = evaluate(model, test_loader, criterion, DEVICE)

    print("---------------------------------------------")
    print(f"Final Test Loss (Best Model): {test_loss:.4f}")
    print(f"Final Test AUPRC (Best Model): {test_auprc:.4f}")
    print("---------------------------------------------")

    print("\nGenerating training visualization plots...")
    
    plot_training_curves(train_losses, val_losses, train_auprcs, val_auprcs, 
                        save_path='training_curves.png')
    
    plot_roc_curves(test_targets, test_outputs, num_classes_to_plot=5, 
                   save_path='roc_curves.png')
    
    plot_prediction_distribution(test_targets, test_outputs, 
                                save_path='prediction_distribution.png')
    
    plot_top_k_accuracy(test_targets, test_outputs, k_values=[1, 3, 5, 10, 20, 50], 
                       save_path='top_k_accuracy.png')
    
    print("\nGenerating AUPRC box plot (Best Model)...")
    plot_per_class_performance(test_targets, test_outputs, top_n=20, 
                              save_path='auprc_boxplot.png')
    
    print("\nAll visualization plots have been generated!")

    