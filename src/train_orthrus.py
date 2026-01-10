"""
Train HPO Classifier with 2 Embeddings: Omics + Orthrus

Supports:
- Fusion modes: 'add' or 'concat'
- Orthrus track types: '4track' or '6track'
- Results saved to res/ folder
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import mygene

from model import HPOClassifier
from utils import plot_training_curves, plot_roc_curves, plot_prediction_distribution, plot_top_k_accuracy, plot_per_class_performance

# ==================== Configuration ====================
HIDDEN_DIM = 1000
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
import random
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==================== Data Loading Functions ====================

def load_omics_embedding():
    """Load Omics embeddings from TSV file."""
    data_path = "../dataset/"
    omic_path = "omics_embeddings"
    emb_path = data_path + omic_path + "/Supplementary_Table_S3_OMICS_EMB.tsv"

    df_emb = pd.read_csv(emb_path, sep="\t")
    gene_col = "gene_id"
    df_emb = df_emb.set_index(gene_col)

    print(f"Loaded Omics embeddings: {df_emb.shape[0]} genes, dim = {df_emb.shape[1]}")
    return df_emb


def load_orthrus_embeddings(track_type="4track"):
    """Load Orthrus embeddings from pkl file."""
    orthrus_path = f"../dataset/orthrus_embeddings/orthrus_{track_type}.pkl"
    
    if not os.path.exists(orthrus_path):
        print(f"Warning: Orthrus embedding file not found: {orthrus_path}")
        return {}
    
    with open(orthrus_path, 'rb') as f:
        gene2orthrus = pickle.load(f)
    
    print(f"Loaded Orthrus ({track_type}) embeddings: {len(gene2orthrus)} genes")
    if gene2orthrus:
        sample_emb = next(iter(gene2orthrus.values()))
        print(f"  Orthrus embedding dim = {sample_emb.shape[0]}")
    return gene2orthrus


def map_ensembl_to_symbol(df_emb):
    """Map Ensembl gene IDs to gene symbols using mygene."""
    mg = mygene.MyGeneInfo()
    ensg_ids = df_emb.index.tolist()

    print("Querying mygene for gene symbols ...")
    out = mg.querymany(
        ensg_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=True
    )

    ensg2symbol = out["symbol"].to_dict()
    df_emb["gene_symbol"] = df_emb.index.map(ensg2symbol)
    df_emb = df_emb.dropna(subset=["gene_symbol"])
    df_emb = df_emb.drop_duplicates(subset=["gene_symbol"], keep='first')
    df_emb = df_emb.set_index("gene_symbol")

    print(f"After mapping and deduplication: {df_emb.shape[0]} genes with symbols")
    return df_emb


def load_hpo_labels():
    """Load HPO gene-phenotype labels."""
    hpo_path = "../dataset/genes_to_phenotype.txt"
    df_hpo = pd.read_csv(hpo_path, sep="\t", comment="#")
    print("HPO columns:", df_hpo.columns.tolist())

    df_hpo = df_hpo[["gene_symbol", "hpo_id"]].dropna()
    gene2hpos = df_hpo.groupby("gene_symbol")["hpo_id"].apply(list)
    print(f"Genes with HPO labels: {len(gene2hpos)}")
    return gene2hpos


# ==================== Data Preparation ====================

def prepare_data_orthrus_omics(orthrus_track="4track", fusion_mode="concat"):
    """
    Load Omics + Orthrus embeddings and prepare data for training.
    
    Args:
        orthrus_track: Orthrus track type ('4track' or '6track')
        fusion_mode: 'add' or 'concat'
    
    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    print("=" * 70)
    print("Loading 2 Embeddings: Omics + Orthrus")
    print("=" * 70)
    
    # Load embeddings
    df_omics = load_omics_embedding()
    df_omics = map_ensembl_to_symbol(df_omics)
    gene2orthrus = load_orthrus_embeddings(track_type=orthrus_track)
    gene2hpos = load_hpo_labels()
    
    # Find common genes
    genes_omics = set(df_omics.index)
    genes_hpo = set(gene2hpos.index)
    genes_orthrus = set(gene2orthrus.keys())
    
    print(f"\nGene set sizes:")
    print(f"  Omics: {len(genes_omics)}")
    print(f"  HPO: {len(genes_hpo)}")
    print(f"  Orthrus: {len(genes_orthrus)}")
    
    genes_common = sorted(genes_omics & genes_hpo & genes_orthrus)
    print(f"\nGenes with ALL 3 sources: {len(genes_common)}")
    
    # Build HPO term index
    hpo_terms = sorted({h for g in genes_common for h in gene2hpos[g]})
    hpo2idx = {h: i for i, h in enumerate(hpo_terms)}
    num_hpo = len(hpo_terms)
    print(f"Number of HPO terms: {num_hpo}")
    
    # Get dimensions
    sample_gene = genes_common[0]
    omics_dim = df_omics.loc[sample_gene].values.shape[0]
    orthrus_dim = gene2orthrus[sample_gene].shape[0]
    
    print(f"\nEmbedding dimensions:")
    print(f"  Omics: {omics_dim}")
    print(f"  Orthrus: {orthrus_dim}")
    
    # ==================== Collect embeddings ====================
    N = len(genes_common)
    
    if fusion_mode == "add":
        max_dim = max(omics_dim, orthrus_dim)
        print(f"\nFusion mode: ADD (element-wise)")
        print(f"  Max dim = {max_dim}")
        output_dim = max_dim
    else:  # concat
        concat_dim = omics_dim + orthrus_dim
        print(f"\nFusion mode: CONCAT")
        print(f"  Concat dim = {omics_dim} + {orthrus_dim} = {concat_dim}")
        output_dim = concat_dim
    
    print(f"\nCollecting embeddings for {N} genes...")
    
    # Pre-allocate arrays
    X_omics = np.zeros((N, omics_dim), dtype=np.float32)
    X_orthrus = np.zeros((N, orthrus_dim), dtype=np.float32)
    Y = np.zeros((N, num_hpo), dtype=np.float32)
    
    for i, g in enumerate(genes_common):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{N} genes...")
        
        X_omics[i] = df_omics.loc[g].values.astype(np.float32)
        X_orthrus[i] = gene2orthrus[g].astype(np.float32)
        
        for h in gene2hpos[g]:
            Y[i, hpo2idx[h]] = 1.0
    
    print(f"  Done! Collected all embeddings.")
    
    # ==================== Train/Val/Test Split ====================
    print("\nSplitting data (70/15/15)...")
    indices = np.arange(N)
    idx_train, idx_tmp = train_test_split(indices, test_size=0.3, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=SEED)
    
    print(f"Split sizes: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")
    
    # Split each embedding
    X_omics_train, X_omics_val, X_omics_test = X_omics[idx_train], X_omics[idx_val], X_omics[idx_test]
    X_orthrus_train, X_orthrus_val, X_orthrus_test = X_orthrus[idx_train], X_orthrus[idx_val], X_orthrus[idx_test]
    Y_train, Y_val, Y_test = Y[idx_train], Y[idx_val], Y[idx_test]
    
    # ==================== Fusion ====================
    if fusion_mode == "add":
        print("\n" + "=" * 70)
        print("Applying ADD Fusion (with padding)")
        print("=" * 70)
        
        def pad_and_add(arr1, arr2, max_dim):
            result = np.zeros((arr1.shape[0], max_dim), dtype=np.float32)
            padded1 = np.pad(arr1, ((0, 0), (0, max_dim - arr1.shape[1])), mode='constant')
            padded2 = np.pad(arr2, ((0, 0), (0, max_dim - arr2.shape[1])), mode='constant')
            result = padded1 + padded2
            return result
        
        X_train = pad_and_add(X_omics_train, X_orthrus_train, max_dim)
        X_val = pad_and_add(X_omics_val, X_orthrus_val, max_dim)
        X_test = pad_and_add(X_omics_test, X_orthrus_test, max_dim)
        
        print(f"Final dimension: {X_train.shape[1]}")
        
    else:  # concat
        print("\n" + "=" * 70)
        print("Applying CONCAT")
        print("=" * 70)
        
        X_train = np.concatenate([X_omics_train, X_orthrus_train], axis=1)
        X_val = np.concatenate([X_omics_val, X_orthrus_val], axis=1)
        X_test = np.concatenate([X_omics_test, X_orthrus_test], axis=1)
        
        print(f"Final dimension: {X_train.shape[1]}")
    
    print(f"\nFinal data shapes: X_train={X_train.shape}, Y_train={Y_train.shape}")
    
    return X_train.astype(np.float32), Y_train, X_val.astype(np.float32), Y_val, X_test.astype(np.float32), Y_test


# ==================== Dataset Class ====================

class GeneHPODataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ==================== Evaluation Function ====================

def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a data loader."""
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


# ==================== Main Training Script ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train HPO classifier with Omics + Orthrus embeddings')
    parser.add_argument('--fusion_mode', type=str, default='concat', choices=['concat', 'add'],
                        help='Fusion mode (default: concat)')
    parser.add_argument('--orthrus_track', type=str, default='4track', choices=['4track', '6track'],
                        help='Orthrus track type (default: 4track)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of epochs (default: {EPOCHS})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM,
                        help=f'Hidden dimension (default: {HIDDEN_DIM})')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("Training HPO Classifier with 2 Embeddings")
    print("Omics + Orthrus")
    print("=" * 70)
    print(f"Fusion Mode: {args.fusion_mode}")
    print(f"Orthrus Track: {args.orthrus_track}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Hidden Dim: {args.hidden_dim}")
    print(f"Device: {DEVICE}")
    print("=" * 70 + "\n")
    
    # Result directory
    res_dir = os.path.join(os.path.dirname(__file__), 'res')
    os.makedirs(res_dir, exist_ok=True)
    
    # Load and prepare data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data_orthrus_omics(
        orthrus_track=args.orthrus_track,
        fusion_mode=args.fusion_mode
    )
    
    # Create datasets and dataloaders
    train_dataset = GeneHPODataset(X_train, Y_train)
    val_dataset = GeneHPODataset(X_val, Y_val)
    test_dataset = GeneHPODataset(X_test, Y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"\nDataLoader Batches: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    
    # Initialize model
    in_dim = X_train.shape[1]
    out_dim = Y_train.shape[1]
    
    model = HPOClassifier(
        input_size=in_dim,
        hidden_size=args.hidden_dim,
        out_size=out_dim
    )
    model.to(DEVICE)
    print("\nModel Architecture:\n", model)
    print(f"Input dim: {in_dim}, Hidden dim: {args.hidden_dim}, Output dim: {out_dim}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\nStarting Training...")
    best_val_auprc = 0.0
    
    train_losses = []
    val_losses = []
    train_auprcs = []
    val_auprcs = []
    
    # Model save path
    prefix = f"orthrus_omics_{args.fusion_mode}_{args.orthrus_track}"
    model_save_path = os.path.join(res_dir, f'best_{prefix}.pth')
    
    for epoch in range(args.epochs):
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
        
        # Evaluate
        train_loss, train_auprc, _, _ = evaluate(model, train_loader, criterion, DEVICE)
        val_loss, val_auprc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_auprcs.append(train_auprc)
        val_auprcs.append(val_auprc)
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Train AUPRC: {train_auprc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUPRC: {val_auprc:.4f}")
        
        # Save best model
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> New best model saved! Val AUPRC: {val_auprc:.4f}")
    
    # Final test evaluation
    print("\n" + "=" * 70)
    print("Final Test Evaluation")
    print("=" * 70)
    print(f"Loading best model from: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
    
    test_loss, test_auprc, test_outputs, test_targets = evaluate(model, test_loader, criterion, DEVICE)
    
    print("-" * 50)
    print(f"Final Test Loss (Best Model): {test_loss:.4f}")
    print(f"Final Test AUPRC (Best Model): {test_auprc:.4f}")
    print("-" * 50)
    
    # Generate visualization plots
    print("\nGenerating visualization plots...")
    
    plot_training_curves(train_losses, val_losses, train_auprcs, val_auprcs,
                        save_path=os.path.join(res_dir, f'{prefix}_training_curves.png'))
    
    plot_roc_curves(test_targets, test_outputs, num_classes_to_plot=5,
                   save_path=os.path.join(res_dir, f'{prefix}_roc_curves.png'))
    
    plot_prediction_distribution(test_targets, test_outputs,
                                save_path=os.path.join(res_dir, f'{prefix}_prediction_distribution.png'))
    
    plot_top_k_accuracy(test_targets, test_outputs, k_values=[1, 3, 5, 10, 20, 50],
                       save_path=os.path.join(res_dir, f'{prefix}_top_k_accuracy.png'))
    
    plot_per_class_performance(test_targets, test_outputs, top_n=20,
                              save_path=os.path.join(res_dir, f'{prefix}_auprc_boxplot.png'))
    
    print("\nAll visualization plots have been generated!")
    print(f"\nTraining completed!")
    print(f"  Best model saved to: {model_save_path}")
    print(f"  Plots saved to: {res_dir}/")
