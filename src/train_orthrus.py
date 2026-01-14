"""
Train HPO Classifier with 2 Embeddings: Omics + Orthrus

Supports:
- Fusion modes: 'add' or 'concat'
- Orthrus track types: '4track' or '6track'
- 5-Fold Cross Validation
- Results saved to res/ folder
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
import mygene

from model import HPOClassifier
from utils import plot_training_curves, plot_roc_curves, plot_prediction_distribution, plot_top_k_accuracy, plot_per_class_performance

# ==================== Configuration ====================
HIDDEN_DIM = 1000
BATCH_SIZE = 32
EPOCHS = 70
LEARNING_RATE = 0.0001
N_FOLDS = 5
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
    Load Omics + Orthrus embeddings and prepare data for 5-fold CV.
    
    Args:
        orthrus_track: Orthrus track type ('4track' or '6track')
        fusion_mode: 'add' or 'concat'
    
    Returns:
        X: Fused embeddings (N, dim)
        Y: HPO labels (N, num_hpo)
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
    else:  # concat
        concat_dim = omics_dim + orthrus_dim
        print(f"\nFusion mode: CONCAT")
        print(f"  Concat dim = {omics_dim} + {orthrus_dim} = {concat_dim}")
    
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
    
    # ==================== Fusion ====================
    if fusion_mode == "add":
        print("\n" + "=" * 70)
        print("Applying ADD Fusion (with padding)")
        print("=" * 70)
        
        max_dim = max(omics_dim, orthrus_dim)
        padded_omics = np.pad(X_omics, ((0, 0), (0, max_dim - omics_dim)), mode='constant')
        padded_orthrus = np.pad(X_orthrus, ((0, 0), (0, max_dim - orthrus_dim)), mode='constant')
        X = (padded_omics + padded_orthrus).astype(np.float32)
        
        print(f"Final dimension: {X.shape[1]}")
        
    else:  # concat
        print("\n" + "=" * 70)
        print("Applying CONCAT")
        print("=" * 70)
        
        X = np.concatenate([X_omics, X_orthrus], axis=1).astype(np.float32)
        
        print(f"Final dimension: {X.shape[1]}")
    
    print(f"\nFinal data shapes: X={X.shape}, Y={Y.shape}")
    
    return X, Y


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


# ==================== Training Function for One Fold ====================

def train_one_fold(model, train_loader, val_loader, criterion, optimizer, 
                   epochs, device, model_save_path):
    """Train model for one fold and return best val AUPRC."""
    best_val_auprc = 0.0
    train_losses = []
    val_losses = []
    train_auprcs = []
    val_auprcs = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Evaluate
        train_loss, train_auprc, _, _ = evaluate(model, train_loader, criterion, device)
        val_loss, val_auprc, _, _ = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_auprcs.append(train_auprc)
        val_auprcs.append(val_auprc)
        
        print(f"  Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Train AUPRC: {train_auprc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUPRC: {val_auprc:.4f}")
        
        # Save best model
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), model_save_path)
    
    return best_val_auprc, train_losses, val_losses, train_auprcs, val_auprcs


# ==================== Main Training Script ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train HPO classifier with Omics + Orthrus embeddings (5-Fold CV)')
    parser.add_argument('--fusion_mode', type=str, default='concat', choices=['concat', 'add'],
                        help='Fusion mode (default: concat)')
    parser.add_argument('--orthrus_track', type=str, default='4track', choices=['4track', '6track'],
                        help='Orthrus track type (default: 4track)')
    parser.add_argument('--n_folds', type=int, default=N_FOLDS,
                        help=f'Number of folds (default: {N_FOLDS})')
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
    print("Training HPO Classifier with 2 Embeddings (5-Fold CV)")
    print("Omics + Orthrus")
    print("=" * 70)
    print(f"Fusion Mode: {args.fusion_mode}")
    print(f"Orthrus Track: {args.orthrus_track}")
    print(f"Number of Folds: {args.n_folds}")
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
    X, Y = prepare_data_orthrus_omics(
        orthrus_track=args.orthrus_track,
        fusion_mode=args.fusion_mode
    )
    
    # ==================== 5-Fold Cross Validation ====================
    print("\n" + "=" * 70)
    print(f"Starting {args.n_folds}-Fold Cross Validation")
    print("=" * 70)
    
    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    
    in_dim = X.shape[1]
    out_dim = Y.shape[1]
    
    fold_results = []
    all_val_outputs = []
    all_val_targets = []
    
    prefix = f"orthrus_omics_{args.fusion_mode}_{args.orthrus_track}"
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n{'='*70}")
        print(f"Fold {fold + 1}/{args.n_folds}")
        print(f"{'='*70}")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        
        # Create datasets and dataloaders
        train_dataset = GeneHPODataset(X_train, Y_train)
        val_dataset = GeneHPODataset(X_val, Y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Initialize model
        model = HPOClassifier(
            input_size=in_dim,
            hidden_size=args.hidden_dim,
            out_size=out_dim
        )
        model.to(DEVICE)
        
        if fold == 0:
            print("\nModel Architecture:\n", model)
            print(f"Input dim: {in_dim}, Hidden dim: {args.hidden_dim}, Output dim: {out_dim}")
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Model save path for this fold
        model_save_path = os.path.join(res_dir, f'best_{prefix}_fold{fold+1}.pth')
        
        # Train
        print(f"\nStarting Training for Fold {fold + 1}...")
        best_val_auprc, train_losses, val_losses, train_auprcs, val_auprcs = train_one_fold(
            model, train_loader, val_loader, criterion, optimizer,
            args.epochs, DEVICE, model_save_path
        )
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_auprc': best_val_auprc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_auprcs': train_auprcs,
            'val_auprcs': val_auprcs
        })
        
        # Load best model and get predictions
        model.load_state_dict(torch.load(model_save_path))
        _, _, val_outputs, val_targets = evaluate(model, val_loader, criterion, DEVICE)
        all_val_outputs.append(val_outputs)
        all_val_targets.append(val_targets)
        
        print(f"\nFold {fold + 1} Best Val AUPRC: {best_val_auprc:.4f}")
        
        # Generate fold-specific plots
        plot_training_curves(train_losses, val_losses, train_auprcs, val_auprcs,
                            save_path=os.path.join(res_dir, f'{prefix}_fold{fold+1}_training_curves.png'))
    
    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("5-Fold Cross Validation Summary")
    print("=" * 70)
    
    auprcs = [r['best_val_auprc'] for r in fold_results]
    mean_auprc = np.mean(auprcs)
    std_auprc = np.std(auprcs)
    
    for r in fold_results:
        print(f"Fold {r['fold']}: Best Val AUPRC = {r['best_val_auprc']:.4f}")
    
    print("-" * 50)
    print(f"Mean AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}")
    print("-" * 50)
    
    # Concatenate all validation results for overall evaluation
    all_val_outputs = np.concatenate(all_val_outputs, axis=0)
    all_val_targets = np.concatenate(all_val_targets, axis=0)
    
    overall_auprc = average_precision_score(all_val_targets, all_val_outputs, average='micro')
    print(f"Overall AUPRC (all folds combined): {overall_auprc:.4f}")
    
    # Generate overall visualization plots
    print("\nGenerating overall visualization plots...")
    
    plot_roc_curves(all_val_targets, all_val_outputs, num_classes_to_plot=5,
                   save_path=os.path.join(res_dir, f'{prefix}_5fold_roc_curves.png'))
    
    plot_prediction_distribution(all_val_targets, all_val_outputs,
                                save_path=os.path.join(res_dir, f'{prefix}_5fold_prediction_distribution.png'))
    
    plot_top_k_accuracy(all_val_targets, all_val_outputs, k_values=[1, 3, 5, 10, 20, 50],
                       save_path=os.path.join(res_dir, f'{prefix}_5fold_top_k_accuracy.png'))
    
    plot_per_class_performance(all_val_targets, all_val_outputs, top_n=20,
                              save_path=os.path.join(res_dir, f'{prefix}_5fold_auprc_boxplot.png'))
    
    # Save summary results
    summary_path = os.path.join(res_dir, f'{prefix}_5fold_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"5-Fold Cross Validation Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Fusion Mode: {args.fusion_mode}\n")
        f.write(f"Orthrus Track: {args.orthrus_track}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Hidden Dim: {args.hidden_dim}\n")
        f.write(f"{'='*50}\n\n")
        for r in fold_results:
            f.write(f"Fold {r['fold']}: Best Val AUPRC = {r['best_val_auprc']:.4f}\n")
        f.write(f"\n{'='*50}\n")
        f.write(f"Mean AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}\n")
        f.write(f"Overall AUPRC (all folds): {overall_auprc:.4f}\n")
    
    print("\nAll visualization plots have been generated!")
    print(f"\nTraining completed!")
    print(f"  Models saved to: {res_dir}/best_{prefix}_fold*.pth")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Plots saved to: {res_dir}/")
