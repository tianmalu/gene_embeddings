import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
import mygene
import joblib

from model import HPOClassifier
from utils import plot_training_curves, plot_roc_curves, plot_prediction_distribution, plot_top_k_accuracy, plot_per_class_performance

# ==================== Configuration ====================
HIDDEN_DIM = 1000
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PCA dimensions for each embedding type
PCA_OMICS_DIM = 256      # Omics: 256 -> 256
PCA_ESM2_DIM = 256       # ESM2: 1280 -> 256
PCA_ENFORMER_DIM = 256   # Enformer: 3072 -> 256

# Set random seed for reproducibility
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


def load_esm2_embeddings(esm2_model="650M"):
    """Load ESM2 protein embeddings from .npy files."""
    if esm2_model == "8M":
        esm2_dir = "../dataset/esm2_embeddings"
    elif esm2_model == "650M":
        esm2_dir = "../dataset/esm2_embeddings_650M"
    elif esm2_model == "35M":
        esm2_dir = "../dataset/esm2_embeddings_35M"
    else:
        raise ValueError(f"Unknown esm2_model: {esm2_model}. Use '8M', '35M', or '650M'.")
    
    gene2esm2 = {}
    
    if not os.path.exists(esm2_dir):
        print(f"Warning: ESM2 embedding directory not found: {esm2_dir}")
        return gene2esm2
    
    for fname in os.listdir(esm2_dir):
        if fname.endswith(".npy"):
            gene_symbol = fname.replace(".npy", "")
            emb_path = os.path.join(esm2_dir, fname)
            gene2esm2[gene_symbol] = np.load(emb_path)
    
    print(f"Loaded ESM2 ({esm2_model}) embeddings: {len(gene2esm2)} genes")
    if gene2esm2:
        sample_emb = next(iter(gene2esm2.values()))
        print(f"  ESM2 embedding dim = {sample_emb.shape[0]}")
    return gene2esm2


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


def load_data_with_pca(esm2_model="650M", 
                       pca_omics_dim=PCA_OMICS_DIM, 
                       pca_esm2_dim=PCA_ESM2_DIM, 
                       pca_enformer_dim=PCA_ENFORMER_DIM):
    """
    Load all three embeddings, apply PCA to each, then concatenate.
    
    Args:
        esm2_model: Which ESM2 model embeddings to use
        pca_omics_dim: Target dimension for Omics after PCA
        pca_esm2_dim: Target dimension for ESM2 after PCA
        pca_enformer_dim: Target dimension for Enformer after PCA
    
    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, pca_models
    """
    print("=" * 60)
    print("Loading 3 Embeddings with PCA Dimensionality Reduction")
    print("=" * 60)
    
    # Load all embeddings
    df_omics = load_omics_embedding()
    df_omics = map_ensembl_to_symbol(df_omics)
    gene2esm2 = load_esm2_embeddings(esm2_model=esm2_model)
    gene2hpos = load_hpo_labels()
    
    # Get available Enformer genes
    enformer_dir = "../dataset/enformer_embeddings_hf"
    enformer_available_genes = set()
    for fname in os.listdir(enformer_dir):
        if fname.endswith(".npy"):
            enformer_available_genes.add(fname.replace(".npy", ""))
    print(f"Enformer embeddings available: {len(enformer_available_genes)} genes")
    
    # Find common genes
    genes_omics = set(df_omics.index)
    genes_hpo = set(gene2hpos.index)
    genes_esm2 = set(gene2esm2.keys())
    genes_enformer = enformer_available_genes
    
    print(f"\nGene set sizes:")
    print(f"  Omics: {len(genes_omics)}")
    print(f"  HPO: {len(genes_hpo)}")
    print(f"  ESM2: {len(genes_esm2)}")
    print(f"  Enformer: {len(genes_enformer)}")
    
    genes_common = sorted(genes_omics & genes_hpo & genes_esm2 & genes_enformer)
    print(f"\nGenes with ALL four sources: {len(genes_common)}")
    
    # Build HPO term index
    hpo_terms = sorted({h for g in genes_common for h in gene2hpos[g]})
    hpo2idx = {h: i for i, h in enumerate(hpo_terms)}
    num_hpo = len(hpo_terms)
    print(f"Number of HPO terms: {num_hpo}")
    
    # ==================== Collect raw embeddings ====================
    print("\nCollecting raw embeddings...")
    
    N = len(genes_common)
    
    # Get dimensions
    sample_gene = genes_common[0]
    sample_enformer = np.load(os.path.join(enformer_dir, f"{sample_gene}.npy"))
    if sample_enformer.ndim == 2:
        sample_enformer = sample_enformer.mean(axis=0)
    
    omics_dim = df_omics.loc[sample_gene].values.shape[0]
    esm2_dim = gene2esm2[sample_gene].shape[0]
    enformer_dim = sample_enformer.shape[0]
    
    print(f"\nOriginal embedding dimensions:")
    print(f"  Omics: {omics_dim}")
    print(f"  ESM2: {esm2_dim}")
    print(f"  Enformer: {enformer_dim}")
    
    # Pre-allocate arrays for raw embeddings
    X_omics = np.zeros((N, omics_dim), dtype=np.float32)
    X_esm2 = np.zeros((N, esm2_dim), dtype=np.float32)
    X_enformer = np.zeros((N, enformer_dim), dtype=np.float32)
    Y = np.zeros((N, num_hpo), dtype=np.float32)
    
    for i, g in enumerate(genes_common):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{N} genes...")
        
        # Omics
        X_omics[i] = df_omics.loc[g].values.astype(np.float32)
        
        # ESM2
        X_esm2[i] = gene2esm2[g].astype(np.float32)
        
        # Enformer
        x_enformer = np.load(os.path.join(enformer_dir, f"{g}.npy")).astype(np.float32)
        if x_enformer.ndim == 2:
            x_enformer = x_enformer.mean(axis=0)
        X_enformer[i] = x_enformer
        
        # Labels
        for h in gene2hpos[g]:
            Y[i, hpo2idx[h]] = 1.0
    
    print(f"  Done! Collected embeddings for {N} genes.")
    
    # Clear ESM2 dict from memory
    del gene2esm2
    import gc
    gc.collect()
    
    # ==================== Train/Val/Test Split ====================
    print("\nSplitting data...")
    indices = np.arange(N)
    idx_train, idx_tmp = train_test_split(indices, test_size=0.3, random_state=42)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=42)
    
    # Split each embedding type
    X_omics_train, X_omics_val, X_omics_test = X_omics[idx_train], X_omics[idx_val], X_omics[idx_test]
    X_esm2_train, X_esm2_val, X_esm2_test = X_esm2[idx_train], X_esm2[idx_val], X_esm2[idx_test]
    X_enformer_train, X_enformer_val, X_enformer_test = X_enformer[idx_train], X_enformer[idx_val], X_enformer[idx_test]
    Y_train, Y_val, Y_test = Y[idx_train], Y[idx_val], Y[idx_test]
    
    print(f"Split sizes: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")
    
    # ==================== Apply PCA (fit on training data only) ====================
    print("\n" + "=" * 60)
    print("Applying PCA Dimensionality Reduction")
    print("=" * 60)
    
    pca_models = {}
    
    # PCA for Omics
    pca_omics_dim = min(pca_omics_dim, omics_dim, X_omics_train.shape[0])
    print(f"\nOmics PCA: {omics_dim} -> {pca_omics_dim}")
    pca_omics = PCA(n_components=pca_omics_dim, random_state=SEED)
    X_omics_train_pca = pca_omics.fit_transform(X_omics_train)
    X_omics_val_pca = pca_omics.transform(X_omics_val)
    X_omics_test_pca = pca_omics.transform(X_omics_test)
    print(f"  Explained variance ratio: {pca_omics.explained_variance_ratio_.sum():.4f}")
    pca_models['omics'] = pca_omics
    
    # PCA for ESM2
    pca_esm2_dim = min(pca_esm2_dim, esm2_dim, X_esm2_train.shape[0])
    print(f"\nESM2 PCA: {esm2_dim} -> {pca_esm2_dim}")
    pca_esm2 = PCA(n_components=pca_esm2_dim, random_state=SEED)
    X_esm2_train_pca = pca_esm2.fit_transform(X_esm2_train)
    X_esm2_val_pca = pca_esm2.transform(X_esm2_val)
    X_esm2_test_pca = pca_esm2.transform(X_esm2_test)
    print(f"  Explained variance ratio: {pca_esm2.explained_variance_ratio_.sum():.4f}")
    pca_models['esm2'] = pca_esm2
    
    # PCA for Enformer
    pca_enformer_dim = min(pca_enformer_dim, enformer_dim, X_enformer_train.shape[0])
    print(f"\nEnformer PCA: {enformer_dim} -> {pca_enformer_dim}")
    pca_enformer = PCA(n_components=pca_enformer_dim, random_state=SEED)
    X_enformer_train_pca = pca_enformer.fit_transform(X_enformer_train)
    X_enformer_val_pca = pca_enformer.transform(X_enformer_val)
    X_enformer_test_pca = pca_enformer.transform(X_enformer_test)
    print(f"  Explained variance ratio: {pca_enformer.explained_variance_ratio_.sum():.4f}")
    pca_models['enformer'] = pca_enformer
    
    # ==================== Concatenate PCA-reduced embeddings ====================
    print("\n" + "=" * 60)
    print("Concatenating PCA-reduced embeddings")
    print("=" * 60)
    
    X_train = np.concatenate([X_omics_train_pca, X_esm2_train_pca, X_enformer_train_pca], axis=1)
    X_val = np.concatenate([X_omics_val_pca, X_esm2_val_pca, X_enformer_val_pca], axis=1)
    X_test = np.concatenate([X_omics_test_pca, X_esm2_test_pca, X_enformer_test_pca], axis=1)
    
    final_dim = X_train.shape[1]
    print(f"\nFinal concatenated dimension: {pca_omics_dim} + {pca_esm2_dim} + {pca_enformer_dim} = {final_dim}")
    print(f"Final data shapes: X_train={X_train.shape}, Y_train={Y_train.shape}")
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, pca_models


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
    
    parser = argparse.ArgumentParser(description='Train HPO classifier with PCA + Concat')
    parser.add_argument('--esm2_model', type=str, default='650M', choices=['8M', '35M', '650M'],
                        help='ESM2 model size (default: 650M)')
    parser.add_argument('--pca_omics', type=int, default=PCA_OMICS_DIM,
                        help=f'PCA dimension for Omics (default: {PCA_OMICS_DIM})')
    parser.add_argument('--pca_esm2', type=int, default=PCA_ESM2_DIM,
                        help=f'PCA dimension for ESM2 (default: {PCA_ESM2_DIM})')
    parser.add_argument('--pca_enformer', type=int, default=PCA_ENFORMER_DIM,
                        help=f'PCA dimension for Enformer (default: {PCA_ENFORMER_DIM})')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of epochs (default: {EPOCHS})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM,
                        help=f'Hidden dimension (default: {HIDDEN_DIM})')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Training HPO Classifier with PCA + Concat")
    print("Omics + ESM2 + Enformer (PCA reduced)")
    print("=" * 60)
    print(f"ESM2 Model: {args.esm2_model}")
    print(f"PCA Dimensions: Omics={args.pca_omics}, ESM2={args.pca_esm2}, Enformer={args.pca_enformer}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Hidden Dim: {args.hidden_dim}")
    print(f"Device: {DEVICE}")
    print("=" * 60 + "\n")
    
    # Result directory
    res_dir = os.path.join(os.path.dirname(__file__), 'res')
    os.makedirs(res_dir, exist_ok=True)
    
    # Load and split data with PCA
    X_train, Y_train, X_val, Y_val, X_test, Y_test, pca_models = load_data_with_pca(
        esm2_model=args.esm2_model,
        pca_omics_dim=args.pca_omics,
        pca_esm2_dim=args.pca_esm2,
        pca_enformer_dim=args.pca_enformer
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
    
    # Model save paths
    prefix = f"pca_{args.pca_omics}_{args.pca_esm2}_{args.pca_enformer}_{args.esm2_model}"
    model_save_path = os.path.join(res_dir, f'best_{prefix}.pth')
    pca_save_path = os.path.join(res_dir, f'pca_models_{prefix}.pkl')
    
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
            # Also save PCA models
            joblib.dump(pca_models, pca_save_path)
            print(f"  -> New best model saved! Val AUPRC: {val_auprc:.4f}")
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    print(f"Loading best model from: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
    
    test_loss, test_auprc, test_outputs, test_targets = evaluate(model, test_loader, criterion, DEVICE)
    
    print("-" * 45)
    print(f"Final Test Loss (Best Model): {test_loss:.4f}")
    print(f"Final Test AUPRC (Best Model): {test_auprc:.4f}")
    print("-" * 45)
    
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
    print(f"  PCA models saved to: {pca_save_path}")
