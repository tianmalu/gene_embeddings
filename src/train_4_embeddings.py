"""
Train HPO Classifier with 4 Embeddings: Omics + ESM2 + Enformer + Orthrus

Supports:
- Fusion modes: 'add' or 'concat'
- For concat: 'early_pca' (PCA each embedding separately) or 'late_pca' (PCA after concat) or 'none'
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
from sklearn.decomposition import PCA
import mygene
import joblib
import gc

from model import HPOClassifier
from utils import plot_training_curves, plot_roc_curves, plot_prediction_distribution, plot_top_k_accuracy, plot_per_class_performance

# ==================== Configuration ====================
HIDDEN_DIM = 1000
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PCA dimensions
PCA_OMICS_DIM = 256
PCA_ESM2_DIM = 256
PCA_ENFORMER_DIM = 256
PCA_ORTHRUS_DIM = 256
PCA_LATE_DIM = 512  # For late PCA (after concat)

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


def load_esm2_embeddings(esm2_model="650M"):
    """Load ESM2 protein embeddings from .npy files."""
    if esm2_model == "8M":
        esm2_dir = "../dataset/esm2_embeddings"
    elif esm2_model == "650M":
        esm2_dir = "../dataset/esm2_embeddings_650M"
    elif esm2_model == "35M":
        esm2_dir = "../dataset/esm2_embeddings_35M"
    else:
        raise ValueError(f"Unknown esm2_model: {esm2_model}.")
    
    gene2esm2 = {}
    if not os.path.exists(esm2_dir):
        print(f"Warning: ESM2 embedding directory not found: {esm2_dir}")
        return gene2esm2
    
    for fname in os.listdir(esm2_dir):
        if fname.endswith(".npy"):
            gene_symbol = fname.replace(".npy", "")
            gene2esm2[gene_symbol] = np.load(os.path.join(esm2_dir, fname))
    
    print(f"Loaded ESM2 ({esm2_model}) embeddings: {len(gene2esm2)} genes")
    if gene2esm2:
        sample_emb = next(iter(gene2esm2.values()))
        print(f"  ESM2 embedding dim = {sample_emb.shape[0]}")
    return gene2esm2


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


def get_enformer_available_genes():
    """Get list of available Enformer genes without loading data."""
    enformer_dir = "../dataset/enformer_embeddings_hf"
    available_genes = set()
    if os.path.exists(enformer_dir):
        for fname in os.listdir(enformer_dir):
            if fname.endswith(".npy"):
                available_genes.add(fname.replace(".npy", ""))
    print(f"Enformer embeddings available: {len(available_genes)} genes")
    return available_genes


def load_enformer_embedding(gene, enformer_dir="../dataset/enformer_embeddings_hf"):
    """Load single Enformer embedding."""
    emb = np.load(os.path.join(enformer_dir, f"{gene}.npy"))
    if emb.ndim == 2:
        emb = emb.mean(axis=0)
    return emb.astype(np.float32)


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

def prepare_data_4_embeddings(esm2_model="650M", orthrus_track="4track", 
                               fusion_mode="concat", pca_mode="none",
                               pca_omics_dim=PCA_OMICS_DIM, pca_esm2_dim=PCA_ESM2_DIM,
                               pca_enformer_dim=PCA_ENFORMER_DIM, pca_orthrus_dim=PCA_ORTHRUS_DIM,
                               pca_late_dim=PCA_LATE_DIM):
    """
    Load 4 embeddings and prepare data for training.
    
    Args:
        esm2_model: ESM2 model size ('8M', '35M', '650M')
        orthrus_track: Orthrus track type ('4track' or '6track')
        fusion_mode: 'add' or 'concat'
        pca_mode: 'none', 'early_pca', 'late_pca' (only for concat mode)
        pca_*_dim: PCA dimensions for each embedding (early_pca) or total (late_pca)
    
    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, pca_models (if pca applied)
    """
    print("=" * 70)
    print("Loading 4 Embeddings: Omics + ESM2 + Enformer + Orthrus")
    print("=" * 70)
    
    # Load all embeddings
    df_omics = load_omics_embedding()
    df_omics = map_ensembl_to_symbol(df_omics)
    gene2esm2 = load_esm2_embeddings(esm2_model=esm2_model)
    gene2orthrus = load_orthrus_embeddings(track_type=orthrus_track)
    gene2hpos = load_hpo_labels()
    enformer_genes = get_enformer_available_genes()
    
    # Find common genes across all 5 sources
    genes_omics = set(df_omics.index)
    genes_hpo = set(gene2hpos.index)
    genes_esm2 = set(gene2esm2.keys())
    genes_orthrus = set(gene2orthrus.keys())
    genes_enformer = enformer_genes
    
    print(f"\nGene set sizes:")
    print(f"  Omics: {len(genes_omics)}")
    print(f"  HPO: {len(genes_hpo)}")
    print(f"  ESM2: {len(genes_esm2)}")
    print(f"  Enformer: {len(genes_enformer)}")
    print(f"  Orthrus: {len(genes_orthrus)}")
    
    genes_common = sorted(genes_omics & genes_hpo & genes_esm2 & genes_enformer & genes_orthrus)
    print(f"\nGenes with ALL 5 sources: {len(genes_common)}")
    
    # Build HPO term index
    hpo_terms = sorted({h for g in genes_common for h in gene2hpos[g]})
    hpo2idx = {h: i for i, h in enumerate(hpo_terms)}
    num_hpo = len(hpo_terms)
    print(f"Number of HPO terms: {num_hpo}")
    
    # Get dimensions
    sample_gene = genes_common[0]
    omics_dim = df_omics.loc[sample_gene].values.shape[0]
    esm2_dim = gene2esm2[sample_gene].shape[0]
    enformer_dim = load_enformer_embedding(sample_gene).shape[0]
    orthrus_dim = gene2orthrus[sample_gene].shape[0]
    
    print(f"\nEmbedding dimensions:")
    print(f"  Omics: {omics_dim}")
    print(f"  ESM2: {esm2_dim}")
    print(f"  Enformer: {enformer_dim}")
    print(f"  Orthrus: {orthrus_dim}")
    
    # ==================== Collect embeddings ====================
    N = len(genes_common)
    
    if fusion_mode == "add":
        max_dim = max(omics_dim, esm2_dim, enformer_dim, orthrus_dim)
        print(f"\nFusion mode: ADD (element-wise)")
        print(f"  Max dim = {max_dim}")
        output_dim = max_dim
    else:  # concat
        concat_dim = omics_dim + esm2_dim + enformer_dim + orthrus_dim
        print(f"\nFusion mode: CONCAT")
        print(f"  Concat dim = {omics_dim} + {esm2_dim} + {enformer_dim} + {orthrus_dim} = {concat_dim}")
        output_dim = concat_dim
    
    print(f"\nCollecting embeddings for {N} genes...")
    
    # Pre-allocate arrays
    X_omics = np.zeros((N, omics_dim), dtype=np.float32)
    X_esm2 = np.zeros((N, esm2_dim), dtype=np.float32)
    X_enformer = np.zeros((N, enformer_dim), dtype=np.float32)
    X_orthrus = np.zeros((N, orthrus_dim), dtype=np.float32)
    Y = np.zeros((N, num_hpo), dtype=np.float32)
    
    for i, g in enumerate(genes_common):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{N} genes...")
        
        X_omics[i] = df_omics.loc[g].values.astype(np.float32)
        X_esm2[i] = gene2esm2[g].astype(np.float32)
        X_enformer[i] = load_enformer_embedding(g)
        X_orthrus[i] = gene2orthrus[g].astype(np.float32)
        
        for h in gene2hpos[g]:
            Y[i, hpo2idx[h]] = 1.0
    
    print(f"  Done! Collected all embeddings.")
    
    # Clear memory
    del gene2esm2, gene2orthrus
    gc.collect()
    
    # ==================== Train/Val/Test Split ====================
    print("\nSplitting data (70/15/15)...")
    indices = np.arange(N)
    idx_train, idx_tmp = train_test_split(indices, test_size=0.3, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=SEED)
    
    print(f"Split sizes: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")
    
    # Split each embedding
    X_omics_train, X_omics_val, X_omics_test = X_omics[idx_train], X_omics[idx_val], X_omics[idx_test]
    X_esm2_train, X_esm2_val, X_esm2_test = X_esm2[idx_train], X_esm2[idx_val], X_esm2[idx_test]
    X_enformer_train, X_enformer_val, X_enformer_test = X_enformer[idx_train], X_enformer[idx_val], X_enformer[idx_test]
    X_orthrus_train, X_orthrus_val, X_orthrus_test = X_orthrus[idx_train], X_orthrus[idx_val], X_orthrus[idx_test]
    Y_train, Y_val, Y_test = Y[idx_train], Y[idx_val], Y[idx_test]
    
    pca_models = {}
    
    # ==================== Fusion ====================
    if fusion_mode == "add":
        print("\n" + "=" * 70)
        print("Applying ADD Fusion (with padding)")
        print("=" * 70)
        
        def pad_and_add(*arrays, max_dim):
            result = np.zeros((arrays[0].shape[0], max_dim), dtype=np.float32)
            for arr in arrays:
                padded = np.pad(arr, ((0, 0), (0, max_dim - arr.shape[1])), mode='constant')
                result += padded
            return result
        
        X_train = pad_and_add(X_omics_train, X_esm2_train, X_enformer_train, X_orthrus_train, max_dim=max_dim)
        X_val = pad_and_add(X_omics_val, X_esm2_val, X_enformer_val, X_orthrus_val, max_dim=max_dim)
        X_test = pad_and_add(X_omics_test, X_esm2_test, X_enformer_test, X_orthrus_test, max_dim=max_dim)
        
        print(f"Final dimension: {X_train.shape[1]}")
        
    else:  # concat
        if pca_mode == "early_pca":
            print("\n" + "=" * 70)
            print("Applying EARLY PCA (PCA each embedding, then concat)")
            print("=" * 70)
            
            # PCA for Omics
            pca_omics_dim_adj = min(pca_omics_dim, omics_dim, len(idx_train))
            print(f"\nOmics PCA: {omics_dim} -> {pca_omics_dim_adj}")
            pca_omics = PCA(n_components=pca_omics_dim_adj, random_state=SEED)
            X_omics_train = pca_omics.fit_transform(X_omics_train)
            X_omics_val = pca_omics.transform(X_omics_val)
            X_omics_test = pca_omics.transform(X_omics_test)
            print(f"  Explained variance: {pca_omics.explained_variance_ratio_.sum():.4f}")
            pca_models['omics'] = pca_omics
            
            # PCA for ESM2
            pca_esm2_dim_adj = min(pca_esm2_dim, esm2_dim, len(idx_train))
            print(f"\nESM2 PCA: {esm2_dim} -> {pca_esm2_dim_adj}")
            pca_esm2 = PCA(n_components=pca_esm2_dim_adj, random_state=SEED)
            X_esm2_train = pca_esm2.fit_transform(X_esm2_train)
            X_esm2_val = pca_esm2.transform(X_esm2_val)
            X_esm2_test = pca_esm2.transform(X_esm2_test)
            print(f"  Explained variance: {pca_esm2.explained_variance_ratio_.sum():.4f}")
            pca_models['esm2'] = pca_esm2
            
            # PCA for Enformer
            pca_enformer_dim_adj = min(pca_enformer_dim, enformer_dim, len(idx_train))
            print(f"\nEnformer PCA: {enformer_dim} -> {pca_enformer_dim_adj}")
            pca_enformer = PCA(n_components=pca_enformer_dim_adj, random_state=SEED)
            X_enformer_train = pca_enformer.fit_transform(X_enformer_train)
            X_enformer_val = pca_enformer.transform(X_enformer_val)
            X_enformer_test = pca_enformer.transform(X_enformer_test)
            print(f"  Explained variance: {pca_enformer.explained_variance_ratio_.sum():.4f}")
            pca_models['enformer'] = pca_enformer
            
            # PCA for Orthrus
            pca_orthrus_dim_adj = min(pca_orthrus_dim, orthrus_dim, len(idx_train))
            print(f"\nOrthrus PCA: {orthrus_dim} -> {pca_orthrus_dim_adj}")
            pca_orthrus = PCA(n_components=pca_orthrus_dim_adj, random_state=SEED)
            X_orthrus_train = pca_orthrus.fit_transform(X_orthrus_train)
            X_orthrus_val = pca_orthrus.transform(X_orthrus_val)
            X_orthrus_test = pca_orthrus.transform(X_orthrus_test)
            print(f"  Explained variance: {pca_orthrus.explained_variance_ratio_.sum():.4f}")
            pca_models['orthrus'] = pca_orthrus
            
            # Concatenate
            X_train = np.concatenate([X_omics_train, X_esm2_train, X_enformer_train, X_orthrus_train], axis=1)
            X_val = np.concatenate([X_omics_val, X_esm2_val, X_enformer_val, X_orthrus_val], axis=1)
            X_test = np.concatenate([X_omics_test, X_esm2_test, X_enformer_test, X_orthrus_test], axis=1)
            
            final_dim = X_train.shape[1]
            print(f"\nFinal concatenated dimension: {pca_omics_dim_adj} + {pca_esm2_dim_adj} + {pca_enformer_dim_adj} + {pca_orthrus_dim_adj} = {final_dim}")
            
        elif pca_mode == "late_pca":
            print("\n" + "=" * 70)
            print("Applying LATE PCA (concat first, then PCA)")
            print("=" * 70)
            
            # Concatenate first
            X_train_concat = np.concatenate([X_omics_train, X_esm2_train, X_enformer_train, X_orthrus_train], axis=1)
            X_val_concat = np.concatenate([X_omics_val, X_esm2_val, X_enformer_val, X_orthrus_val], axis=1)
            X_test_concat = np.concatenate([X_omics_test, X_esm2_test, X_enformer_test, X_orthrus_test], axis=1)
            
            concat_dim = X_train_concat.shape[1]
            print(f"Concatenated dimension: {concat_dim}")
            
            # Apply PCA
            pca_late_dim_adj = min(pca_late_dim, concat_dim, len(idx_train))
            print(f"Late PCA: {concat_dim} -> {pca_late_dim_adj}")
            pca_late = PCA(n_components=pca_late_dim_adj, random_state=SEED)
            X_train = pca_late.fit_transform(X_train_concat)
            X_val = pca_late.transform(X_val_concat)
            X_test = pca_late.transform(X_test_concat)
            print(f"  Explained variance: {pca_late.explained_variance_ratio_.sum():.4f}")
            pca_models['late'] = pca_late
            
            print(f"\nFinal dimension: {X_train.shape[1]}")
            
        else:  # no PCA
            print("\n" + "=" * 70)
            print("Applying CONCAT (no PCA)")
            print("=" * 70)
            
            X_train = np.concatenate([X_omics_train, X_esm2_train, X_enformer_train, X_orthrus_train], axis=1)
            X_val = np.concatenate([X_omics_val, X_esm2_val, X_enformer_val, X_orthrus_val], axis=1)
            X_test = np.concatenate([X_omics_test, X_esm2_test, X_enformer_test, X_orthrus_test], axis=1)
            
            print(f"Final dimension: {X_train.shape[1]}")
    
    print(f"\nFinal data shapes: X_train={X_train.shape}, Y_train={Y_train.shape}")
    
    return X_train.astype(np.float32), Y_train, X_val.astype(np.float32), Y_val, X_test.astype(np.float32), Y_test, pca_models


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
    
    parser = argparse.ArgumentParser(description='Train HPO classifier with 4 embeddings')
    parser.add_argument('--fusion_mode', type=str, default='concat', choices=['concat', 'add'],
                        help='Fusion mode (default: concat)')
    parser.add_argument('--pca_mode', type=str, default='none', choices=['none', 'early_pca', 'late_pca'],
                        help='PCA mode for concat (default: none)')
    parser.add_argument('--esm2_model', type=str, default='650M', choices=['8M', '35M', '650M'],
                        help='ESM2 model size (default: 650M)')
    parser.add_argument('--orthrus_track', type=str, default='4track', choices=['4track', '6track'],
                        help='Orthrus track type (default: 4track)')
    parser.add_argument('--pca_omics', type=int, default=PCA_OMICS_DIM,
                        help=f'PCA dim for Omics (early_pca, default: {PCA_OMICS_DIM})')
    parser.add_argument('--pca_esm2', type=int, default=PCA_ESM2_DIM,
                        help=f'PCA dim for ESM2 (early_pca, default: {PCA_ESM2_DIM})')
    parser.add_argument('--pca_enformer', type=int, default=PCA_ENFORMER_DIM,
                        help=f'PCA dim for Enformer (early_pca, default: {PCA_ENFORMER_DIM})')
    parser.add_argument('--pca_orthrus', type=int, default=PCA_ORTHRUS_DIM,
                        help=f'PCA dim for Orthrus (early_pca, default: {PCA_ORTHRUS_DIM})')
    parser.add_argument('--pca_late', type=int, default=PCA_LATE_DIM,
                        help=f'PCA dim for late_pca (default: {PCA_LATE_DIM})')
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
    print("Training HPO Classifier with 4 Embeddings")
    print("Omics + ESM2 + Enformer + Orthrus")
    print("=" * 70)
    print(f"Fusion Mode: {args.fusion_mode}")
    print(f"PCA Mode: {args.pca_mode}")
    print(f"ESM2 Model: {args.esm2_model}")
    print(f"Orthrus Track: {args.orthrus_track}")
    if args.pca_mode == "early_pca":
        print(f"PCA Dims: Omics={args.pca_omics}, ESM2={args.pca_esm2}, Enformer={args.pca_enformer}, Orthrus={args.pca_orthrus}")
    elif args.pca_mode == "late_pca":
        print(f"Late PCA Dim: {args.pca_late}")
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
    X_train, Y_train, X_val, Y_val, X_test, Y_test, pca_models = prepare_data_4_embeddings(
        esm2_model=args.esm2_model,
        orthrus_track=args.orthrus_track,
        fusion_mode=args.fusion_mode,
        pca_mode=args.pca_mode,
        pca_omics_dim=args.pca_omics,
        pca_esm2_dim=args.pca_esm2,
        pca_enformer_dim=args.pca_enformer,
        pca_orthrus_dim=args.pca_orthrus,
        pca_late_dim=args.pca_late
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
    if args.pca_mode == "early_pca":
        prefix = f"4emb_{args.fusion_mode}_early_{args.pca_omics}_{args.pca_esm2}_{args.pca_enformer}_{args.pca_orthrus}_{args.esm2_model}_{args.orthrus_track}"
    elif args.pca_mode == "late_pca":
        prefix = f"4emb_{args.fusion_mode}_late_{args.pca_late}_{args.esm2_model}_{args.orthrus_track}"
    else:
        prefix = f"4emb_{args.fusion_mode}_{args.esm2_model}_{args.orthrus_track}"
    
    model_save_path = os.path.join(res_dir, f'best_{prefix}.pth')
    pca_save_path = os.path.join(res_dir, f'pca_{prefix}.pkl') if pca_models else None
    
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
            if pca_models:
                joblib.dump(pca_models, pca_save_path)
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
    if pca_save_path:
        print(f"  PCA models saved to: {pca_save_path}")
    print(f"  Plots saved to: {res_dir}/")
