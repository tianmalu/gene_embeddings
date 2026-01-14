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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
import mygene
import joblib
import gc
import matplotlib.pyplot as plt

from model import ImprovedHPOClassifier
from utils import plot_training_curves, plot_roc_curves, plot_prediction_distribution, plot_top_k_accuracy, plot_per_class_performance

# ==================== Configuration ====================
HIDDEN_DIM = 1000
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5  # Number of folds for cross-validation

# PCA variance ratios (float < 1.0 = variance ratio, int >= 1 = number of components)
PCA_OMICS_VARIANCE = 0.95  # Keep 95% variance
PCA_ESM2_VARIANCE = 0.95
PCA_ENFORMER_VARIANCE = 0.95
PCA_ORTHRUS_VARIANCE = 0.95
PCA_LATE_VARIANCE = 0.95  # For late PCA (after concat)

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

def create_fold_labels(genes, gene2hpos):
    """
    Create label matrix for a specific fold containing only labels present in that fold.
    
    Args:
        genes: List of gene symbols in this fold
        gene2hpos: Dictionary mapping genes to HPO terms
    
    Returns:
        Y: Label matrix (n_genes, n_labels_in_fold)
        hpo2idx: Mapping from HPO term to index in Y
        hpo_terms: List of HPO terms in this fold
    """
    # Get all HPO terms present in this fold
    hpo_terms = sorted({h for g in genes for h in gene2hpos[g]})
    hpo2idx = {h: i for i, h in enumerate(hpo_terms)}
    
    # Create label matrix
    Y = np.zeros((len(genes), len(hpo_terms)), dtype=np.float32)
    for i, g in enumerate(genes):
        for h in gene2hpos[g]:
            Y[i, hpo2idx[h]] = 1.0
    
    return Y, hpo2idx, hpo_terms


def apply_pca_to_fold(X_train, X_val, pca_mode, pca_var, embedding_dims, seed=SEED):
    """
    Apply PCA to training and validation data for a fold.
    
    Args:
        X_train: Training features
        X_val: Validation features
        pca_mode: 'none', 'early_pca', 'late_pca'
        pca_var: PCA variance ratio (float < 1.0) or n_components (int >= 1)
                 For late_pca: single value
                 For early_pca: dict with values for each embedding
        embedding_dims: Dict with dimensions of each embedding type
        seed: Random seed
    
    Returns:
        X_train_pca, X_val_pca, pca_models
    """
    pca_models = {}
    
    if pca_mode == "none":
        return X_train, X_val, pca_models
    
    elif pca_mode == "late_pca":
        # Apply PCA to concatenated features
        concat_dim = X_train.shape[1]
        
        print(f"  Late PCA: {concat_dim} -> auto (variance={pca_var})")
        pca_late = PCA(n_components=pca_var, random_state=seed)
        X_train_pca = pca_late.fit_transform(X_train)
        X_val_pca = pca_late.transform(X_val)
        print(f"    Selected {pca_late.n_components_} components")
        print(f"    Explained variance: {pca_late.explained_variance_ratio_.sum():.4f}")
        pca_models['late'] = pca_late
        
        return X_train_pca, X_val_pca, pca_models
    
    elif pca_mode == "early_pca":
        # Split concatenated features back into individual embeddings
        omics_dim = embedding_dims['omics_dim']
        esm2_dim = embedding_dims['esm2_dim']
        enformer_dim = embedding_dims['enformer_dim']
        orthrus_dim = embedding_dims['orthrus_dim']
        
        # Split train
        X_omics_train = X_train[:, :omics_dim]
        X_esm2_train = X_train[:, omics_dim:omics_dim+esm2_dim]
        X_enformer_train = X_train[:, omics_dim+esm2_dim:omics_dim+esm2_dim+enformer_dim]
        X_orthrus_train = X_train[:, omics_dim+esm2_dim+enformer_dim:]
        
        # Split val
        X_omics_val = X_val[:, :omics_dim]
        X_esm2_val = X_val[:, omics_dim:omics_dim+esm2_dim]
        X_enformer_val = X_val[:, omics_dim+esm2_dim:omics_dim+esm2_dim+enformer_dim]
        X_orthrus_val = X_val[:, omics_dim+esm2_dim+enformer_dim:]
        
        # Apply PCA to each embedding
        print(f"  Omics PCA: {omics_dim} -> auto (variance={pca_var['omics']})")
        pca_omics = PCA(n_components=pca_var['omics'], random_state=seed)
        X_omics_train = pca_omics.fit_transform(X_omics_train)
        X_omics_val = pca_omics.transform(X_omics_val)
        print(f"    Selected {pca_omics.n_components_} components, variance: {pca_omics.explained_variance_ratio_.sum():.4f}")
        pca_models['omics'] = pca_omics
        
        print(f"  ESM2 PCA: {esm2_dim} -> auto (variance={pca_var['esm2']})")
        pca_esm2 = PCA(n_components=pca_var['esm2'], random_state=seed)
        X_esm2_train = pca_esm2.fit_transform(X_esm2_train)
        X_esm2_val = pca_esm2.transform(X_esm2_val)
        print(f"    Selected {pca_esm2.n_components_} components, variance: {pca_esm2.explained_variance_ratio_.sum():.4f}")
        pca_models['esm2'] = pca_esm2
        
        print(f"  Enformer PCA: {enformer_dim} -> auto (variance={pca_var['enformer']})")
        pca_enformer = PCA(n_components=pca_var['enformer'], random_state=seed)
        X_enformer_train = pca_enformer.fit_transform(X_enformer_train)
        X_enformer_val = pca_enformer.transform(X_enformer_val)
        print(f"    Selected {pca_enformer.n_components_} components, variance: {pca_enformer.explained_variance_ratio_.sum():.4f}")
        pca_models['enformer'] = pca_enformer
        
        print(f"  Orthrus PCA: {orthrus_dim} -> auto (variance={pca_var['orthrus']})")
        pca_orthrus = PCA(n_components=pca_var['orthrus'], random_state=seed)
        X_orthrus_train = pca_orthrus.fit_transform(X_orthrus_train)
        X_orthrus_val = pca_orthrus.transform(X_orthrus_val)
        print(f"    Selected {pca_orthrus.n_components_} components, variance: {pca_orthrus.explained_variance_ratio_.sum():.4f}")
        pca_models['orthrus'] = pca_orthrus
        
        # Concatenate
        X_train_pca = np.concatenate([X_omics_train, X_esm2_train, X_enformer_train, X_orthrus_train], axis=1)
        X_val_pca = np.concatenate([X_omics_val, X_esm2_val, X_enformer_val, X_orthrus_val], axis=1)
        print(f"    Final concatenated dimension: {X_train_pca.shape[1]}")
        
        return X_train_pca, X_val_pca, pca_models
    
    return X_train, X_val, pca_models


def prepare_data_4_embeddings(esm2_model="650M", orthrus_track="4track", 
                               fusion_mode="concat", pca_mode="none",
                               pca_omics_var=PCA_OMICS_VARIANCE, pca_esm2_var=PCA_ESM2_VARIANCE,
                               pca_enformer_var=PCA_ENFORMER_VARIANCE, pca_orthrus_var=PCA_ORTHRUS_VARIANCE,
                               pca_late_var=PCA_LATE_VARIANCE):
    """
    Load 4 embeddings and prepare data for 5-fold cross-validation.
    Note: Does NOT build HPO label matrix here - labels will be created per fold.
    
    Args:
        esm2_model: ESM2 model size ('8M', '35M', '650M')
        orthrus_track: Orthrus track type ('4track' or '6track')
        fusion_mode: 'add' or 'concat'
        pca_mode: 'none', 'early_pca', 'late_pca' (only for concat mode)
        pca_*_var: PCA variance ratio (float < 1.0) or n_components (int >= 1)
    
    Returns:
        X (features), genes_common (gene list), gene2hpos (HPO labels dict)
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
    
    for i, g in enumerate(genes_common):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{N} genes...")
        
        X_omics[i] = df_omics.loc[g].values.astype(np.float32)
        X_esm2[i] = gene2esm2[g].astype(np.float32)
        X_enformer[i] = load_enformer_embedding(g)
        X_orthrus[i] = gene2orthrus[g].astype(np.float32)
    
    print(f"  Done! Collected all embeddings.")
    
    # Clear memory
    del gene2esm2, gene2orthrus
    gc.collect()
    
    # ==================== Fusion (all data) ====================
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
        
        X = pad_and_add(X_omics, X_esm2, X_enformer, X_orthrus, max_dim=max_dim)
        print(f"Final dimension: {X.shape[1]}")
        
    else:  # concat - no PCA yet (will be done per fold)
        print("\n" + "=" * 70)
        print("Applying CONCAT")
        print("=" * 70)
        
        X = np.concatenate([X_omics, X_esm2, X_enformer, X_orthrus], axis=1)
        print(f"Final dimension: {X.shape[1]}")
    
    print(f"\nFinal data shape: X={X.shape}")
    if fusion_mode == "concat":
        print(f"PCA will be applied separately for each fold during cross-validation.")
    else:
        print(f"ADD mode: No PCA will be applied.")
    
    # Return features, gene list, and HPO labels dict
    return X.astype(np.float32), genes_common, gene2hpos, {
        'omics_dim': omics_dim,
        'esm2_dim': esm2_dim,
        'enformer_dim': enformer_dim,
        'orthrus_dim': orthrus_dim,
        'fusion_mode': fusion_mode
    }


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
    parser.add_argument('--pca_omics', type=float, default=PCA_OMICS_VARIANCE,
                        help=f'PCA variance ratio for Omics (early_pca, default: {PCA_OMICS_VARIANCE})')
    parser.add_argument('--pca_esm2', type=float, default=PCA_ESM2_VARIANCE,
                        help=f'PCA variance ratio for ESM2 (early_pca, default: {PCA_ESM2_VARIANCE})')
    parser.add_argument('--pca_enformer', type=float, default=PCA_ENFORMER_VARIANCE,
                        help=f'PCA variance ratio for Enformer (early_pca, default: {PCA_ENFORMER_VARIANCE})')
    parser.add_argument('--pca_orthrus', type=float, default=PCA_ORTHRUS_VARIANCE,
                        help=f'PCA variance ratio for Orthrus (early_pca, default: {PCA_ORTHRUS_VARIANCE})')
    parser.add_argument('--pca_late', type=float, default=PCA_LATE_VARIANCE,
                        help=f'PCA variance ratio for late_pca (default: {PCA_LATE_VARIANCE})')
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
        print(f"PCA Variance Ratios: Omics={args.pca_omics}, ESM2={args.pca_esm2}, Enformer={args.pca_enformer}, Orthrus={args.pca_orthrus}")
    elif args.pca_mode == "late_pca":
        print(f"Late PCA Variance Ratio: {args.pca_late}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Hidden Dim: {args.hidden_dim}")
    print(f"Number of Folds: {N_FOLDS}")
    print(f"Device: {DEVICE}")
    print("=" * 70 + "\n")
    
    # Result directory
    res_dir = os.path.join(os.path.dirname(__file__), 'res')
    os.makedirs(res_dir, exist_ok=True)
    
    # Load and prepare data (returns features without labels)
    X, genes_common, gene2hpos, embedding_dims = prepare_data_4_embeddings(
        esm2_model=args.esm2_model,
        orthrus_track=args.orthrus_track,
        fusion_mode=args.fusion_mode,
        pca_mode=args.pca_mode,
        pca_omics_var=args.pca_omics,
        pca_esm2_var=args.pca_esm2,
        pca_enformer_var=args.pca_enformer,
        pca_orthrus_var=args.pca_orthrus,
        pca_late_var=args.pca_late
    )
    
    # Prepare PCA variance config
    if args.pca_mode == "early_pca":
        pca_var_config = {
            'omics': args.pca_omics,
            'esm2': args.pca_esm2,
            'enformer': args.pca_enformer,
            'orthrus': args.pca_orthrus
        }
    elif args.pca_mode == "late_pca":
        pca_var_config = args.pca_late
    else:
        pca_var_config = None
    
    # ==================== 5-Fold Cross-Validation ====================
    print("\n" + "=" * 70)
    print(f"Starting {N_FOLDS}-Fold Cross-Validation")
    print("=" * 70)
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    genes_array = np.array(genes_common)
    
    fold_results = []
    all_fold_outputs = []
    all_fold_targets = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print("\n" + "=" * 70)
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print("=" * 70)
        
        # Split data
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        genes_train = genes_array[train_idx].tolist()
        genes_val = genes_array[val_idx].tolist()
        
        print(f"Train samples: {len(genes_train)}, Val samples: {len(genes_val)}")
        
        # Apply PCA for this fold (only for concat mode)
        if embedding_dims['fusion_mode'] == 'concat':
            print(f"\nApplying PCA for fold {fold_idx + 1}...")
            X_train_fold, X_val_fold, pca_models_fold = apply_pca_to_fold(
                X_train_fold, X_val_fold, args.pca_mode, pca_var_config, embedding_dims, SEED
            )
        else:
            print(f"\nADD mode: Skipping PCA for fold {fold_idx + 1}")
            pca_models_fold = {}
        
        # Create fold-specific labels (only labels present in training set)
        print(f"\nCreating fold-specific labels...")
        Y_train_fold, hpo2idx_train, hpo_terms_train = create_fold_labels(genes_train, gene2hpos)
        print(f"  Training set: {len(hpo_terms_train)} HPO terms")
        
        # Create validation labels using TRAINING set's HPO terms
        # (validation samples may have labels not seen in training - those are ignored)
        Y_val_fold = np.zeros((len(genes_val), len(hpo_terms_train)), dtype=np.float32)
        for i, g in enumerate(genes_val):
            for h in gene2hpos[g]:
                if h in hpo2idx_train:  # Only include if seen in training
                    Y_val_fold[i, hpo2idx_train[h]] = 1.0
        
        print(f"  Created label matrices: Train={Y_train_fold.shape}, Val={Y_val_fold.shape}")
        
        # Create datasets and dataloaders
        train_dataset = GeneHPODataset(X_train_fold, Y_train_fold)
        val_dataset = GeneHPODataset(X_val_fold, Y_val_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Initialize model for this fold
        in_dim = X_train_fold.shape[1]
        out_dim = Y_train_fold.shape[1]
        
        print(f"\nInitializing model: Input dim={in_dim}, Hidden dim={args.hidden_dim}, Output dim={out_dim}")
        model = ImprovedHPOClassifier(
            input_size=in_dim,
            out_size=out_dim,
            hidden_size=args.hidden_dim,
        )
        model.to(DEVICE)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Training loop for this fold
        print(f"\nTraining fold {fold_idx + 1}...")
        best_val_auprc = 0.0
        best_model_state = None
        
        train_losses = []
        val_losses = []
        train_auprcs = []
        val_auprcs = []
        
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
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}: "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train AUPRC: {train_auprc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val AUPRC: {val_auprc:.4f}")
            
            # Save best model for this fold
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model_state = model.state_dict().copy()
        
        # Load best model for this fold
        model.load_state_dict(best_model_state)
        
        # Final evaluation on this fold
        val_loss, val_auprc, val_outputs, val_targets = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"\n  Fold {fold_idx + 1} Best Val AUPRC: {val_auprc:.4f}")
        
        # Store results
        fold_results.append({
            'fold': fold_idx + 1,
            'val_auprc': val_auprc,
            'val_loss': val_loss,
            'n_train': len(genes_train),
            'n_val': len(genes_val),
            'n_labels': len(hpo_terms_train)
        })
        
        all_fold_outputs.append(val_outputs)
        all_fold_targets.append(val_targets)
        
        # Save fold-specific model and results
        if args.fusion_mode == "add":
            prefix = f"4emb_add_{args.esm2_model}_{args.orthrus_track}"
        elif args.pca_mode == "early_pca":
            prefix = f"4emb_{args.fusion_mode}_early_{args.pca_omics}_{args.pca_esm2}_{args.pca_enformer}_{args.pca_orthrus}_{args.esm2_model}_{args.orthrus_track}"
        elif args.pca_mode == "late_pca":
            prefix = f"4emb_{args.fusion_mode}_late_{args.pca_late}_{args.esm2_model}_{args.orthrus_track}"
        else:
            prefix = f"4emb_{args.fusion_mode}_{args.esm2_model}_{args.orthrus_track}"
        
        fold_prefix = f"{prefix}_fold{fold_idx+1}"
        model_save_path = os.path.join(res_dir, f'best_{fold_prefix}.pth')
        torch.save(best_model_state, model_save_path)
        
        if pca_models_fold:
            pca_save_path = os.path.join(res_dir, f'pca_{fold_prefix}.pkl')
            joblib.dump(pca_models_fold, pca_save_path)
        
        # Save training curves for this fold
        plot_training_curves(train_losses, val_losses, train_auprcs, val_auprcs,
                            save_path=os.path.join(res_dir, f'{fold_prefix}_training_curves.png'))
        
        # Clear memory
        del model, optimizer, train_dataset, val_dataset, train_loader, val_loader
        del X_train_fold, X_val_fold, Y_train_fold, Y_val_fold
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ==================== Cross-Validation Summary ====================
    print("\n" + "=" * 70)
    print("5-Fold Cross-Validation Summary")
    print("=" * 70)
    
    val_auprcs = [r['val_auprc'] for r in fold_results]
    mean_auprc = np.mean(val_auprcs)
    std_auprc = np.std(val_auprcs)
    
    print(f"\nResults per fold:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Val AUPRC = {r['val_auprc']:.4f}, "
              f"Train samples = {r['n_train']}, Val samples = {r['n_val']}, "
              f"Labels = {r['n_labels']}")
    
    print(f"\nOverall Performance:")
    print(f"  Mean Val AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}")
    print(f"  Min Val AUPRC: {min(val_auprcs):.4f}")
    print(f"  Max Val AUPRC: {max(val_auprcs):.4f}")
    
    # Generate boxplot for cross-validation results
    print(f"\nGenerating cross-validation boxplot...")
    boxplot_path = os.path.join(res_dir, f'{prefix}_cv_boxplot.png')
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(val_auprcs, patch_artist=True, 
                     boxprops=dict(facecolor='lightblue', color='blue'),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='blue'),
                     capprops=dict(color='blue'))
    plt.scatter([1]*len(val_auprcs), val_auprcs, color='darkblue', s=50, alpha=0.6, zorder=3)
    plt.title(f'5-Fold Cross-Validation AUPRC Scores\nMean: {mean_auprc:.4f} ± {std_auprc:.4f}', fontsize=12, fontweight='bold')
    plt.ylabel('AUPRC', fontsize=11)
    plt.xticks([1], ['5-Fold CV'], fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([min(val_auprcs) - 0.01, max(val_auprcs) + 0.01])
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Boxplot saved to: {boxplot_path}")
    
    # Save summary
    summary_path = os.path.join(res_dir, f'{prefix}_cv_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("5-Fold Cross-Validation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Fusion mode: {args.fusion_mode}\n")
        f.write(f"  PCA mode: {args.pca_mode}\n")
        f.write(f"  ESM2 model: {args.esm2_model}\n")
        f.write(f"  Orthrus track: {args.orthrus_track}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Hidden dim: {args.hidden_dim}\n\n")
        f.write("Results per fold:\n")
        for r in fold_results:
            f.write(f"  Fold {r['fold']}: Val AUPRC = {r['val_auprc']:.4f}, "
                   f"Train samples = {r['n_train']}, Val samples = {r['n_val']}, "
                   f"Labels = {r['n_labels']}\n")
        f.write(f"\nOverall Performance:\n")
        f.write(f"  Mean Val AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}\n")
        f.write(f"  Min Val AUPRC: {min(val_auprcs):.4f}\n")
        f.write(f"  Max Val AUPRC: {max(val_auprcs):.4f}\n")
    
    print(f"\nCross-validation completed!")
    print(f"  Models saved to: {res_dir}/")
    print(f"  Summary saved to: {summary_path}")
