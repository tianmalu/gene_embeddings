import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mygene
import os

# ---------- 1. Read Omics embedding ----------
def load_omics_embedding(dataset_path="/p/scratch/hai_1134/reimt/dataset"):
    data_path = dataset_path
    omic_path = "/omics_embeddings"
    emb_path = data_path + omic_path + "/Supplementary_Table_S3_OMICS_EMB.tsv"

    df_emb = pd.read_csv(emb_path, sep="\t")

    gene_col = "gene_id"
    df_emb = df_emb.set_index(gene_col)

    X_emb = df_emb.values.astype(np.float32)   
    emb_genes = df_emb.index.tolist()
    print("Embedding genes:", len(emb_genes), "dim =", X_emb.shape[1])
    return df_emb

# ---------- 1.5 Read ESM2 embeddings ----------
def load_esm2_embeddings(esm2_model="8M"):
    """
    Load ESM2 protein embeddings from .npy files.
    
    Args:
        esm2_model: Which ESM2 model embeddings to load. Options:
                    - "8M": esm2_t6_8M_UR50D (dim=320)
                    - "650M": esm2_t33_650M_UR50D (dim=1280)
    """
    if esm2_model == "8M":
        esm2_dir = "../dataset/esm2_embeddings"
    elif esm2_model == "650M":
        esm2_dir = "../dataset/esm2_embeddings_650M"
    elif esm2_model == "35M":
        esm2_dir = "../dataset/esm2_embeddings_35M"
    else:
        raise ValueError(f"Unknown esm2_model: {esm2_model}. Use '8M' or '650M'.")
    
    gene2esm2 = {}
    
    if not os.path.exists(esm2_dir):
        print(f"Warning: ESM2 embedding directory not found: {esm2_dir}")
        return gene2esm2
    
    for fname in os.listdir(esm2_dir):
        if fname.endswith(".npy"):
            gene_symbol = fname.replace(".npy", "")
            emb_path = os.path.join(esm2_dir, fname)
            gene2esm2[gene_symbol] = np.load(emb_path)
    
    print(f"Loaded ESM2 ({esm2_model}) embeddings for {len(gene2esm2)} genes")
    if gene2esm2:
        sample_emb = next(iter(gene2esm2.values()))
        print(f"ESM2 embedding dim = {sample_emb.shape[0]}")
    return gene2esm2

# ---------------- 2. Ensembl ID -> gene symbol ----------------
def map_ensembl_to_symbol(df_emb):
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

    print("After mapping and deduplication:", df_emb.shape[0], "genes with symbols")
    return df_emb

# ---------------- 3. Read HPO gene labels ----------------
def load_hpo_labels(dataset_path=None):
    hpo_path = f"{dataset_path}/genes_to_phenotype.txt"  

    df_hpo = pd.read_csv(hpo_path, sep="\t", comment="#")
    print("HPO columns:", df_hpo.columns)

    df_hpo = df_hpo[["gene_symbol", "hpo_id"]].dropna()

    # gene -> [hpo1, hpo2, ...]
    gene2hpos = df_hpo.groupby("gene_symbol")["hpo_id"].apply(list)
    print("Genes with HPO labels:", len(gene2hpos))
    return gene2hpos

# ---------------- 4. Align genes with both embedding and HPO labels ----------------
def align_genes(df_emb, gene2hpos, gene2esm2=None):
    """Align genes that have all required data (omics embedding, HPO labels, and optionally ESM2 embedding)"""
    genes_with_omics_and_hpo = set(df_emb.index) & set(gene2hpos.index)
    
    if gene2esm2 is not None:
        # Only keep genes that also have ESM2 embeddings
        genes_common = sorted(genes_with_omics_and_hpo & set(gene2esm2.keys()))
        print(f"Genes with omics embedding and HPO labels: {len(genes_with_omics_and_hpo)}")
        print(f"Genes with all three (omics + HPO + ESM2): {len(genes_common)}")
    else:
        genes_common = sorted(genes_with_omics_and_hpo)
        print("Genes with both embedding and HPO labels:", len(genes_common))

    hpo_terms = sorted({h for g in genes_common for h in gene2hpos[g]})
    hpo2idx = {h: i for i, h in enumerate(hpo_terms)}
    num_hpo = len(hpo_terms)
    print("Number of HPO terms:", num_hpo)
    return genes_common, hpo2idx, num_hpo

# ---------------- 5. Generate X, Y matrices ----------------
def split_data(use_esm2=True, fusion_mode="concat", esm2_model="8M"):
    """
    Load embeddings and split data for training.
    
    Args:
        use_esm2: If True, combine ESM2 embeddings with omics embeddings.
                  If False, use only omics embeddings (original behavior).
        fusion_mode: How to combine omics and ESM2 embeddings. Options:
                     - "concat": Concatenate embeddings (default)
                     - "add": Element-wise addition (with padding if dims differ)
        esm2_model: Which ESM2 model embeddings to use. Options:
                    - "8M": esm2_t6_8M_UR50D (dim=320)
                    - "650M": esm2_t33_650M_UR50D (dim=1280)
    """
    df_emb = load_omics_embedding()
    df_emb = map_ensembl_to_symbol(df_emb)
    gene2hpos = load_hpo_labels()
    
    # Load ESM2 embeddings if requested
    gene2esm2 = None
    if use_esm2:
        gene2esm2 = load_esm2_embeddings(esm2_model=esm2_model)
    
    genes_common, hpo2idx, num_hpo = align_genes(df_emb, gene2hpos, gene2esm2)
    X_list, Y_list = [], []
    
    # Get dimensions for padding if using add mode
    omics_dim = None
    esm2_dim = None
    if use_esm2 and gene2esm2 is not None and fusion_mode == "add":
        sample_gene = genes_common[0]
        omics_dim = df_emb.loc[sample_gene].values.shape[0]
        esm2_dim = gene2esm2[sample_gene].shape[0]
        max_dim = max(omics_dim, esm2_dim)
        print(f"Fusion mode: {fusion_mode}")
        print(f"  Omics dim: {omics_dim}, ESM2 dim: {esm2_dim}, Output dim: {max_dim}")

    for g in genes_common:
        # Get omics embedding
        x_omics = df_emb.loc[g].values.astype(np.float32)
        
        # Combine with ESM2 embedding if available
        if use_esm2 and gene2esm2 is not None:
            x_esm2 = gene2esm2[g].astype(np.float32)
            
            if fusion_mode == "concat":
                x = np.concatenate([x_omics, x_esm2])
            elif fusion_mode == "add":
                # Pad shorter embedding to match longer one
                max_dim = max(len(x_omics), len(x_esm2))
                x_omics_padded = np.pad(x_omics, (0, max_dim - len(x_omics)), mode='constant')
                x_esm2_padded = np.pad(x_esm2, (0, max_dim - len(x_esm2)), mode='constant')
                x = x_omics_padded + x_esm2_padded
            else:
                raise ValueError(f"Unknown fusion_mode: {fusion_mode}. Use 'concat' or 'add'.")
        else:
            x = x_omics
        
        y = np.zeros(num_hpo, dtype=np.float32)
        for h in gene2hpos[g]:
            idx = hpo2idx[h]
            y[idx] = 1.0
        X_list.append(x)
        Y_list.append(y)

    X = np.stack(X_list)   # [N, 256 + 320] if concat, [N, max(256, 320)] if add
    Y = np.stack(Y_list)   # [N, num_hpo]

    print("X shape:", X.shape, "Y shape:", Y.shape)
    if use_esm2:
        if fusion_mode == "concat":
            print(f"  -> Omics dim + ESM2 dim = {X.shape[1]} (concatenated)")
        else:
            print(f"  -> Output dim = {X.shape[1]} (element-wise add)")
# ---------------- 6. Split train/val/test ----------------
    N = X.shape[0]
    indices = np.arange(N)

    idx_train, idx_tmp = train_test_split(indices, test_size=0.3, random_state=42)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=42)

    def take(arr, idx):
        return arr[idx]

    X_train, Y_train = take(X, idx_train), take(Y, idx_train)
    X_val,   Y_val   = take(X, idx_val),   take(Y, idx_val)
    X_test,  Y_test  = take(X, idx_test),  take(Y, idx_test)

    print("train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# ---------------- 7. Visualization functions ----------------
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import label_binarize

def plot_training_curves(train_losses, val_losses, train_auprcs, val_auprcs, save_path='training_curves.png'):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_auprcs, 'b-o', label='Train AUPRC', linewidth=2, markersize=4)
    ax2.plot(epochs, val_auprcs, 'r-s', label='Val AUPRC', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('AUPRC', fontsize=12)
    ax2.set_title('Training and Validation AUPRC', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to: {save_path}")

def plot_roc_curves(y_true, y_pred, num_classes_to_plot=5, save_path='roc_curves.png'):

    plt.figure(figsize=(10, 8))
    
    class_counts = y_true.sum(axis=0)
    top_classes = np.argsort(class_counts)[-num_classes_to_plot:]
    
    for i, class_idx in enumerate(top_classes):
        if class_counts[class_idx] > 0:
            fpr, tpr, _ = roc_curve(y_true[:, class_idx], y_pred[:, class_idx])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'HPO {class_idx} (AUC = {roc_auc:.3f}, n={int(class_counts[class_idx])})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Top HPO Classes', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to: {save_path}")

def plot_prediction_distribution(y_true, y_pred, save_path='prediction_distribution.png'):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    positive_preds = y_pred[y_true == 1]
    ax1.hist(positive_preds, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Prediction Distribution for Positive Samples', fontsize=14, fontweight='bold')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    negative_preds = y_pred[y_true == 0]
    ax2.hist(negative_preds, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Prediction Distribution for Negative Samples', fontsize=14, fontweight='bold')
    ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Threshold=0.5')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction distribution plot saved to: {save_path}")

def plot_top_k_accuracy(y_true, y_pred, k_values=[1, 3, 5, 10, 20, 50], save_path='top_k_accuracy.png'):

    n_samples = y_true.shape[0]
    accuracies = []
    
    for k in k_values:
        correct = 0
        for i in range(n_samples):
            top_k_idx = np.argsort(y_pred[i])[-k:]
            if np.any(y_true[i][top_k_idx] == 1):
                correct += 1
        accuracy = correct / n_samples
        accuracies.append(accuracy)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('K', fontsize=12)
    plt.ylabel('Top-K Accuracy', fontsize=12)
    plt.title('Top-K Accuracy vs K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for k, acc in zip(k_values, accuracies):
        plt.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top-K accuracy plot saved to: {save_path}")

def plot_per_class_performance(y_true, y_pred, top_n=20, save_path='per_class_performance.png'):
  
    from sklearn.metrics import average_precision_score
    
    num_classes = y_true.shape[1]
    class_counts = y_true.sum(axis=0)
    
    all_auprcs = []
    for class_idx in range(num_classes):
        if class_counts[class_idx] > 0:
            try:
                auprc = average_precision_score(y_true[:, class_idx], y_pred[:, class_idx])
                all_auprcs.append(auprc)
            except:
                pass
    
    all_auprcs = np.array(all_auprcs)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot(all_auprcs, patch_artist=True, vert=True)
    
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    
    ax.set_ylabel('AUPRC', fontsize=12)
    ax.set_title('Distribution of Per-Class AUPRC', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['All HPO Classes'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 
    stats_text = (f'Mean: {np.mean(all_auprcs):.4f}\n'
                  f'Median: {np.median(all_auprcs):.4f}\n'
                  f'Std: {np.std(all_auprcs):.4f}\n'
                  f'Min: {np.min(all_auprcs):.4f}\n'
                  f'Max: {np.max(all_auprcs):.4f}\n'
                  f'N Classes: {len(all_auprcs)}')
    ax.text(1.15, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUPRC box plot saved to: {save_path}")

