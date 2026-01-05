import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mygene

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
def align_genes(df_emb, gene2hpos):
    genes_common = sorted(set(df_emb.index) & set(gene2hpos.index))
    print("Genes with both embedding and HPO labels:", len(genes_common))

    hpo_terms = sorted({h for g in genes_common for h in gene2hpos[g]})
    hpo2idx = {h: i for i, h in enumerate(hpo_terms)}
    num_hpo = len(hpo_terms)
    print("Number of HPO terms:", num_hpo)
    return genes_common, hpo2idx, num_hpo

# ---------------- 5. Generate X, Y matrices ----------------
def split_data():
    df_emb = load_omics_embedding()
    df_emb = map_ensembl_to_symbol(df_emb)
    gene2hpos = load_hpo_labels()
    genes_common, hpo2idx, num_hpo = align_genes(df_emb, gene2hpos)
    X_list, Y_list = [], []

    for g in genes_common:
        x = df_emb.loc[g].values.astype(np.float32)  
        y = np.zeros(num_hpo, dtype=np.float32)
        for h in gene2hpos[g]:
            idx = hpo2idx[h]
            y[idx] = 1.0
        X_list.append(x)
        Y_list.append(y)

    X = np.stack(X_list)   # [N, 256]
    Y = np.stack(Y_list)   # [N, num_hpo]

    print("X shape:", X.shape, "Y shape:", Y.shape)
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
    
    top_classes = np.argsort(class_counts)[-top_n:]
    
    auprcs = []
    counts = []
    
    for class_idx in top_classes:
        if class_counts[class_idx] > 0:
            try:
                auprc = average_precision_score(y_true[:, class_idx], y_pred[:, class_idx])
            except:
                auprc = 0.0
            auprcs.append(auprc)
            counts.append(int(class_counts[class_idx]))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    x_pos = np.arange(len(auprcs))
    ax1.bar(x_pos, auprcs, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_ylabel('AUPRC', fontsize=12)
    ax1.set_title(f'Per-Class AUPRC (Top {top_n} Classes by Sample Count)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'HPO {i}' for i in top_classes], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(x_pos, counts, color='coral', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('HPO Class', fontsize=12)
    ax2.set_ylabel('Number of Positive Samples', fontsize=12)
    ax2.set_title(f'Sample Distribution (Top {top_n} Classes)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'HPO {i}' for i in top_classes], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class performance plot saved to: {save_path}")

