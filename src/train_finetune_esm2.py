"""
End-to-end training with ESM2 fine-tuning + Omics embeddings.
This script jointly trains the ESM2 protein language model with omics embeddings
and a downstream HPO classifier.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer, AutoModel
import mygene
import matplotlib.pyplot as plt
from model import HPOClassifier
# ============== Set Seed ==============
import random
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============== Hyperparameters ==============
HIDDEN_DIM = 1000
BATCH_SIZE = 8  # Smaller batch size for fine-tuning ESM2 (reduce if OOM)
GRAD_ACCUM_STEPS = 4  # Gradient accumulation to simulate larger batch
EPOCHS = 100
LEARNING_RATE = 1e-4  # Lower LR for fine-tuning
ESM2_LR = 1e-6  # Even lower LR for ESM2 backbone
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 512  # Reduced from 1024 to save memory during fine-tuning
FREEZE_ESM2_EPOCHS = 2  # Freeze ESM2 for first N epochs

# ============== Model Definition ==============
class ESM2OmicsHPOClassifier(nn.Module):
    """
    End-to-end model combining ESM2 (fine-tunable) + Omics embeddings (fixed) for HPO classification.
    """
    def __init__(self, num_hpo_classes, omics_dim, hidden_size=512, 
                 esm2_model_name="facebook/esm2_t6_8M_UR50D",
                 freeze_esm2=False, fusion_mode="concat"):
        super().__init__()
        
        # Load ESM2 model
        self.esm2 = AutoModel.from_pretrained(esm2_model_name)
        self.esm2_dim = self.esm2.config.hidden_size  # 320 for 8M model
        self.omics_dim = omics_dim
        self.fusion_mode = fusion_mode
        
        # Optionally freeze ESM2 parameters
        if freeze_esm2:
            self.freeze_esm2()
        
        # Calculate input dimension based on fusion mode
        if fusion_mode == "concat":
            classifier_input_dim = self.esm2_dim + omics_dim
        elif fusion_mode == "add":
            # Need projection layers to match dimensions
            self.max_dim = max(self.esm2_dim, omics_dim)
            self.esm2_proj = nn.Linear(self.esm2_dim, self.max_dim) if self.esm2_dim != self.max_dim else nn.Identity()
            self.omics_proj = nn.Linear(omics_dim, self.max_dim) if omics_dim != self.max_dim else nn.Identity()
            classifier_input_dim = self.max_dim
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        
        # Downstream classifier
        self.classifier = HPOClassifier(
            input_size=classifier_input_dim,
            hidden_size=hidden_size,
            out_size=num_hpo_classes
        )
        
        print(f"Model: ESM2 dim={self.esm2_dim}, Omics dim={omics_dim}, "
              f"Fusion={fusion_mode}, Classifier input={classifier_input_dim}")
    
    def freeze_esm2(self):
        """Freeze ESM2 parameters"""
        for param in self.esm2.parameters():
            param.requires_grad = False
        print("ESM2 parameters frozen")
    
    def unfreeze_esm2(self):
        """Unfreeze ESM2 parameters for fine-tuning"""
        for param in self.esm2.parameters():
            param.requires_grad = True
        # Enable gradient checkpointing to reduce memory usage
        self.esm2.gradient_checkpointing_enable()
        print("ESM2 parameters unfrozen for fine-tuning (gradient checkpointing enabled)")
    
    def forward(self, input_ids, attention_mask, omics_emb):
        """
        Forward pass through ESM2, fuse with omics, then classify.
        
        Args:
            input_ids: Tokenized protein sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            omics_emb: Omics embeddings [batch_size, omics_dim]
        
        Returns:
            logits: HPO classification logits [batch_size, num_hpo_classes]
        """
        # Get ESM2 embeddings
        esm2_output = self.esm2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = esm2_output.last_hidden_state  # [batch, seq_len, hidden]
        
        # Mean pooling over sequence (excluding padding)
        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = last_hidden_state * mask
        sum_hidden = masked_hidden.sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        esm2_pooled = sum_hidden / count  # [batch, esm2_dim]
        
        # Fuse ESM2 and Omics embeddings
        if self.fusion_mode == "concat":
            fused = torch.cat([esm2_pooled, omics_emb], dim=1)
        elif self.fusion_mode == "add":
            esm2_proj = self.esm2_proj(esm2_pooled)
            omics_proj = self.omics_proj(omics_emb)
            fused = esm2_proj + omics_proj
        
        # Classify
        logits = self.classifier(fused)
        return logits


# ============== Dataset ==============
class ProteinOmicsHPODataset(Dataset):
    """
    Dataset that loads protein sequences, omics embeddings, and HPO labels.
    Tokenizes sequences on-the-fly.
    """
    def __init__(self, gene_symbols, gene2seq, gene2omics, gene2hpos, hpo2idx, tokenizer, max_len=1024):
        self.gene_symbols = gene_symbols
        self.gene2seq = gene2seq
        self.gene2omics = gene2omics
        self.gene2hpos = gene2hpos
        self.hpo2idx = hpo2idx
        self.num_hpo = len(hpo2idx)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.gene_symbols)
    
    def __getitem__(self, idx):
        gene = self.gene_symbols[idx]
        seq = self.gene2seq[gene]
        omics = self.gene2omics[gene]
        
        # Tokenize protein sequence
        encoded = self.tokenizer(
            seq, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_len,
            padding="max_length"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # Omics embedding
        omics_emb = torch.from_numpy(omics).float()
        
        # Create HPO label vector
        y = torch.zeros(self.num_hpo, dtype=torch.float32)
        for hpo in self.gene2hpos[gene]:
            if hpo in self.hpo2idx:
                y[self.hpo2idx[hpo]] = 1.0
        
        return input_ids, attention_mask, omics_emb, y


# ============== Data Loading ==============
def load_omics_embeddings():
    """Load omics embeddings and map to gene symbols"""
    data_path = "../dataset/"
    omic_path = "omics_embeddings"
    emb_path = data_path + omic_path + "/Supplementary_Table_S3_OMICS_EMB.tsv"
    
    df_emb = pd.read_csv(emb_path, sep="\t")
    gene_col = "gene_id"
    df_emb = df_emb.set_index(gene_col)
    
    print(f"Loaded omics embeddings: {df_emb.shape[0]} genes, dim={df_emb.shape[1]}")
    
    # Map Ensembl ID to gene symbol
    mg = mygene.MyGeneInfo()
    ensg_ids = df_emb.index.tolist()
    
    print("Querying mygene for gene symbols...")
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
    
    print(f"After mapping: {df_emb.shape[0]} genes with symbols")
    
    # Convert to dict
    gene2omics = {gene: df_emb.loc[gene].values.astype(np.float32) for gene in df_emb.index}
    omics_dim = df_emb.shape[1]
    
    return gene2omics, omics_dim


def load_data_for_finetuning():
    """
    Load protein sequences, omics embeddings, and HPO labels.
    Returns train/val/test splits with all required data.
    """
    # Load gene2seq from pickle (generated by extract_esm2.py)
    gene2seq_path = "../dataset/gene2seq.pkl"
    if not os.path.exists(gene2seq_path):
        raise FileNotFoundError(
            f"{gene2seq_path} not found. Run extract_esm2.py first to generate it."
        )
    
    with open(gene2seq_path, "rb") as f:
        gene2seq = pickle.load(f)
    print(f"Loaded {len(gene2seq)} protein sequences")
    
    # Load omics embeddings
    gene2omics, omics_dim = load_omics_embeddings()
    
    # Load HPO labels
    hpo_path = "../dataset/genes_to_phenotype.txt"
    df_hpo = pd.read_csv(hpo_path, sep="\t", comment="#")
    df_hpo = df_hpo[["gene_symbol", "hpo_id"]].dropna()
    gene2hpos = df_hpo.groupby("gene_symbol")["hpo_id"].apply(list).to_dict()
    print(f"Loaded HPO labels for {len(gene2hpos)} genes")
    
    # Find common genes (must have sequence, omics, AND HPO labels)
    common_genes = sorted(
        set(gene2seq.keys()) & set(gene2omics.keys()) & set(gene2hpos.keys())
    )
    print(f"Genes with sequence + omics + HPO labels: {len(common_genes)}")
    
    # Build HPO vocabulary
    all_hpos = sorted({h for g in common_genes for h in gene2hpos[g]})
    hpo2idx = {h: i for i, h in enumerate(all_hpos)}
    print(f"Number of HPO terms: {len(hpo2idx)}")
    
    # Split data
    train_genes, temp_genes = train_test_split(common_genes, test_size=0.3, random_state=42)
    val_genes, test_genes = train_test_split(temp_genes, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_genes)}, Val: {len(val_genes)}, Test: {len(test_genes)}")
    
    return train_genes, val_genes, test_genes, gene2seq, gene2omics, gene2hpos, hpo2idx, omics_dim


# ============== Training Functions ==============
def evaluate(model, dataloader, criterion, device, return_predictions=False):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for input_ids, attention_mask, omics_emb, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            omics_emb = omics_emb.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, attention_mask, omics_emb)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_outputs.append(probs)
            all_targets.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    try:
        auprc = average_precision_score(all_targets, all_outputs, average='micro')
    except ValueError:
        auprc = np.nan
    
    if return_predictions:
        return avg_loss, auprc, all_outputs, all_targets
    return avg_loss, auprc


def plot_per_class_auprc_boxplot(y_true, y_pred, save_path='auprc_boxplot_finetune.png'):
    """
    Plot box plot of per-class AUPRC scores.
    
    Args:
        y_true: Ground truth labels [n_samples, n_classes]
        y_pred: Predicted probabilities [n_samples, n_classes]
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    num_classes = y_true.shape[1]
    class_counts = y_true.sum(axis=0)
    
    # Calculate per-class AUPRC
    all_auprcs = []
    for class_idx in range(num_classes):
        if class_counts[class_idx] > 0:  # Only for classes with positive samples
            try:
                auprc = average_precision_score(y_true[:, class_idx], y_pred[:, class_idx])
                all_auprcs.append(auprc)
            except:
                pass
    
    all_auprcs = np.array(all_auprcs)
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot(all_auprcs, patch_artist=True, vert=True)
    
    # Styling
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    
    ax.set_ylabel('AUPRC', fontsize=12)
    ax.set_title('Distribution of Per-Class AUPRC\n(ESM2 + Omics Fine-tuned Model)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(['All HPO Classes'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
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
    
    return all_auprcs


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_accum_steps=1):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for batch_idx, (input_ids, attention_mask, omics_emb, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        omics_emb = omics_emb.to(device)
        labels = labels.to(device)
        
        outputs = model(input_ids, attention_mask, omics_emb)
        loss = criterion(outputs, labels)
        loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
        loss.backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            # Clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_loss += loss.item() * grad_accum_steps  # Unscale for logging
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item() * grad_accum_steps:.4f}")
    
    return total_loss / len(dataloader)


# ============== Main ==============
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # Load data
    (train_genes, val_genes, test_genes, 
     gene2seq, gene2omics, gene2hpos, hpo2idx, omics_dim) = load_data_for_finetuning()
    num_hpo = len(hpo2idx)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    # Create datasets
    train_dataset = ProteinOmicsHPODataset(
        train_genes, gene2seq, gene2omics, gene2hpos, hpo2idx, tokenizer, MAX_SEQ_LEN
    )
    val_dataset = ProteinOmicsHPODataset(
        val_genes, gene2seq, gene2omics, gene2hpos, hpo2idx, tokenizer, MAX_SEQ_LEN
    )
    test_dataset = ProteinOmicsHPODataset(
        test_genes, gene2seq, gene2omics, gene2hpos, hpo2idx, tokenizer, MAX_SEQ_LEN
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\nDataLoader batches: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    
    # Create model (start with frozen ESM2)
    model = ESM2OmicsHPOClassifier(
        num_hpo_classes=num_hpo,
        omics_dim=omics_dim,
        hidden_size=HIDDEN_DIM,
        esm2_model_name="facebook/esm2_t6_8M_UR50D",
        freeze_esm2=True,  # Start frozen
        fusion_mode="add"  # or "add"
    )
    model.to(DEVICE)
    print(f"\nModel created. ESM2 dim: {model.esm2_dim}, Omics dim: {omics_dim}, HPO classes: {num_hpo}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    
    # Use different learning rates for ESM2 and classifier
    optimizer = torch.optim.AdamW([
        {"params": model.esm2.parameters(), "lr": ESM2_LR},
        {"params": model.classifier.parameters(), "lr": LEARNING_RATE}
    ])
    
    # Training loop
    best_val_auprc = 0.0
    print("\n" + "="*50)
    print("Starting Training (ESM2 + Omics -> HPO)")
    print("="*50)
    
    losses = []  # 用于保存每个epoch的train loss
    val_losses = []  # 用于保存每个epoch的val loss
    for epoch in range(EPOCHS):
        # Unfreeze ESM2 after warm-up epochs
        if epoch == FREEZE_ESM2_EPOCHS:
            model.unfreeze_esm2()
            print(f"Epoch {epoch+1}: ESM2 unfrozen for fine-tuning")
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, GRAD_ACCUM_STEPS)
        losses.append(train_loss)
        # Evaluate
        val_loss, val_auprc = evaluate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUPRC: {val_auprc:.4f}")
        # Save best model
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), "best_esm2_omics_hpo_finetuned.pth")
            print(f"  -> New best model saved! AUPRC: {val_auprc:.4f}")

    plt.figure()
    plt.plot(range(1, EPOCHS+1), losses, marker='o', label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, marker='s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot_esm2_omics_finetune.png', dpi=300)
    plt.show()
    print("Loss plot saved to: loss_plot_esm2_omics_finetune.png")

    # Final evaluation
    print("\n" + "="*50)
    print("Final Test Evaluation")
    print("="*50)
    
    model.load_state_dict(torch.load("best_esm2_omics_hpo_finetuned.pth"))
    test_loss, test_auprc, test_outputs, test_targets = evaluate(
        model, test_loader, criterion, DEVICE, return_predictions=True
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUPRC (micro): {test_auprc:.4f}")
    print("="*50)
    
    # Plot per-class AUPRC box plot
    print("\nGenerating AUPRC box plot...")
    per_class_auprcs = plot_per_class_auprc_boxplot(
        test_targets, test_outputs, 
        save_path='auprc_boxplot_esm2_omics_finetune.png'
    )
    print(f"Per-class AUPRC - Mean: {np.mean(per_class_auprcs):.4f}, "
          f"Median: {np.median(per_class_auprcs):.4f}")
    print("="*50)
