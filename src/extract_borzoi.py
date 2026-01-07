import torch
import pandas as pd
import numpy as np
from borzoi_pytorch import Borzoi
from pyfaidx import Fasta
from tqdm import tqdm
from pybiomart import Server


GENOME_PATH = "../data/hg38.fa"
DEVICE = "cpu"
SEQ_LEN = 524288


print(f"Loading Borzoi on {DEVICE}...")
model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE)
model.eval()


class StopExecution(Exception):
    """Custom error to stop the model early and save CPU time"""
    pass

captured_embedding = None

def transformer_hook(module, input, output):
    global captured_embedding
    # output is the tensor we want (Batch, Length, Dim)
    captured_embedding = output.detach().cpu()
    # Raise error to stop the model from running the rest (Upsampling/Heads)
    raise StopExecution()


handle = model.transformer.register_forward_hook(transformer_hook)

# prepare genes

print("Fetching gene locations...")
server = Server(host='http://www.ensembl.org')
dataset = server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl']
gene_locs = dataset.query(attributes=['ensembl_gene_id', 'chromosome_name', 'transcription_start_site'])
gene_locs.columns = ['ensembl_id', 'chrom', 'tss']
valid_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']
gene_locs = gene_locs[gene_locs['chrom'].isin(valid_chroms)]

# TEST MODE: First 10 genes. 
batch_genes = gene_locs.iloc[:10] 

# inference
fasta = Fasta(GENOME_PATH)
embeddings = []
valid_gene_ids = []

print(f"Starting inference on {len(batch_genes)} genes...")

def get_one_hot(chrom, tss):
    chrom_str = f"chr{chrom}"
    start = tss - (SEQ_LEN // 2)
    end = start + SEQ_LEN
    
    if chrom_str not in fasta.keys(): return None
    if start < 0 or end > len(fasta[chrom_str]): return None
    
    seq = fasta[chrom_str][start:end].seq.upper()
    if len(seq) != SEQ_LEN: return None

    # Manual One-Hot (A:0, C:1, G:2, T:3)
    seq_code = np.zeros((4, SEQ_LEN), dtype=np.float32)
    for i, char in enumerate(seq):
        if char == 'A': seq_code[0, i] = 1
        elif char == 'C': seq_code[1, i] = 1
        elif char == 'G': seq_code[2, i] = 1
        elif char == 'T': seq_code[3, i] = 1
    return seq_code

for _, row in tqdm(batch_genes.iterrows(), total=len(batch_genes)):
    try:
        x_arr = get_one_hot(row['chrom'], row['tss'])
        if x_arr is None: continue
        
        x_tensor = torch.tensor(x_arr).unsqueeze(0).to(DEVICE)
        
        # Run model - it will crash on purpose inside the hook
        try:
            model(x_tensor)
        except StopExecution:
            pass # We caught the embedding, so we are good!
        except Exception as e:
            print(f"Real error on {row['ensembl_id']}: {e}")
            continue
            
        # Process the captured embedding
        if captured_embedding is not None:
            # Shape is likely (Batch, Bins, Dim) -> (1, ~2048, 1536/1920)
            # Take center bins
            mid = captured_embedding.shape[1] // 2
            # Average 3 middle bins
            emb = captured_embedding[:, mid-1:mid+2, :].mean(dim=1)
            
            embeddings.append(emb.squeeze().numpy())
            valid_gene_ids.append(row['ensembl_id'])
            
            # Clear memory
            captured_embedding = None

    except Exception as e:
        print(f"Skipping {row['ensembl_id']}: {e}")
        continue


if embeddings:
    X_borzoi = pd.DataFrame(np.vstack(embeddings), index=valid_gene_ids)
    print(f"Success! Shape: {X_borzoi.shape}")
    X_borzoi.to_csv("borzoi_embeddings.csv")
else:
    print("No embeddings generated.")


handle.remove()