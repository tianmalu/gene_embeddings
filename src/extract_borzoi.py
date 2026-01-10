import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch
from pybiomart import Server
from pyfaidx import Fasta
from tqdm import tqdm
from borzoi_pytorch import Borzoi


BASE_DIR = "/content/drive/MyDrive/systems_genetics"
GENOME_PATH = os.path.join(BASE_DIR, "hg38.fa")
GENE_LIST_PATH = os.path.join(BASE_DIR, "common_genes.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "borzoi_embeddings.pkl")


SEQ_LEN = 524288
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1  # Keep 1 to prevent OOM on Colab T4

print(f"Running on: {DEVICE}")


def get_canonical_gene_locations(gene_list_path):
    print("Loading gene list...")
    try:
        # load ensemble IDs from HPO dataset
        my_genes = pd.read_csv(gene_list_path, header=None)[0].tolist()
        print(f"✅ Loaded {len(my_genes)} genes of interest.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {gene_list_path}. Check your Drive path.")

    print("Fetching Canonical TSS from BioMart...")
    server = Server(host='http://www.ensembl.org')
    dataset = server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl']

    
    gene_locs = dataset.query(attributes=[
        'ensembl_gene_id',
        'chromosome_name',
        'transcription_start_site',
        'transcript_is_canonical'
    ])
    gene_locs.columns = ['ensembl_id', 'chrom', 'tss', 'is_canonical']

    # keep only requested genes
    gene_locs = gene_locs[gene_locs['ensembl_id'].isin(my_genes)]

    
    valid_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']
    gene_locs = gene_locs[gene_locs['chrom'].isin(valid_chroms)]


    
    canonical = gene_locs[gene_locs['is_canonical'] == 1].copy()
    others = gene_locs[gene_locs['is_canonical'] != 1].copy()

    canonical = canonical.drop_duplicates(subset=['ensembl_id'])

    found_ids = set(canonical['ensembl_id'])
    missing_ids = set(my_genes) - found_ids 
    
    fallbacks = others[others['ensembl_id'].isin(missing_ids)].copy()
    fallbacks = fallbacks.drop_duplicates(subset=['ensembl_id'])


    final_df = pd.concat([canonical, fallbacks])
    
    print(f"Locations prepared: {len(canonical)} canonical + {len(fallbacks)} fallbacks.")
    print(f"Total Unique Genes: {len(final_df)}")
    
    return final_df


print("Loading Borzoi Model...")
model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE)
model.eval()

class StopExecution(Exception):
    """Custom error to stop model early"""
    pass

captured_embedding = None

def transformer_hook(module, input, output):
    global captured_embedding
    captured_embedding = output.detach().cpu()
    raise StopExecution()


handle = model.transformer.register_forward_hook(transformer_hook)


def run_inference():
    
    df_genes = get_canonical_gene_locations(GENE_LIST_PATH)
    
    
    if not os.path.exists(GENOME_PATH):
        raise FileNotFoundError(f"Genome file not found at {GENOME_PATH}. Please download hg38.fa.")
    
    fasta = Fasta(GENOME_PATH)
    
    # dict to store results
    gene2emb = {}
    
    print("Starting Inference...")
    
    # one-hot
    def get_one_hot(chrom, tss):
        chrom_str = f"chr{chrom}"
        start = tss - (SEQ_LEN // 2)
        end = start + SEQ_LEN
        
        if chrom_str not in fasta.keys(): return None
    
        if start < 0 or end > len(fasta[chrom_str]): return None
        
        seq = fasta[chrom_str][start:end].seq.upper()
        if len(seq) != SEQ_LEN: return None
        
        
        arr = np.zeros((4, SEQ_LEN), dtype=np.float32)
        for i, char in enumerate(seq):
            if char == 'A': arr[0, i] = 1
            elif char == 'C': arr[1, i] = 1
            elif char == 'G': arr[2, i] = 1
            elif char == 'T': arr[3, i] = 1
        return arr

    
    global captured_embedding
    
    for i, row in tqdm(df_genes.iterrows(), total=len(df_genes)):
        gene_id = row['ensembl_id']
        
        try:
            
            x_arr = get_one_hot(row['chrom'], row['tss'])
            if x_arr is None:
                continue
            
            x_tensor = torch.tensor(x_arr).unsqueeze(0).to(DEVICE)
            
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(): # Save VRAM
                    try:
                        model(x_tensor)
                    except StopExecution:
                        pass # Hook triggered successfully
            
            # Process Output
            if captured_embedding is not None:
                mid = captured_embedding.shape[1] // 2
                emb = captured_embedding[:, mid-1:mid+2, :].mean(dim=1)
                
                gene2emb[gene_id] = emb.squeeze().numpy()
                
                captured_embedding = None
            
            if i % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"Error processing {gene_id}: {e}")
            torch.cuda.empty_cache()
            continue

    return gene2emb


if __name__ == "__main__":
    embeddings_dict = run_inference()
    
    if embeddings_dict:
        print(f"Generated {len(embeddings_dict)} embeddings.")
        print(f"Saving to pickle: {OUTPUT_PATH}")
        
        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump(embeddings_dict, f)
            
        print("Done.")
    else:
        print("No embeddings were generated. Check your paths and gene list.")

    # Cleanup hook
    handle.remove()