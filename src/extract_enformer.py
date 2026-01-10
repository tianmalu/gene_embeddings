from Bio import SeqIO
import re
import pandas as pd
import pickle
import numpy as np
import os
from pyfaidx import Fasta

import torch
from enformer_pytorch import from_pretrained

WINDOW = 196_608
HALF = WINDOW // 2

# ---------------- 1) HPO -> genes ----------------
def load_genes_from_hpo(hpo_path: str) -> list[str]:
    df = pd.read_csv(hpo_path, sep="\t", comment="#")
    genes = sorted(set(df["gene_symbol"].dropna().unique()))
    print("HPO genes:", len(genes))
    return genes

# ---------------- 2) GTF -> gene_symbol -> (chrom, tss, strand) ----------------
def load_gene_tss_from_gtf(gtf_path: str) -> dict[str, tuple[str, int, str]]:
    gene2tss = {}
    gene_name_re = re.compile(r'gene_name "([^"]+)"')

    with open(gtf_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = line.rstrip("\n").split("\t")
            if feature != "gene":
                continue

            m = gene_name_re.search(attrs)
            if not m:
                continue
            gene = m.group(1)

            start = int(start); end = int(end)
            tss = start if strand == "+" else end

            if gene not in gene2tss:
                gene2tss[gene] = (chrom, tss, strand)
            else:
                # baseline: '+' -> min(start), '-' -> max(end)
                if strand == "+" and tss < gene2tss[gene][1]:
                    gene2tss[gene] = (chrom, tss, strand)
                if strand == "-" and tss > gene2tss[gene][1]:
                    gene2tss[gene] = (chrom, tss, strand)

    print("Genes with TSS:", len(gene2tss))
    return gene2tss

# ---------------- 3) FASTA window extraction ----------------
def fetch_window(genome: Fasta, chrom: str, tss_1based: int) -> str:
    tss0 = tss_1based - 1
    start = tss0 - HALF
    end   = tss0 + HALF

    chrom_len = len(genome[chrom])
    pad_left = max(0, -start)
    pad_right = max(0, end - chrom_len)
    start = max(0, start)
    end = min(chrom_len, end)

    seq = str(genome[chrom][start:end]).upper()
    if pad_left:
        seq = "N" * pad_left + seq
    if pad_right:
        seq = seq + "N" * pad_right

    if len(seq) != WINDOW:
        raise ValueError(f"Window length mismatch: {len(seq)}")
    return seq

def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(comp)[::-1]

# ---------------- 4) DNA -> one-hot (196608, 4) ----------------
def dna_to_onehot(seq: str) -> np.ndarray:
    mapping = {
        "A": [1,0,0,0],
        "C": [0,1,0,0],
        "G": [0,0,1,0],
        "T": [0,0,0,1],
        "N": [0,0,0,0],
    }
    return np.array([mapping.get(b, [0,0,0,0]) for b in seq], dtype=np.float32)

# ---------------- 5) Load Enformer weights from HF ----------------
def generate_enformer_embeddings(gene2tss: dict, genes: list, genome: Fasta, out_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Enformer model to {device}...")
    
    model = from_pretrained('EleutherAI/enformer-official-rough')
    model.to(device)
    model.eval()

    print(f"Starting inference on {len(genes)} genes...")
    count = 0
    skipped = 0

    for gene in genes:
        if gene not in gene2tss:
            continue
        
        chrom, tss, strand = gene2tss[gene]
        
        if chrom not in genome:
            if f"chr{chrom}" in genome:
                chrom = f"chr{chrom}"
            else:
                skipped += 1
                continue

        try:
            seq = fetch_window(genome, chrom, tss)
            
            if strand == "-":
                seq = reverse_complement(seq)
            
            one_hot = dna_to_onehot(seq)
        except Exception as e:
            print(f"Error processing sequence for {gene}: {e}")
            skipped += 1
            continue

        input_tensor = torch.from_numpy(one_hot).unsqueeze(0).to(device)

        with torch.no_grad():
            # Use return_embeddings=True to get the actual embeddings
            # Returns tuple: (predictions_dict, embeddings)
            output = model(input_tensor, return_embeddings=True)
            
            # output is a tuple: (predictions, embeddings)
            # embeddings shape: (batch, 896, 3072) - 896 bins x 3072 embedding dim
            embeddings = output[1].cpu().numpy()[0]  # (896, 3072)
            
            # Mean pooling across sequence bins to get (3072,) embedding
            gene_embedding = embeddings.mean(axis=0)  # (3072,)

        save_path = os.path.join(out_dir, f"{gene}.npy")
        np.save(save_path, gene_embedding.astype(np.float16))

        count += 1
        if count % 10 == 0:
            print(f"Processed {count} genes (Skipped {skipped})...", end="\r")

    print(f"\nCompleted. Processed: {count}, Skipped: {skipped}. Results saved to {out_dir}")

    

if __name__ == "__main__":
    # ---------------- Load gene symbols from HPO labels ----------------
    hpo_path = "../dataset/genes_to_phenotype.txt"
    gtf_path = "../dataset/Homo_sapiens.GRCh38.115.gtf"
    genome_fa = "../dataset/Homo_sapiens.GRCh38.dna.primary_assembly.fa"  

    out_dir = "../dataset/enformer_embeddings_hf"
    os.makedirs(out_dir, exist_ok=True)

    genes = load_genes_from_hpo(hpo_path)
    print("Loaded genes:", len(genes))
    gene2tss = load_gene_tss_from_gtf(gtf_path)
    print("Loaded genes and TSS information")
    genome = Fasta(genome_fa)
    print("Loaded genome:", len(genome_fa))

    # ---------------- Generate embeddings ----------------
    generate_enformer_embeddings(gene2tss, genes, genome, out_dir)

