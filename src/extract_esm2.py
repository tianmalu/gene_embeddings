from Bio import SeqIO
import re
import pandas as pd
import pickle
import numpy as np
import os

import torch
from transformers import AutoTokenizer, AutoModel

def gene_symbol_to_fasta(fasta_path, genes):
    """
    Inputs:
      fasta_path: UniProt FASTA
      genes: set/list of gene_symbol, e.g. {"AARS1", "NAT2"}

    Outputs:
      gene_symbol -> protein sequence (string)
    """
    gene2seq = {}
    gn_re = re.compile(r"\bGN=([A-Za-z0-9\-]+)\b")

    for record in SeqIO.parse(fasta_path, "fasta"):
        desc = record.description
        m = gn_re.search(desc)
        if not m:
            continue

        gene = m.group(1)
        if gene not in genes:
            continue

        seq = str(record.seq)

        # One gene may have multiple proteins: take the longest
        if gene not in gene2seq or len(seq) > len(gene2seq[gene]):
            gene2seq[gene] = seq

    return gene2seq



def generate_esm2_embeddings(gene2seq, output_dir="../dataset/esm2_embeddings", model_name="facebook/esm2_t6_8M_UR50D"):
    """
    Generate ESM2 embeddings for protein sequences.
    
    Args:
        gene2seq: dict mapping gene symbols to protein sequences
        output_dir: directory to save embeddings
        model_name: ESM2 model to use. Options:
            - "facebook/esm2_t6_8M_UR50D" (8M params, dim=320)
            - "facebook/esm2_t33_650M_UR50D" (650M params, dim=1280)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)     
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    gene2emb = {}

    for i, (gene, seq) in enumerate(gene2seq.items()):
        if i % 100 == 0:
            print(f"Processing {i}/{len(gene2seq)}: {gene}")

        # Tokenize
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        
        last_hidden_state = outputs.last_hidden_state[0]
        emb = last_hidden_state[1:-1].mean(dim=0).cpu().numpy()

        gene2emb[gene] = emb
        np.save(os.path.join(output_dir, f"{gene}.npy"), emb)

    print("Finished generation.")
    if 'emb' in locals():
        print("Embedding dim:", emb.shape[0])
    
    return gene2emb



if __name__ == "__main__":
    # ---------------- Load gene symbols from HPO labels ----------------
    hpo_path = "../dataset/genes_to_phenotype.txt"
    df_hpo = pd.read_csv(hpo_path, sep="\t", comment="#")
    print("HPO columns:", df_hpo.columns)
    genes = set(df_hpo["gene_symbol"].unique())

    gene2seq = gene_symbol_to_fasta(
        fasta_path="../dataset/uniprot_sprot.fasta",
        genes=genes
    )
    print("Sample sequence (AARS1):", gene2seq.get("AARS1", "Not found")[:50] + "...")
    print("Genes with protein sequence:", len(gene2seq))
    if "AARS1" in gene2seq:
        print("AARS1 sequence length:", len(gene2seq["AARS1"]))
    
    with open("../dataset/gene2seq.pkl", "wb") as f:
        pickle.dump(gene2seq, f)
    
    # ---------------- Generate ESM2 embeddings ----------------
    generate_esm2_embeddings(
        gene2seq, 
        output_dir="../dataset/esm2_embeddings_35M",
        model_name="facebook/esm2_t12_35M_UR50D"  # 35M model, dim=320
    )
