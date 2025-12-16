from Bio import SeqIO
import re
import pandas as pd
import pickle
import numpy as np
import os

import torch
from transformers import AutoTokenizer, AutoModel

def gene_to_rna_from_cdna(cdna_fasta, genes):
    gene2rna = {}

    for rec in SeqIO.parse(cdna_fasta, "fasta"):
        desc = rec.description
        if "gene_symbol:" not in desc:
            continue

        gene = desc.split("gene_symbol:")[1].split()[0]
        if gene not in genes:
            continue

        seq = str(rec.seq).upper()

        seq = seq.replace("T", "U")

        if gene not in gene2rna or len(seq) > len(gene2rna[gene]):
            gene2rna[gene] = seq

    return gene2rna


def generate_orthrus_embeddings(gene2rna):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model =AutoModel.from_pretrained(
        "quietflamingo/orthrus-base-4-track",
        trust_remote_code=True
    ).to(device)
    model.eval()

    os.makedirs("../dataset/orthrus_embeddings", exist_ok=True)
    gene2emb = {}

    for i, (gene, seq) in enumerate(gene2rna.items()):
        if (i + 1) % 100 == 0 or (i + 1) == len(gene2rna):
            print(f"Processing sequence {i + 1}/{len(gene2rna)}: {gene}")

        try:
            seq_ohe = model.seq_to_oh(seq.upper()).to(device)
            
            model_input_tt = seq_ohe.unsqueeze(0)
            
            lengths = torch.Tensor([model_input_tt.shape[1]]).to(device)

            with torch.no_grad():
                embedding = model.representation(
                    model_input_tt,
                    lengths,
                    channel_last=True
                )
            
            gene2emb[gene] = embedding.squeeze(0).cpu().numpy()

        except Exception as e:
            print(f"Error generating embedding for gene {gene}: {e}")

    print("Embedding generation complete.")
    
    return gene2emb

if __name__ == "__main__":
    # ---------------- Load gene symbols from HPO labels ----------------
    hpo_path = "../dataset/genes_to_phenotype.txt"
    df_hpo = pd.read_csv(hpo_path, sep="\t", comment="#")
    print("HPO columns:", df_hpo.columns)
    genes = set(df_hpo["gene_symbol"].unique())
    gene2rna = gene_to_rna_from_cdna(
        cdna_fasta="../dataset/Homo_sapiens.GRCh38.cdna.all.fa",
        genes=genes
    )
    print(gene2rna["AARS1"])
    print("Genes with RNA sequence:", len(gene2rna))
    print("Example AARS1 RNA length:", len(gene2rna["AARS1"]))
    with open("../dataset/gene2rna.pkl", "wb") as f:
        pickle.dump(gene2rna, f)
    
    # ---------------- Generate embeddings ----------------
    generate_orthrus_embeddings(gene2rna)