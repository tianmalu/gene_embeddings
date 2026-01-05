import os
import re
import pandas as pd
import numpy as np
import torch
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from utils import load_omics_embedding, map_ensembl_to_symbol, load_hpo_labels

# Settings
OUT_DIR = "/p/scratch/hai_1134/reimt/dataset/esm_c_embeddings"
UNIPROT_FASTA = "/p/scratch/hai_1134/reimt/dataset/uniprot_human.fasta"  # Download from UniProt
# ESM-3 C-series public checkpoints
#   "esmc_300m"  (smaller)
#   "esmc_600m"  (larger)
MODEL_NAME = "esmc_600m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Truncate very long proteins for safety; adjust if needed.
MAX_SEQ_LEN = 1024
MIN_GENES_PER_HPO = 20


def gene_symbol_to_protein_fasta(fasta_path: str, gene_symbols: set[str]) -> dict[str, str]:
    """Load protein sequences from UniProt FASTA, keyed by gene symbol (GN=).
    Returns the longest sequence if multiple entries exist for the same gene."""
    gene2seq = {}
    gn_re = re.compile(r"\bGN=([A-Za-z0-9\-]+)\b")

    for record in SeqIO.parse(fasta_path, "fasta"):
        desc = record.description
        m = gn_re.search(desc)
        if not m:
            continue

        gene = m.group(1)
        if gene not in gene_symbols:
            continue

        seq = str(record.seq)

        # Take the longest sequence if multiple entries exist
        if gene not in gene2seq or len(seq) > len(gene2seq[gene]):
            gene2seq[gene] = seq

    return gene2seq


def load_model():
    print(f"Loading {MODEL_NAME} to {DEVICE} ...")
    client = ESMC.from_pretrained(MODEL_NAME).to(DEVICE)
    client.eval()
    return client


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load omics genes and map to symbols
    df_emb = load_omics_embedding()
    df_emb = map_ensembl_to_symbol(df_emb)
    omics_syms = set(df_emb.index.tolist())
    print("Omics genes (symbols):", len(omics_syms))

    # Filter HPO genes (>=MIN_GENES_PER_HPO per term)
    gene2hpos = load_hpo_labels()
    hpo_counts = gene2hpos.explode().value_counts()
    keep_terms = set(hpo_counts[hpo_counts >= MIN_GENES_PER_HPO].index.tolist())
    hpo_syms = {g for g, terms in gene2hpos.items() if any(t in keep_terms for t in terms)}
    print("HPO genes (filtered):", len(hpo_syms))

    # Intersect
    genes_to_embed = omics_syms & hpo_syms
    print("Genes to embed (omics ∩ HPO filtered):", len(genes_to_embed))

    # Load protein sequences from FASTA
    if not os.path.exists(UNIPROT_FASTA):
        raise FileNotFoundError(f"UniProt FASTA not found at {UNIPROT_FASTA}. Download from https://www.uniprot.org/help/downloads")
    
    gene2seq = gene_symbol_to_protein_fasta(UNIPROT_FASTA, genes_to_embed)
    print(f"Loaded {len(gene2seq)} protein sequences from FASTA")

    # Load model
    client = load_model()

    # Generate embeddings
    gene2emb = {}
    summary = []
    status_counts = {"ok": 0, "no_sequence": 0, "embed_failed": 0}
    first_shape_logged = False

    for i, gene in enumerate(genes_to_embed, 1):
        seq = gene2seq.get(gene)
        if not seq:
            summary.append((gene, 0, "no_sequence"))
            status_counts["no_sequence"] += 1
            continue

        try:
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[:MAX_SEQ_LEN]

            protein = ESMProtein(sequence=seq)

            with torch.no_grad():
                protein_tensor = client.encode(protein).to(DEVICE)
                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True),
                )

            embeddings = logits_output.embeddings  # shape: (1, L, D)
            emb = embeddings.squeeze(0).mean(dim=0).cpu().numpy()
            gene2emb[gene] = emb
            summary.append((gene, emb.shape[0], "ok"))
            status_counts["ok"] += 1
            
            if not first_shape_logged:
                print(f"First embedding shape: {emb.shape}")
                first_shape_logged = True

        except Exception as e:
            summary.append((gene, 0, "embed_failed"))
            status_counts["embed_failed"] += 1
            print(f"Error generating embedding for {gene}: {e}")

        if i % 50 == 0:
            print(f"Processed {i}/{len(genes_to_embed)} genes", end="\r")

    # Save summary
    summary_path = os.path.join(OUT_DIR, "embedding_summary.tsv")
    pd.DataFrame(summary, columns=["gene_symbol", "emb_dim", "status"]).to_csv(
        summary_path,
        sep="\t",
        index=False,
    )
    print(f"\nDone. Saved embeddings to {OUT_DIR}. Summary: {summary_path}")
    print(
        "Status counts -> ok: {ok}, embed_failed: {embed_failed}, "
        "no_sequence: {no_sequence}".format(**status_counts)
    )


if __name__ == "__main__":
    main()
