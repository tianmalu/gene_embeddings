import os
import pickle
import re
import numpy as np
import torch
from Bio import SeqIO
from transformers import AutoModel
from utils import load_omics_embedding, map_ensembl_to_symbol, load_hpo_labels
try:
    from genomekit import Genome
except ImportError:
    Genome = None

# Settings
OUT_DIR = "dataset/orthrus_embeddings"
CDNA_FASTA = "dataset/ensembl_human_cds.fasta"  # Download from Ensembl
TRACK_TYPE = "4"  # choose "4" or "6"; six-track requires GenomeKit assets
MIN_GENES_PER_HPO = 20
GENOME_NAME = "gencode.v29"  # required for six-track via GenomeKit

def gene_symbol_to_cdna_fasta(fasta_path: str, gene_symbols: set[str]) -> dict[str, str]:
    """Load cDNA sequences from Ensembl FASTA, keyed by gene symbol.
    Ensembl FASTA headers typically contain gene info; extract symbol from description.
    Returns the longest sequence if multiple entries exist for the same gene."""
    gene2seq = {}
    
    # Build a lookup for faster matching
    symbols_lower = {s.lower(): s for s in gene_symbols}

    for record in SeqIO.parse(fasta_path, "fasta"):
        desc = record.description
        # Ensembl cDNA headers: "ENST... chromosome:...:... gene:ENSG...:SYMBOL:..."
        # Look for gene symbol in description
        gene = None
        
        # Try to parse header to find symbol
        parts = desc.split()
        for part in parts:
            if ":" in part:
                subparts = part.split(":")
                for sp in subparts:
                    sp_lower = sp.lower()
                    if sp_lower in symbols_lower:
                        gene = symbols_lower[sp_lower]
                        break
            if gene:
                break

        if not gene:
            continue

        seq = str(record.seq).replace("T", "U").upper()  # Convert to RNA

        # Take the longest sequence if multiple entries exist
        if gene not in gene2seq or len(seq) > len(gene2seq[gene]):
            gene2seq[gene] = seq

    return gene2seq


def filter_hpo_genes(min_genes_per_term: int = 20) -> set[str]:
    """Return gene symbols present in HPO terms with at least min_genes_per_term genes."""
    gene2hpos = load_hpo_labels()  # Series: gene_symbol -> list of hpo_id
    hpo_counts = gene2hpos.explode().value_counts()
    keep_terms = set(hpo_counts[hpo_counts >= min_genes_per_term].index.tolist())
    return {g for g, terms in gene2hpos.items() if any(t in keep_terms for t in terms)}


def six_track_encoding_for_gene(genome: Genome, gene: str, max_len: int | None = None):
    if Genome is None:
        raise RuntimeError("GenomeKit not installed; please install genomekit>=6.0.0 for six-track")

    txs = genome.transcripts_by_gene_name(gene)
    if not txs:
        return None

    # pick longest coding transcript
    def tx_len(tx):
        cds_len = getattr(tx, "cds_length", None)
        return cds_len if cds_len is not None else getattr(tx, "length", 0)

    tx = max(txs, key=tx_len)

    if hasattr(tx, "six_track_encoding"):
        arr = tx.six_track_encoding()  # expected shape (6, L)
    else:
        raise RuntimeError("GenomeKit transcript missing six_track_encoding; update genomekit")

    if arr is None:
        return None

    if max_len is not None:
        arr = arr[:, :max_len]

    # return shape (L, 6)
    return arr.T


def generate_orthrus_embeddings(gene2rna, track_type: str = "4"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_id = "quietflamingo/orthrus-base-4-track" if track_type == "4" else "quietflamingo/orthrus-base-6-track"
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True
    ).to(device)
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    gene2emb = {}

    genome = None
    if track_type == "6":
        if Genome is None:
            print("GenomeKit not installed; skipping six-track embeddings")
            return gene2emb
        try:
            genome = Genome(GENOME_NAME)
            print(f"Loaded GenomeKit genome: {GENOME_NAME}")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not load GenomeKit genome '{GENOME_NAME}': {exc}. Skipping six-track embeddings.")
            return gene2emb

    for i, (gene, seq) in enumerate(gene2rna.items()):
        if (i + 1) % 100 == 0 or (i + 1) == len(gene2rna):
            print(f"Processing sequence {i + 1}/{len(gene2rna)}: {gene}")

        try:
            if track_type == "4":
                seq_ohe = model.seq_to_oh(seq.upper()).to(device)  # (L,4)
                model_input_tt = seq_ohe.unsqueeze(0)  # (1,L,4)
                lengths = torch.Tensor([model_input_tt.shape[1]]).to(device)
            else:
                six = six_track_encoding_for_gene(genome, gene)
                if six is None:
                    raise RuntimeError(f"No six-track encoding for {gene}")
                six_tt = torch.tensor(six, dtype=torch.float32, device=device)  # (L,6)
                model_input_tt = six_tt.unsqueeze(0)  # (1,L,6)
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
    # ---------------- Load omics genes and map to symbols ----------------
    df_emb = load_omics_embedding()  # index: gene_id
    df_emb = map_ensembl_to_symbol(df_emb)  # reindexed to gene_symbol
    omics_syms = set(df_emb.index.tolist())
    print("Omics genes (symbols):", len(omics_syms))

    # ---------------- Filter HPO genes (>=20 per term) ----------------
    hpo_syms = filter_hpo_genes(min_genes_per_term=MIN_GENES_PER_HPO)
    print("HPO genes (filtered):", len(hpo_syms))

    # ---------------- Intersect ----------------
    genes_to_embed = omics_syms & hpo_syms
    print("Genes to embed (omics ∩ HPO filtered):", len(genes_to_embed))

    # ---------------- Load RNA sequences from FASTA ----------------
    if not os.path.exists(CDNA_FASTA):
        raise FileNotFoundError(f"cDNA FASTA not found at {CDNA_FASTA}. Download from Ensembl or see README for instructions.")
    
    gene2rna = gene_symbol_to_cdna_fasta(CDNA_FASTA, genes_to_embed)
    print("Genes with cDNA sequence:", len(gene2rna))
    if gene2rna:
        first_gene = next(iter(gene2rna))
        print(f"Example {first_gene} RNA length:", len(gene2rna[first_gene]))

    with open("dataset/gene2rna.pkl", "wb") as f:
        pickle.dump(gene2rna, f)
    
    # ---------------- Generate embeddings ----------------
    generate_orthrus_embeddings(gene2rna, track_type=TRACK_TYPE)