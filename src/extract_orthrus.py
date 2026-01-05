import argparse
import os
import pickle
import re
import numpy as np
import torch
from Bio import SeqIO
from transformers import AutoModel
from utils import load_omics_embedding, map_ensembl_to_symbol, load_hpo_labels
# GenomeKit installs as `genome_kit` (module name), but some docs refer to the
# distribution name `genomekit`. Accept both for robustness.
try:
    from genome_kit import Genome  # type: ignore
except Exception:  # noqa: BLE001
    try:
        from genomekit import Genome  # type: ignore
    except Exception:  # noqa: BLE001
        Genome = None

# Settings
DEFAULT_DATASET_DIR = "/p/scratch/hai_1134/reimt/dataset"
OUT_DIR = os.path.join(DEFAULT_DATASET_DIR, "orthrus_embeddings")
CDNA_FASTA = os.path.join(DEFAULT_DATASET_DIR, "ensembl_human_cds.fasta")  # Download from Ensembl
TRACK_TYPE = "4"  # choose "4" or "6"; six-track requires GenomeKit assets
MIN_GENES_PER_HPO = 20
GENOME_NAME = "gencode.v29"  # required for six-track via GenomeKit


def _seq_to_oh_dna(seq: str) -> np.ndarray:
    oh = np.zeros((len(seq), 4), dtype=np.int8)
    for i, base in enumerate(seq.upper()):
        if base == "A":
            oh[i, 0] = 1
        elif base == "C":
            oh[i, 1] = 1
        elif base == "G":
            oh[i, 2] = 1
        elif base == "T":
            oh[i, 3] = 1
        elif base == "U":
            oh[i, 3] = 1
    return oh


def _create_one_hot_encoding(tx, genome) -> np.ndarray:
    seq = "".join([genome.dna(exon) for exon in tx.exons])
    return _seq_to_oh_dna(seq)


def _create_splice_track(tx) -> np.ndarray:
    len_mrna = sum(len(exon) for exon in tx.exons)
    splicing_track = np.zeros(len_mrna, dtype=np.int8)
    cumulative_len = 0
    for exon in tx.exons:
        cumulative_len += len(exon)
        splicing_track[cumulative_len - 1 : cumulative_len] = 1
    return splicing_track


def _create_cds_track(tx) -> np.ndarray:
    transcript_length = sum(len(exon) for exon in tx.exons)
    if transcript_length == 0:
        return np.array([], dtype=np.int8)

    cds_intervals = list(getattr(tx, "cdss", []) or [])
    if not cds_intervals:
        return np.zeros(transcript_length, dtype=np.int8)

    cds_length = sum(len(c) for c in cds_intervals)

    five_utr_length = 0
    for exon in tx.exons:
        if not exon.overlaps(cds_intervals[0]):
            five_utr_length += len(exon)
        else:
            cds_end5 = cds_intervals[0].end5.start
            exon_end5 = exon.end5.start
            diff = abs(cds_end5 - exon_end5)
            five_utr_length += diff
            break

    three_utr_length = transcript_length - (five_utr_length + cds_length)
    if three_utr_length < 0:
        three_utr_length = 0

    cds_track = np.zeros(cds_length, dtype=np.int8)
    cds_track[0::3] = 1

    return np.concatenate(
        [
            np.zeros(five_utr_length, dtype=np.int8),
            cds_track,
            np.zeros(three_utr_length, dtype=np.int8),
        ]
    )


def _create_six_track_encoding(tx, genome, channels_last: bool = True) -> np.ndarray:
    oh = _create_one_hot_encoding(tx, genome)  # (L,4)
    cds_track = _create_cds_track(tx)  # (L,)
    splice_track = _create_splice_track(tx)  # (L,)

    if channels_last:
        return np.concatenate([oh, cds_track[:, None], splice_track[:, None]], axis=1)

    oh_t = oh.T
    return np.concatenate([oh_t, cds_track[None, :], splice_track[None, :]], axis=0)

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


def filter_hpo_genes(min_genes_per_term: int = 20, dataset_dir: str = None) -> set[str]:
    """Return gene symbols present in HPO terms with at least min_genes_per_term genes."""
    gene2hpos = load_hpo_labels(dataset_dir)  # Series: gene_symbol -> list of hpo_id
    hpo_counts = gene2hpos.explode().value_counts()
    keep_terms = set(hpo_counts[hpo_counts >= min_genes_per_term].index.tolist())
    return {g for g, terms in gene2hpos.items() if any(t in keep_terms for t in terms)}


def six_track_encoding_for_gene(genome: Genome, gene: str, max_len: int | None = None):
    if Genome is None:
        raise RuntimeError("GenomeKit not installed; please install genomekit>=6.0.0 for six-track")

    gene_obj = genome.genes.first_by_name(gene)
    if gene_obj is None:
        # Gene symbols can drift across annotation releases (e.g., AARS1 -> AARS).
        # As a conservative fallback, try stripping a single trailing digit *only*
        # if the resulting symbol exists in this GenomeKit genome.
        if gene and gene[-1].isdigit() and (len(gene) < 2 or not gene[-2].isdigit()):
            base = gene[:-1]
            base_obj = genome.genes.first_by_name(base)
            if base_obj is not None:
                print(f"[six-track] Gene '{gene}' not found in {GENOME_NAME}; using '{base}'")
                gene_obj = base_obj
        if gene_obj is None:
            return None

    txs = list(gene_obj.transcripts)
    if not txs:
        return None

    # pick longest coding transcript
    def tx_len(tx):
        cds_len = getattr(tx, "cds_length", None)
        return cds_len if cds_len is not None else getattr(tx, "length", 0)

    tx = max(txs, key=tx_len)

    arr = _create_six_track_encoding(tx, genome, channels_last=True)  # (L,6)

    if arr is None:
        return None

    if max_len is not None:
        arr = arr[:max_len, :]

    return arr


def generate_orthrus_embeddings(gene2rna, track_type: str = "4", out_dir: str = OUT_DIR):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_id = "quietflamingo/orthrus-large-4-track" if track_type == "4" else "quietflamingo/orthrus-large-6-track"
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True
    ).to(device)
    model.eval()

    os.makedirs(out_dir, exist_ok=True)
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
                lengths = torch.tensor([model_input_tt.shape[1]], device=device)
            else:
                six = six_track_encoding_for_gene(genome, gene)
                if six is None:
                    raise RuntimeError(f"No six-track encoding for {gene}")
                six_tt = torch.tensor(six, dtype=torch.float32, device=device)  # (L,6)
                model_input_tt = six_tt.unsqueeze(0)  # (1,L,6)
                lengths = torch.tensor([model_input_tt.shape[1]], device=device)

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

    # Persist embeddings
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"gene2emb_orthrus_{track_type}track.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(gene2emb, f)
    print(f"Saved {len(gene2emb)} embeddings to: {save_path}")

    return gene2emb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Orthrus gene embeddings")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="Path to dataset folder (contains ensembl_human_cds.fasta; outputs written under it)",
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    out_dir = os.path.join(dataset_dir, "orthrus_embeddings")
    cdna_fasta = os.path.join(dataset_dir, "ensembl_human_cds.fasta")
    gene2rna_path = os.path.join(dataset_dir, "gene2rna.pkl")

    # ---------------- Load omics genes and map to symbols ----------------
    df_emb = load_omics_embedding(dataset_dir)  # index: gene_id
    df_emb = map_ensembl_to_symbol(df_emb)  # reindexed to gene_symbol
    omics_syms = set(df_emb.index.tolist())
    print("Omics genes (symbols):", len(omics_syms))

    # ---------------- Filter HPO genes (>=20 per term) ----------------
    hpo_syms = filter_hpo_genes(min_genes_per_term=MIN_GENES_PER_HPO, dataset_dir=dataset_dir)
    print("HPO genes (filtered):", len(hpo_syms))

    # ---------------- Intersect ----------------
    genes_to_embed = omics_syms & hpo_syms
    print("Genes to embed (omics ∩ HPO filtered):", len(genes_to_embed))

    # ---------------- Load RNA sequences from FASTA ----------------
    if not os.path.exists(cdna_fasta):
        raise FileNotFoundError(
            f"cDNA FASTA not found at {cdna_fasta}. Download from Ensembl or see README for instructions."
        )
    
    gene2rna = gene_symbol_to_cdna_fasta(cdna_fasta, genes_to_embed)
    print("Genes with cDNA sequence:", len(gene2rna))
    if gene2rna:
        first_gene = next(iter(gene2rna))
        print(f"Example {first_gene} RNA length:", len(gene2rna[first_gene]))

    with open(gene2rna_path, "wb") as f:
        pickle.dump(gene2rna, f)
    print(f"Saved gene2rna to: {gene2rna_path}")
    
    # ---------------- Generate embeddings ----------------
    generate_orthrus_embeddings(gene2rna, track_type=TRACK_TYPE, out_dir=out_dir)