import os
import requests
import numpy as np
import torch
import mygene
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from utils import load_omics_embedding, load_hpo_labels

# Settings
OUT_DIR = "../dataset/esm_c_embeddings"
# ESM-3 C-series public checkpoints
#   "esmc_300m"  (smaller)
#   "esmc_600m"  (larger)
MODEL_NAME = "esmc_600m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Truncate very long proteins for safety; adjust if needed.
MAX_SEQ_LEN = 1024
MIN_GENES_PER_HPO = 20


def map_gene_id_to_symbol(gene_ids: list[str]) -> dict[str, str]:
    """Map Ensembl gene IDs to official gene symbols using mygene."""
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        gene_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=True,
        returnall=False,
        verbose=False,
    )

    mapping = {}
    for ensg, row in res.iterrows():
        sym = row.get("symbol")
        ensg_id = ensg[0] if isinstance(ensg, tuple) else ensg
        if isinstance(sym, str):
            mapping[ensg_id] = sym
    return mapping


def map_symbol_to_uniprot(gene_symbols: list[str]) -> dict[str, str]:
    """Map gene symbols to a UniProt accession (Swiss-Prot preferred, else TrEMBL)."""
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        gene_symbols,
        scopes="symbol",
        fields="uniprot.Swiss-Prot,uniprot.TrEMBL",
        species="human",
        as_dataframe=True,
        returnall=False,
        verbose=False,
    )

    mapping = {}
    for sym, row in res.iterrows():
        acc = None
        swiss = row.get("uniprot.Swiss-Prot")
        trembl = row.get("uniprot.TrEMBL")

        if isinstance(swiss, list):
            acc = swiss[0]
        elif isinstance(swiss, str):
            acc = swiss
        elif isinstance(trembl, list):
            acc = trembl[0]
        elif isinstance(trembl, str):
            acc = trembl

        sym_key = sym[0] if isinstance(sym, tuple) else sym
        if acc:
            mapping[sym_key] = acc
    return mapping


def fetch_uniprot_fasta(accession: str) -> str | None:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"Failed to fetch FASTA for {accession}: {exc}")
        return None
    return r.text


def fasta_to_seq(fasta_str: str) -> str | None:
    lines = [l.strip() for l in fasta_str.splitlines() if l.strip()]
    seq_lines = [l for l in lines if not l.startswith(">")]  # drop header
    seq = "".join(seq_lines)
    return seq if seq else None


def load_model():
    print(f"Loading {MODEL_NAME} to {DEVICE} ...")
    client = ESMC.from_pretrained(MODEL_NAME).to(DEVICE)
    client.eval()
    return client


def embed_sequence(client, seq: str) -> np.ndarray:
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
    pooled = embeddings.squeeze(0).mean(dim=0).cpu().numpy()
    return pooled


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df_emb = load_omics_embedding()  # reuses utils; index is gene_id
    gene_ids = df_emb.index.tolist()
    total_gene_ids = len(gene_ids)
    print(f"Loaded {total_gene_ids} gene IDs from omics embeddings")

    ensg2sym = map_gene_id_to_symbol(gene_ids)
    mapped_symbols = len(ensg2sym)
    print(f"Mapped {mapped_symbols} Ensembl IDs to symbols; missing {total_gene_ids - mapped_symbols}")

    gene2hpos = load_hpo_labels()
    # Filter terms with >= MIN_GENES_PER_HPO
    hpo_counts = gene2hpos.explode().value_counts()
    keep_terms = set(hpo_counts[hpo_counts >= MIN_GENES_PER_HPO].index.tolist())
    hpo_syms = {g for g, terms in gene2hpos.items() if any(t in keep_terms for t in terms)}
    print(f"HPO genes (after filtering terms with >= {MIN_GENES_PER_HPO} genes): {len(hpo_syms)}")

    genes_common = [gid for gid, sym in ensg2sym.items() if sym in hpo_syms]
    print(f"Genes with both omics embeddings and filtered HPO labels: {len(genes_common)}")

    symbols_common = [ensg2sym[gid] for gid in genes_common]
    sym2uniprot = map_symbol_to_uniprot(symbols_common)
    print(f"Mapped {len(sym2uniprot)} symbols to UniProt; missing {len(symbols_common) - len(sym2uniprot)}")

    client = load_model()

    summary = []
    status_counts = {"ok": 0, "no_uniprot": 0, "fetch_failed": 0, "no_sequence": 0, "embed_failed": 0, "skipped_existing": 0}
    first_shape_logged = False

    for i, gene_id in enumerate(genes_common, 1):
        out_path = os.path.join(OUT_DIR, f"{gene_id}.npy")
        if os.path.exists(out_path):
            status_counts["skipped_existing"] += 1
            continue

        sym = ensg2sym.get(gene_id)
        acc = sym2uniprot.get(sym)
        if not acc:
            summary.append((gene_id, None, 0, "no_uniprot"))
            status_counts["no_uniprot"] += 1
            continue

        fasta = fetch_uniprot_fasta(acc)
        if not fasta:
            summary.append((gene_id, acc, 0, "fetch_failed"))
            status_counts["fetch_failed"] += 1
            continue

        seq = fasta_to_seq(fasta)
        if not seq:
            summary.append((gene_id, acc, 0, "no_sequence"))
            status_counts["no_sequence"] += 1
            continue

        try:
            emb = embed_sequence(client, seq)
            np.save(out_path, emb)
            summary.append((gene_id, acc, len(seq), "ok"))
            status_counts["ok"] += 1
            if not first_shape_logged:
                print(f"First embedding shape: {emb.shape} (gene_id {gene_id}, symbol {sym})")
                first_shape_logged = True
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to embed {gene_id} ({acc}): {exc}")
            summary.append((gene_id, acc, len(seq), "embed_failed"))
            status_counts["embed_failed"] += 1

        if i % 10 == 0:
            print(f"Processed {i}/{len(genes_common)} genes", end="\r")

    summary_path = os.path.join(OUT_DIR, "embedding_summary.tsv")
    pd.DataFrame(summary, columns=["gene_id", "uniprot_acc", "seq_len", "status"]).to_csv(
        summary_path,
        sep="\t",
        index=False,
    )
    print(f"\nDone. Saved embeddings to {OUT_DIR}. Summary: {summary_path}")
    print(
        "Status counts -> ok: {ok}, embed_failed: {embed_failed}, fetch_failed: {fetch_failed}, "
        "no_sequence: {no_sequence}, no_uniprot: {no_uniprot}, skipped_existing: {skipped_existing}".format(**status_counts)
    )


if __name__ == "__main__":
    main()
