import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mygene

# ---------- 1. Read Omics embedding ----------
def load_omics_embedding():
    data_path = "../dataset/"
    omic_path = "omics_embeddings"
    emb_path = data_path + omic_path + "/Supplementary_Table_S3_OMICS_EMB.tsv"

    df_emb = pd.read_csv(emb_path, sep="\t")

    gene_col = "gene_id"
    df_emb = df_emb.set_index(gene_col)

    X_emb = df_emb.values.astype(np.float32)   
    emb_genes = df_emb.index.tolist()
    print("Embedding genes:", len(emb_genes), "dim =", X_emb.shape[1])
    return df_emb

# ---------------- 2. Ensembl ID -> gene symbol ----------------
def map_ensembl_to_symbol(df_emb):
    mg = mygene.MyGeneInfo()
    ensg_ids = df_emb.index.tolist()

    print("Querying mygene for gene symbols ...")
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

    print("After mapping and deduplication:", df_emb.shape[0], "genes with symbols")
    return df_emb

# ---------------- 3. Read HPO gene labels ----------------
def load_hpo_labels():
    hpo_path = "../dataset/genes_to_phenotype.txt"  

    df_hpo = pd.read_csv(hpo_path, sep="\t", comment="#")
    print("HPO columns:", df_hpo.columns)

    df_hpo = df_hpo[["gene_symbol", "hpo_id"]].dropna()

    # gene -> [hpo1, hpo2, ...]
    gene2hpos = df_hpo.groupby("gene_symbol")["hpo_id"].apply(list)
    print("Genes with HPO labels:", len(gene2hpos))
    return gene2hpos

# ---------------- 4. Align genes with both embedding and HPO labels ----------------
def align_genes(df_emb, gene2hpos):
    genes_common = sorted(set(df_emb.index) & set(gene2hpos.index))
    print("Genes with both embedding and HPO labels:", len(genes_common))

    hpo_terms = sorted({h for g in genes_common for h in gene2hpos[g]})
    hpo2idx = {h: i for i, h in enumerate(hpo_terms)}
    num_hpo = len(hpo_terms)
    print("Number of HPO terms:", num_hpo)
    return genes_common, hpo2idx, num_hpo

# ---------------- 5. Generate X, Y matrices ----------------
def split_data():
    df_emb = load_omics_embedding()
    df_emb = map_ensembl_to_symbol(df_emb)
    gene2hpos = load_hpo_labels()
    genes_common, hpo2idx, num_hpo = align_genes(df_emb, gene2hpos)
    X_list, Y_list = [], []

    for g in genes_common:
        x = df_emb.loc[g].values.astype(np.float32)  
        y = np.zeros(num_hpo, dtype=np.float32)
        for h in gene2hpos[g]:
            idx = hpo2idx[h]
            y[idx] = 1.0
        X_list.append(x)
        Y_list.append(y)

    X = np.stack(X_list)   # [N, 256]
    Y = np.stack(Y_list)   # [N, num_hpo]

    print("X shape:", X.shape, "Y shape:", Y.shape)
# ---------------- 6. Split train/val/test ----------------
    N = X.shape[0]
    indices = np.arange(N)

    idx_train, idx_tmp = train_test_split(indices, test_size=0.3, random_state=42)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=42)

    def take(arr, idx):
        return arr[idx]

    X_train, Y_train = take(X, idx_train), take(Y, idx_train)
    X_val,   Y_val   = take(X, idx_val),   take(Y, idx_val)
    X_test,  Y_test  = take(X, idx_test),  take(Y, idx_test)

    print("train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

