"""
分析 train/validation/test 数据集的 HPO label 分布
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mygene
from collections import Counter
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_gene_files(dataset_path="../dataset"):
    """读取 train/val/test 基因文件"""
    train_df = pd.read_csv(f"{dataset_path}/train_genes.tsv", sep="\t")
    val_df = pd.read_csv(f"{dataset_path}/validation_genes.tsv", sep="\t")
    test_df = pd.read_csv(f"{dataset_path}/test_genes.tsv", sep="\t")
    
    train_genes = train_df.iloc[:, 0].tolist()
    val_genes = val_df.iloc[:, 0].tolist()
    test_genes = test_df.iloc[:, 0].tolist()
    
    print(f"Train genes: {len(train_genes)}")
    print(f"Validation genes: {len(val_genes)}")
    print(f"Test genes: {len(test_genes)}")
    
    return train_genes, val_genes, test_genes

def map_ensembl_to_symbol(ensg_ids):
    """将 Ensembl ID 映射到 gene symbol"""
    mg = mygene.MyGeneInfo()
    print("Querying mygene for gene symbols...")
    
    out = mg.querymany(
        ensg_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=True,
        verbose=False
    )
    
    ensg2symbol = {}
    if 'symbol' in out.columns:
        ensg2symbol = out["symbol"].dropna().to_dict()
    
    print(f"Successfully mapped {len(ensg2symbol)} out of {len(ensg_ids)} genes")
    return ensg2symbol

def load_hpo_mapping(dataset_path="../dataset"):
    """加载 gene symbol 到 HPO 的映射"""
    hpo_path = f"{dataset_path}/genes_to_phenotype.txt"
    df_hpo = pd.read_csv(hpo_path, sep="\t", comment="#")
    
    print(f"HPO file columns: {df_hpo.columns.tolist()}")
    
    df_hpo = df_hpo[["gene_symbol", "hpo_id", "hpo_name"]].dropna()
    
    # gene -> [(hpo_id, hpo_name), ...]
    gene2hpos = df_hpo.groupby("gene_symbol").apply(
        lambda x: list(zip(x["hpo_id"], x["hpo_name"]))
    )
    
    print(f"Genes with HPO annotations: {len(gene2hpos)}")
    return gene2hpos, df_hpo

def get_hpo_for_genes(ensg_ids, ensg2symbol, gene2hpos):
    """获取一组基因的 HPO labels"""
    hpo_list = []
    genes_with_hpo = 0
    genes_without_hpo = 0
    
    for ensg in ensg_ids:
        symbol = ensg2symbol.get(ensg)
        if symbol and symbol in gene2hpos.index:
            hpos = gene2hpos[symbol]
            hpo_list.extend([h[0] for h in hpos])  # 只取 hpo_id
            genes_with_hpo += 1
        else:
            genes_without_hpo += 1
    
    return hpo_list, genes_with_hpo, genes_without_hpo

def analyze_hpo_distribution(hpo_list, name, top_n=20):
    """分析 HPO label 分布"""
    counter = Counter(hpo_list)
    total_labels = len(hpo_list)
    unique_labels = len(counter)
    
    print(f"\n{'='*60}")
    print(f"{name} HPO 分布分析")
    print(f"{'='*60}")
    print(f"总 HPO labels 数量: {total_labels}")
    print(f"唯一 HPO terms 数量: {unique_labels}")
    
    if total_labels > 0:
        top_hpos = counter.most_common(top_n)
        print(f"\nTop {top_n} HPO terms:")
        print("-" * 50)
        for i, (hpo, count) in enumerate(top_hpos, 1):
            pct = count / total_labels * 100
            print(f"{i:3}. {hpo}: {count:5} ({pct:.2f}%)")
    
    return counter, total_labels, unique_labels

def plot_distribution(counter, title, save_path, top_n=30):
    """绘制 HPO 分布图"""
    if not counter:
        print(f"No data to plot for {title}")
        return
    
    top_hpos = counter.most_common(top_n)
    hpos = [h[0] for h in top_hpos]
    counts = [h[1] for h in top_hpos]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(hpos)), counts, color='steelblue')
    ax.set_yticks(range(len(hpos)))
    ax.set_yticklabels(hpos, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title(f'{title} - Top {top_n} HPO Terms', fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to: {save_path}")

def plot_comparison(train_counter, val_counter, test_counter, save_path, top_n=20):
    """比较三个数据集的 HPO 分布"""
    # 合并所有 HPO 并取 top N
    all_counter = train_counter + val_counter + test_counter
    top_hpos = [h[0] for h in all_counter.most_common(top_n)]
    
    train_counts = [train_counter.get(h, 0) for h in top_hpos]
    val_counts = [val_counter.get(h, 0) for h in top_hpos]
    test_counts = [test_counter.get(h, 0) for h in top_hpos]
    
    x = np.arange(len(top_hpos))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 10))
    bars1 = ax.barh(x - width, train_counts, width, label='Train', color='steelblue')
    bars2 = ax.barh(x, val_counts, width, label='Validation', color='orange')
    bars3 = ax.barh(x + width, test_counts, width, label='Test', color='green')
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_hpos, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title(f'HPO Distribution Comparison - Top {top_n} Terms', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {save_path}")

def plot_hpo_count_distribution(counters, names, save_path):
    """绘制每个 HPO term 出现次数的分布直方图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    all_counter = Counter()
    for i, (counter, name) in enumerate(zip(counters, names)):
        if counter:
            counts = list(counter.values())
            axes[i].hist(counts, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
            axes[i].set_xlabel('Frequency (times a HPO term appears)', fontsize=10)
            axes[i].set_ylabel('Number of HPO terms', fontsize=10)
            axes[i].set_title(f'{name} - HPO Term Frequency Distribution', fontsize=12, fontweight='bold')
            axes[i].set_yscale('log')
            all_counter += counter
    
    # 总体分布
    if all_counter:
        counts = list(all_counter.values())
        axes[3].hist(counts, bins=50, color='purple', edgecolor='white', alpha=0.7)
        axes[3].set_xlabel('Frequency (times a HPO term appears)', fontsize=10)
        axes[3].set_ylabel('Number of HPO terms', fontsize=10)
        axes[3].set_title('Overall - HPO Term Frequency Distribution', fontsize=12, fontweight='bold')
        axes[3].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Frequency distribution plot saved to: {save_path}")

def print_summary_statistics(counters, names, genes_info):
    """打印汇总统计信息"""
    print("\n" + "="*80)
    print("汇总统计 (Summary Statistics)")
    print("="*80)
    
    print(f"\n{'Dataset':<15} {'Genes':<10} {'w/ HPO':<10} {'w/o HPO':<10} {'Total Labels':<15} {'Unique HPO':<12}")
    print("-"*80)
    
    all_counter = Counter()
    for counter, name, info in zip(counters, names, genes_info):
        genes_with, genes_without = info
        total = sum(counter.values())
        unique = len(counter)
        print(f"{name:<15} {genes_with + genes_without:<10} {genes_with:<10} {genes_without:<10} {total:<15} {unique:<12}")
        all_counter += counter
    
    # 总体统计
    total_genes = sum(info[0] + info[1] for info in genes_info)
    total_with_hpo = sum(info[0] for info in genes_info)
    total_without_hpo = sum(info[1] for info in genes_info)
    print("-"*80)
    print(f"{'Overall':<15} {total_genes:<10} {total_with_hpo:<10} {total_without_hpo:<10} {sum(all_counter.values()):<15} {len(all_counter):<12}")
    
    return all_counter

def compare_three_sets(train_counter, val_counter, test_counter, df_hpo, output_dir):
    """比较三个数据集的 HPO label 分布"""
    train_set = set(train_counter.keys())
    val_set = set(val_counter.keys())
    test_set = set(test_counter.keys())
    
    # 集合运算
    all_hpos = train_set | val_set | test_set
    common_all = train_set & val_set & test_set
    train_only = train_set - val_set - test_set
    val_only = val_set - train_set - test_set
    test_only = test_set - train_set - val_set
    train_val = (train_set & val_set) - test_set
    train_test = (train_set & test_set) - val_set
    val_test = (val_set & test_set) - train_set
    
    print("\n" + "="*80)
    print("三个数据集 HPO Label 集合比较")
    print("="*80)
    print(f"\n总共唯一 HPO terms: {len(all_hpos)}")
    print(f"三个集合共有: {len(common_all)} ({len(common_all)/len(all_hpos)*100:.1f}%)")
    print(f"Train 独有: {len(train_only)} ({len(train_only)/len(all_hpos)*100:.1f}%)")
    print(f"Validation 独有: {len(val_only)} ({len(val_only)/len(all_hpos)*100:.1f}%)")
    print(f"Test 独有: {len(test_only)} ({len(test_only)/len(all_hpos)*100:.1f}%)")
    print(f"仅 Train+Val 共有: {len(train_val)}")
    print(f"仅 Train+Test 共有: {len(train_test)}")
    print(f"仅 Val+Test 共有: {len(val_test)}")
    
    # 创建比较表格
    hpo_name_map = df_hpo.drop_duplicates("hpo_id").set_index("hpo_id")["hpo_name"].to_dict()
    
    comparison_data = []
    for hpo in sorted(all_hpos):
        comparison_data.append({
            "hpo_id": hpo,
            "hpo_name": hpo_name_map.get(hpo, ""),
            "train_count": train_counter.get(hpo, 0),
            "val_count": val_counter.get(hpo, 0),
            "test_count": test_counter.get(hpo, 0),
            "total_count": train_counter.get(hpo, 0) + val_counter.get(hpo, 0) + test_counter.get(hpo, 0),
            "in_train": "✓" if hpo in train_set else "",
            "in_val": "✓" if hpo in val_set else "",
            "in_test": "✓" if hpo in test_set else "",
            "category": get_category(hpo, train_set, val_set, test_set)
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values("total_count", ascending=False)
    df_comparison.to_csv(f"{output_dir}/hpo_three_sets_comparison.csv", index=False)
    print(f"\n详细比较表保存至: {output_dir}/hpo_three_sets_comparison.csv")
    
    # 打印 Top HPO terms 比较
    print("\n" + "-"*100)
    print("Top 30 HPO Terms 在三个集合中的分布:")
    print("-"*100)
    print(f"{'HPO ID':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10} {'Category':<20}")
    print("-"*100)
    
    for _, row in df_comparison.head(30).iterrows():
        print(f"{row['hpo_id']:<15} {row['train_count']:<10} {row['val_count']:<10} {row['test_count']:<10} {row['total_count']:<10} {row['category']:<20}")
    
    # 绘制 Venn 图 (简化版本用条形图)
    plot_set_comparison(train_set, val_set, test_set, f"{output_dir}/hpo_sets_comparison.png")
    
    # 绘制热力图比较
    plot_heatmap_comparison(df_comparison.head(50), f"{output_dir}/hpo_top50_heatmap.png")
    
    return df_comparison

def get_category(hpo, train_set, val_set, test_set):
    """获取 HPO 所属的类别"""
    in_train = hpo in train_set
    in_val = hpo in val_set
    in_test = hpo in test_set
    
    if in_train and in_val and in_test:
        return "All Three"
    elif in_train and in_val:
        return "Train+Val Only"
    elif in_train and in_test:
        return "Train+Test Only"
    elif in_val and in_test:
        return "Val+Test Only"
    elif in_train:
        return "Train Only"
    elif in_val:
        return "Val Only"
    else:
        return "Test Only"

def plot_set_comparison(train_set, val_set, test_set, save_path):
    """绘制集合比较图"""
    all_hpos = train_set | val_set | test_set
    common_all = train_set & val_set & test_set
    train_only = train_set - val_set - test_set
    val_only = val_set - train_set - test_set
    test_only = test_set - train_set - val_set
    train_val = (train_set & val_set) - test_set
    train_test = (train_set & test_set) - val_set
    val_test = (val_set & test_set) - train_set
    
    categories = ['All Three', 'Train Only', 'Val Only', 'Test Only', 
                  'Train+Val', 'Train+Test', 'Val+Test']
    counts = [len(common_all), len(train_only), len(val_only), len(test_only),
              len(train_val), len(train_test), len(val_test)]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e91e63']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 条形图
    bars = ax1.bar(categories, counts, color=colors, edgecolor='white', linewidth=1.5)
    ax1.set_ylabel('Number of HPO Terms', fontsize=12)
    ax1.set_title('HPO Terms Distribution Across Datasets', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                 str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 饼图
    ax2.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('HPO Terms Category Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Set comparison plot saved to: {save_path}")

def plot_heatmap_comparison(df, save_path):
    """绘制热力图比较 Top HPO"""
    import matplotlib.colors as mcolors
    
    # 准备数据
    hpo_ids = df['hpo_id'].tolist()
    train_counts = df['train_count'].tolist()
    val_counts = df['val_count'].tolist()
    test_counts = df['test_count'].tolist()
    
    # 归一化 (每行归一化)
    data = np.array([train_counts, val_counts, test_counts]).T
    row_sums = data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除以0
    data_normalized = data / row_sums
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 14))
    
    # 原始计数热力图
    im1 = ax1.imshow(data, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(['Train', 'Validation', 'Test'], fontsize=11)
    ax1.set_yticks(range(len(hpo_ids)))
    ax1.set_yticklabels(hpo_ids, fontsize=8)
    ax1.set_title('HPO Counts (Absolute)', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Count')
    
    # 添加数值标注
    for i in range(len(hpo_ids)):
        for j in range(3):
            ax1.text(j, i, str(int(data[i, j])), ha='center', va='center', 
                     fontsize=7, color='black' if data[i, j] < data.max()*0.7 else 'white')
    
    # 归一化比例热力图
    im2 = ax2.imshow(data_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['Train', 'Validation', 'Test'], fontsize=11)
    ax2.set_yticks(range(len(hpo_ids)))
    ax2.set_yticklabels(hpo_ids, fontsize=8)
    ax2.set_title('HPO Distribution (Row Normalized)', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Proportion')
    
    # 添加比例标注
    for i in range(len(hpo_ids)):
        for j in range(3):
            ax2.text(j, i, f'{data_normalized[i, j]:.2f}', ha='center', va='center', 
                     fontsize=7, color='black' if data_normalized[i, j] < 0.7 else 'white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap comparison saved to: {save_path}")

def main():
    dataset_path = "../dataset"
    output_dir = "../dataset/hpo_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取基因文件
    print("="*60)
    print("Step 1: 读取基因文件")
    print("="*60)
    train_genes, val_genes, test_genes = load_gene_files(dataset_path)
    
    # 2. 将 Ensembl ID 映射到 gene symbol
    print("\n" + "="*60)
    print("Step 2: 映射 Ensembl ID 到 Gene Symbol")
    print("="*60)
    all_genes = train_genes + val_genes + test_genes
    ensg2symbol = map_ensembl_to_symbol(all_genes)
    
    # 3. 加载 HPO 映射
    print("\n" + "="*60)
    print("Step 3: 加载 HPO 映射")
    print("="*60)
    gene2hpos, df_hpo = load_hpo_mapping(dataset_path)
    
    # 4. 获取每个数据集的 HPO labels
    print("\n" + "="*60)
    print("Step 4: 获取 HPO Labels")
    print("="*60)
    
    train_hpos, train_with, train_without = get_hpo_for_genes(train_genes, ensg2symbol, gene2hpos)
    val_hpos, val_with, val_without = get_hpo_for_genes(val_genes, ensg2symbol, gene2hpos)
    test_hpos, test_with, test_without = get_hpo_for_genes(test_genes, ensg2symbol, gene2hpos)
    
    print(f"Train: {train_with} genes with HPO, {train_without} genes without HPO")
    print(f"Validation: {val_with} genes with HPO, {val_without} genes without HPO")
    print(f"Test: {test_with} genes with HPO, {test_without} genes without HPO")
    
    # 5. 分析分布
    train_counter, train_total, train_unique = analyze_hpo_distribution(train_hpos, "Train", top_n=20)
    val_counter, val_total, val_unique = analyze_hpo_distribution(val_hpos, "Validation", top_n=20)
    test_counter, test_total, test_unique = analyze_hpo_distribution(test_hpos, "Test", top_n=20)
    
    # 6. 总体分布
    all_hpos = train_hpos + val_hpos + test_hpos
    all_counter, all_total, all_unique = analyze_hpo_distribution(all_hpos, "Overall", top_n=20)
    
    # 7. 汇总统计
    counters = [train_counter, val_counter, test_counter]
    names = ["Train", "Validation", "Test"]
    genes_info = [(train_with, train_without), (val_with, val_without), (test_with, test_without)]
    print_summary_statistics(counters, names, genes_info)
    
    # 7.5 三个集合比较
    df_comparison = compare_three_sets(train_counter, val_counter, test_counter, df_hpo, output_dir)
    
    # 8. 绘图
    print("\n" + "="*60)
    print("Step 5: 生成可视化图表")
    print("="*60)
    
    plot_distribution(train_counter, "Train", f"{output_dir}/train_hpo_distribution.png")
    plot_distribution(val_counter, "Validation", f"{output_dir}/validation_hpo_distribution.png")
    plot_distribution(test_counter, "Test", f"{output_dir}/test_hpo_distribution.png")
    plot_distribution(all_counter, "Overall", f"{output_dir}/overall_hpo_distribution.png")
    
    plot_comparison(train_counter, val_counter, test_counter, f"{output_dir}/hpo_comparison.png")
    plot_hpo_count_distribution(counters, names, f"{output_dir}/hpo_frequency_distribution.png")
    
    # 9. 保存详细数据到 CSV
    print("\n" + "="*60)
    print("Step 6: 保存详细数据")
    print("="*60)
    
    # 保存每个数据集的 HPO 计数
    for counter, name in [(train_counter, "train"), (val_counter, "validation"), 
                          (test_counter, "test"), (all_counter, "overall")]:
        df = pd.DataFrame(counter.most_common(), columns=["hpo_id", "count"])
        # 添加 HPO name
        hpo_name_map = df_hpo.drop_duplicates("hpo_id").set_index("hpo_id")["hpo_name"].to_dict()
        df["hpo_name"] = df["hpo_id"].map(hpo_name_map)
        df.to_csv(f"{output_dir}/{name}_hpo_counts.csv", index=False)
        print(f"Saved {name}_hpo_counts.csv")
    
    print("\n" + "="*60)
    print("分析完成！")
    print(f"结果保存在: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
