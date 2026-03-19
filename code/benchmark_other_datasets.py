import argparse
import gzip
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as scipy_io
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import spectral_embedding
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

try:
    from .run_amvf import normalize_log_counts, run_amvf, select_hvg
except ImportError:
    from run_amvf import normalize_log_counts, run_amvf, select_hvg


def knn_affinity(view, n_neighbors=15):
    n_neighbors = min(n_neighbors, view.shape[0] - 1)
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(view)
    distances, indices = neighbors.kneighbors(view)
    sigma = np.maximum(distances[:, -1], 1e-6)
    rows = []
    cols = []
    values = []
    for row_index in range(view.shape[0]):
        for local_index, col_index in enumerate(indices[row_index, 1:], start=1):
            affinity = np.exp(-(distances[row_index, local_index] ** 2) / (sigma[row_index] * sigma[col_index] + 1e-6))
            rows.append(row_index)
            cols.append(col_index)
            values.append(affinity)
    from scipy import sparse

    graph = sparse.csr_matrix((values, (rows, cols)), shape=(view.shape[0], view.shape[0]))
    return graph.maximum(graph.T)


def stratified_subset(mapping, max_cells, random_state=0):
    if len(mapping) <= max_cells:
        return mapping.reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    by_label = mapping.groupby("cell_type")
    allocations = {}
    for label, frame in by_label:
        allocations[label] = max(1, int(round(max_cells * len(frame) / len(mapping))))

    total = sum(allocations.values())
    labels = list(allocations)
    while total > max_cells:
        label = max(labels, key=lambda value: allocations[value])
        if allocations[label] > 1:
            allocations[label] -= 1
            total -= 1
        else:
            break
    while total < max_cells:
        label = max(labels, key=lambda value: len(by_label.get_group(value)) - allocations[value])
        if allocations[label] < len(by_label.get_group(label)):
            allocations[label] += 1
            total += 1
        else:
            break

    parts = []
    for label, frame in by_label:
        take = min(len(frame), allocations[label])
        chosen = np.sort(rng.choice(len(frame), size=take, replace=False))
        parts.append(frame.iloc[chosen])
    subset = pd.concat(parts, ignore_index=True)
    return subset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_guo_counts(selected_cells):
    header = pd.read_csv("data/datasets/Guo/GSE99254.tsv", sep="\t", nrows=0)
    first_column = header.columns[0]
    keep = [first_column] + list(selected_cells)
    frame = pd.read_csv("data/datasets/Guo/GSE99254.tsv", sep="\t", usecols=keep, index_col=0)
    return frame.T.to_numpy(dtype=np.float32), frame.columns.astype(str).to_numpy()


def load_mtx_counts(matrix_path, barcodes_path, selected_cells=None):
    open_fn = gzip.open if str(matrix_path).endswith(".gz") else open
    with open_fn(matrix_path, "rb") as handle:
        matrix = scipy_io.mmread(handle).tocsr().T
    open_text = gzip.open if str(barcodes_path).endswith(".gz") else open
    with open_text(barcodes_path, "rt") as handle:
        barcodes = np.array([line.strip() for line in handle])
    if selected_cells is not None:
        index = pd.Index(barcodes)
        selected_index = index.get_indexer(selected_cells)
        if (selected_index < 0).any():
            missing = list(np.array(selected_cells)[selected_index < 0][:5])
            raise ValueError(f"Missing selected cells in matrix: {missing}")
        matrix = matrix[selected_index]
        barcodes = barcodes[selected_index]
    return matrix.toarray().astype(np.float32), barcodes


def benchmark_dataset(name, counts, labels, n_clusters, n_hvg=200, latent_dim=20):
    log_counts = normalize_log_counts(counts)
    gene_idx = select_hvg(log_counts, top_genes=n_hvg)
    n_components = min(latent_dim, len(counts) - 1, len(gene_idx))
    n_components = max(2, n_components)
    pca_view = PCA(n_components=n_components, random_state=0).fit_transform(log_counts[:, gene_idx])

    methods = []

    prediction = KMeans(n_clusters=n_clusters, n_init=100, random_state=0).fit_predict(pca_view)
    methods.append(("KMeans-PCA", prediction))

    prediction = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(pca_view)
    methods.append(("Agglomerative-PCA", prediction))

    graph = knn_affinity(pca_view, n_neighbors=15)
    spectral_view = spectral_embedding(graph, n_components=n_clusters, random_state=0, drop_first=False)
    prediction = KMeans(n_clusters=n_clusters, n_init=100, random_state=0).fit_predict(spectral_view)
    methods.append(("Spectral-kNN", prediction))

    amvf = run_amvf(
        counts,
        n_clusters=n_clusters,
        embedding=None,
        n_hvg=max(500, n_hvg),
        latent_dim=latent_dim,
        n_neighbors=15,
        adaptivity=0.0,
    )
    methods.append(("AMVF", amvf["prediction"]))

    rows = []
    for method_name, prediction in methods:
        rows.append({
            "method": method_name,
            "ARI": round(adjusted_rand_score(labels, prediction), 4),
            "NMI": round(normalized_mutual_info_score(labels, prediction), 4),
        })
    return rows


def load_standardized_mapping():
    mapping_path = Path("result/standardized_celltype_mappings.tsv")
    if not mapping_path.exists():
        raise FileNotFoundError("Run code/build_standardized_mappings.py first")
    return pd.read_csv(mapping_path, sep="\t", low_memory=False)


def benchmark_configs(mapping):
    available = mapping[mapping["label_status"] == "available"].copy()
    available = available[available["cell_type"].astype(str) != ""]

    configs = []

    guo = available[available["dataset_id"] == "Guo"].copy()
    configs.append({
        "dataset_id": "Guo",
        "sample_id": "Guo",
        "label_kind": guo["label_kind"].iloc[0],
        "mapping": guo,
        "loader": lambda cells: load_guo_counts(cells),
    })

    pbmc = available[available["dataset_id"] == "pbmc6k"].copy()
    configs.append({
        "dataset_id": "pbmc6k",
        "sample_id": "pbmc6k",
        "label_kind": pbmc["label_kind"].iloc[0],
        "mapping": pbmc,
        "loader": lambda cells: load_mtx_counts(
            "data/datasets/pbmc/pbmc6k_matrices/matrix.mtx",
            "data/datasets/pbmc/pbmc6k_matrices/barcodes.tsv",
            cells,
        ),
    })

    for sample_id, matrix_path, barcode_path in [
        ("sampleA", "data/datasets/pbmc_perturb/sampleA/GSM2560245_A.mat.gz", "data/datasets/pbmc_perturb/sampleA/GSM2560245_barcodes.tsv.gz"),
        ("sampleB", "data/datasets/pbmc_perturb/sampleB/GSM2560246_B.mat.gz", "data/datasets/pbmc_perturb/sampleB/GSM2560246_barcodes.tsv.gz"),
        ("sampleC", "data/datasets/pbmc_perturb/sampleC/GSM2560247_C.mat.gz", "data/datasets/pbmc_perturb/sampleC/GSM2560247_barcodes.tsv.gz"),
        ("sample2.1", "data/datasets/pbmc_perturb/sample2.1/GSM2560248_2.1.mtx.gz", "data/datasets/pbmc_perturb/sample2.1/GSM2560248_barcodes.tsv.gz"),
        ("sample2.2", "data/datasets/pbmc_perturb/sample2.2/GSM2560249_2.2.mtx.gz", "data/datasets/pbmc_perturb/sample2.2/GSM2560249_barcodes.tsv.gz"),
    ]:
        sample_mapping = available[(available["dataset_id"] == "pbmc_perturb") & (available["sample_id"] == sample_id)].copy()
        configs.append({
            "dataset_id": "pbmc_perturb",
            "sample_id": sample_id,
            "label_kind": sample_mapping["label_kind"].iloc[0],
            "mapping": sample_mapping,
            "loader": lambda cells, matrix_path=matrix_path, barcode_path=barcode_path: load_mtx_counts(matrix_path, barcode_path, cells),
        })

    return configs


def main():
    parser = argparse.ArgumentParser(description="Benchmark classical baselines and AMVF on labeled datasets")
    parser.add_argument("--max_cells", default=2000, type=int)
    parser.add_argument("--output_path", default="result/other_dataset_benchmark.tsv")
    args = parser.parse_args()

    mapping = load_standardized_mapping()
    rows = []

    for config in benchmark_configs(mapping):
        subset = stratified_subset(config["mapping"], max_cells=args.max_cells, random_state=0)
        counts, loaded_cells = config["loader"](subset["cell_id"].tolist())
        if list(loaded_cells.astype(str)) != subset["cell_id"].astype(str).tolist():
            order = pd.Index(loaded_cells.astype(str)).get_indexer(subset["cell_id"].astype(str))
            counts = counts[order]
        labels = pd.Categorical(subset["cell_type"].astype(str)).codes
        n_clusters = subset["cell_type"].nunique()
        metrics = benchmark_dataset(
            f"{config['dataset_id']}::{config['sample_id']}",
            counts,
            labels,
            n_clusters=n_clusters,
        )
        for metric in metrics:
            rows.append({
                "dataset_id": config["dataset_id"],
                "sample_id": config["sample_id"],
                "label_kind": config["label_kind"],
                "n_cells_total": len(config["mapping"]),
                "n_cells_used": len(subset),
                "n_clusters": n_clusters,
                **metric,
            })
        print(f"Benchmarked {config['dataset_id']}::{config['sample_id']} on {len(subset)} cells")

    result = pd.DataFrame(rows).sort_values(["dataset_id", "sample_id", "ARI"], ascending=[True, True, False])
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False)
    print(result.to_string(index=False))
    print(f"Saved benchmark to {output_path}")


if __name__ == "__main__":
    main()
