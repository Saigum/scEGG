from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import spectral_embedding
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

try:
    from .run_amvf import load_expression_matrix, load_gene_embedding, load_labels, normalize_log_counts, select_hvg, run_amvf
except ImportError:
    from run_amvf import load_expression_matrix, load_gene_embedding, load_labels, normalize_log_counts, select_hvg, run_amvf


def knn_affinity(view, n_neighbors=15):
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
    graph = sparse.csr_matrix((values, (rows, cols)), shape=(view.shape[0], view.shape[0]))
    return graph.maximum(graph.T)


def main():
    counts, _, _ = load_expression_matrix("data/Bjorklund/Bjorklund_data.csv")
    labels = load_labels("data/Bjorklund/label.ann")
    embedding = load_gene_embedding("data/Bjorklund/Bjorklund.emb")
    n_clusters = len(np.unique(labels))

    log_counts = normalize_log_counts(counts)
    gene_idx = select_hvg(log_counts, top_genes=200)
    pca_view = PCA(n_components=20, random_state=0).fit_transform(log_counts[:, gene_idx])

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
        embedding=embedding,
        n_hvg=500,
        latent_dim=20,
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

    results = pd.DataFrame(rows).sort_values("ARI", ascending=False)
    Path("result").mkdir(exist_ok=True)
    output_path = Path("result/bjorklund_benchmark.tsv")
    results.to_csv(output_path, sep="\t", index=False)
    print(results.to_string(index=False))
    print(f"Saved benchmark to {output_path}")


if __name__ == "__main__":
    main()
