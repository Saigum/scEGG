import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


def normalize_log_counts(counts, scale_factor=1e4):
    library = counts.sum(axis=1, keepdims=True)
    library[library == 0] = 1.0
    return np.log1p(counts / library * scale_factor)


def select_hvg(log_counts, top_genes=200):
    means = log_counts.mean(axis=0)
    variances = log_counts.var(axis=0)
    dispersion = variances / np.maximum(means, 1e-8)
    top_genes = min(top_genes, log_counts.shape[1])
    gene_idx = np.argsort(dispersion)[-top_genes:]
    return gene_idx


def tfidf_view(counts):
    library = counts.sum(axis=1, keepdims=True)
    library[library == 0] = 1.0
    term_frequency = counts / library
    inverse_document_frequency = np.log1p(counts.shape[0] / (1 + (counts > 0).sum(axis=0, keepdims=True)))
    return np.log1p(term_frequency * inverse_document_frequency)


def binary_program_view(counts):
    return normalize((counts > 0).astype(np.float32), norm="l1")


def reduce_view(matrix, n_components, use_pca=True):
    n_components = min(n_components, matrix.shape[0] - 1, matrix.shape[1])
    n_components = max(n_components, 2)
    if sparse.issparse(matrix) or not use_pca:
        reduced = TruncatedSVD(n_components=n_components, random_state=0).fit_transform(matrix)
    else:
        reduced = PCA(n_components=n_components, random_state=0).fit_transform(matrix)
    return StandardScaler().fit_transform(reduced)


def load_expression_matrix(data_path):
    data = pd.read_csv(data_path, index_col=0)
    return data.values.T.astype(np.float32), data.columns.astype(str).to_numpy(), data.index.astype(str).to_numpy()


def load_gene_embedding(embedding_path):
    if embedding_path is None:
        return None
    embedding = pd.read_csv(embedding_path, sep="\s+", header=None, skiprows=1)
    return embedding.drop(columns=[0]).values.astype(np.float32)


def load_labels(label_path):
    labels = pd.read_csv(label_path, sep="\t")
    label_series = labels.iloc[:, -1]
    if np.issubdtype(label_series.dtype, np.number):
        return label_series.to_numpy()
    return pd.Categorical(label_series.astype(str)).codes


def local_view_confidence(view, n_neighbors=15):
    nn = min(n_neighbors, view.shape[0] - 1)
    neighbors = NearestNeighbors(n_neighbors=nn)
    neighbors.fit(view)
    distances, _ = neighbors.kneighbors(view)
    confidence = 1.0 / (1e-6 + distances[:, 1:].mean(axis=1))
    return confidence


def build_views(counts, embedding=None, n_hvg=500, latent_dim=20):
    log_counts = normalize_log_counts(counts)
    gene_idx = select_hvg(log_counts, top_genes=n_hvg)

    views = {
        "expression": reduce_view(log_counts[:, gene_idx], latent_dim, use_pca=True),
        "rare_program": reduce_view(binary_program_view(counts), latent_dim, use_pca=False),
        "rarity_tfidf": reduce_view(tfidf_view(counts), latent_dim, use_pca=False),
    }

    if embedding is not None and embedding.shape[0] == counts.shape[1]:
        embedding_projection = log_counts @ embedding
        views["gene_geometry"] = reduce_view(embedding_projection, latent_dim, use_pca=True)

    return views


def adaptive_weighted_representation(views, n_neighbors=15, adaptivity=0.0):
    view_names = list(views.keys())
    confidence_matrix = np.column_stack([
        local_view_confidence(views[name], n_neighbors=n_neighbors)
        for name in view_names
    ])
    global_weights = confidence_matrix.mean(axis=0)
    global_weights = global_weights / np.maximum(global_weights.sum(), 1e-8)
    equal_weights = np.full_like(global_weights, 1.0 / global_weights.shape[0])
    global_weights = (1.0 - adaptivity) * equal_weights + adaptivity * global_weights

    weighted_parts = []
    for column_idx, name in enumerate(view_names):
        weighted_parts.append(views[name] * np.sqrt(global_weights[column_idx]))
    weights = np.repeat(global_weights.reshape(1, -1), confidence_matrix.shape[0], axis=0)
    return np.hstack(weighted_parts), weights, view_names


def run_amvf(counts, n_clusters, embedding=None, n_hvg=500, latent_dim=20, n_neighbors=15, adaptivity=0.0):
    views = build_views(counts, embedding=embedding, n_hvg=n_hvg, latent_dim=latent_dim)
    fused_representation, weights, view_names = adaptive_weighted_representation(
        views,
        n_neighbors=n_neighbors,
        adaptivity=adaptivity,
    )
    prediction = KMeans(n_clusters=n_clusters, n_init=100, random_state=0).fit_predict(fused_representation)
    return {
        "prediction": prediction,
        "representation": fused_representation,
        "weights": weights,
        "view_names": view_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Adaptive Multi-View Fusion clustering for single-cell data")
    parser.add_argument("--data_path", required=True, help="Gene x cell CSV matrix")
    parser.add_argument("--label_path", help="Tab-delimited labels for evaluation")
    parser.add_argument("--embedding_path", help="Optional exogenous gene embedding file")
    parser.add_argument("--n_clusters", required=True, type=int)
    parser.add_argument("--n_hvg", default=500, type=int)
    parser.add_argument("--latent_dim", default=20, type=int)
    parser.add_argument("--n_neighbors", default=15, type=int)
    parser.add_argument("--adaptivity", default=0.0, type=float)
    parser.add_argument("--output_path", help="Optional TSV output path for predictions")
    args = parser.parse_args()

    counts, cells, _ = load_expression_matrix(args.data_path)
    embedding = load_gene_embedding(args.embedding_path)
    result = run_amvf(
        counts,
        n_clusters=args.n_clusters,
        embedding=embedding,
        n_hvg=args.n_hvg,
        latent_dim=args.latent_dim,
        n_neighbors=args.n_neighbors,
        adaptivity=args.adaptivity,
    )

    output = pd.DataFrame({"cell": cells, "label": result["prediction"]})
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output_path, sep="\t", index=False)

    print(f"Views: {', '.join(result['view_names'])}")
    print(f"Predicted clusters: {np.bincount(result['prediction'])}")
    if args.label_path:
        truth = load_labels(args.label_path)
        ari = adjusted_rand_score(truth, result["prediction"])
        nmi = normalized_mutual_info_score(truth, result["prediction"])
        print(f"ARI={ari:.4f}, NMI={nmi:.4f}")


if __name__ == "__main__":
    main()
