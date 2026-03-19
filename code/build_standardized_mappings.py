import gzip
import io
import tarfile
from pathlib import Path

import pandas as pd


RESULT_DIR = Path("result")
MAPPING_DIR = RESULT_DIR / "standardized_mappings"
STANDARD_COLUMNS = [
    "dataset_id",
    "sample_id",
    "cell_id",
    "cell_barcode",
    "cell_type",
    "label_kind",
    "label_source",
    "label_status",
]


def make_rows(dataset_id, sample_id, cell_ids, cell_types=None, cell_barcodes=None, label_kind="cell_type", label_source="", label_status="available"):
    if cell_barcodes is None:
        cell_barcodes = cell_ids
    if cell_types is None:
        cell_types = [""] * len(cell_ids)
    rows = []
    for cell_id, cell_barcode, cell_type in zip(cell_ids, cell_barcodes, cell_types):
        rows.append({
            "dataset_id": dataset_id,
            "sample_id": sample_id,
            "cell_id": str(cell_id),
            "cell_barcode": str(cell_barcode),
            "cell_type": "" if pd.isna(cell_type) else str(cell_type),
            "label_kind": label_kind,
            "label_source": label_source,
            "label_status": label_status,
        })
    return rows


def build_bjorklund_mapping():
    labels = pd.read_csv("data/Bjorklund/label.ann", sep="\t")
    cell_types = labels["cell"].str.extract(r"_(NK|ILC\d)_expression\.txt$")[0].fillna(labels["label"].astype(str))
    return make_rows(
        dataset_id="Bjorklund",
        sample_id="Bjorklund",
        cell_ids=labels["cell"].astype(str).tolist(),
        cell_types=cell_types.tolist(),
        label_kind="cell_type",
        label_source="data/Bjorklund/label.ann",
    )


def build_guo_mapping():
    labels = pd.read_csv("data/datasets/Guo/subtype.ann", sep="\t")
    labels.columns = [column.strip() for column in labels.columns]
    return make_rows(
        dataset_id="Guo",
        sample_id="Guo",
        cell_ids=labels["UniqueCell_ID"].astype(str).tolist(),
        cell_types=labels["majorCluster"].astype(str).tolist(),
        label_kind="majorCluster",
        label_source="data/datasets/Guo/subtype.ann",
    )


def build_pbmc_mapping():
    barcodes = pd.read_csv("data/datasets/pbmc/pbmc6k_matrices/barcodes.tsv", sep="\t", header=None, names=["cell_id"])
    with tarfile.open("data/datasets/pbmc/pbmc6k_analysis.tar.gz", "r:gz") as handle:
        with handle.extractfile("analysis_csv/kmeans/10_clusters/clusters.csv") as member:
            clusters = pd.read_csv(member)
    merged = barcodes.merge(clusters, how="left", left_on="cell_id", right_on="Barcode")
    cell_types = merged["Cluster"].apply(lambda value: f"cluster_{int(value)}" if pd.notna(value) else "")
    return make_rows(
        dataset_id="pbmc6k",
        sample_id="pbmc6k",
        cell_ids=merged["cell_id"].astype(str).tolist(),
        cell_types=cell_types.tolist(),
        label_kind="kmeans_10x_cluster",
        label_source="data/datasets/pbmc/pbmc6k_analysis.tar.gz:analysis_csv/kmeans/10_clusters/clusters.csv",
        label_status="available",
    )


def build_pbmc_perturb_mapping():
    rows = []

    batch1 = pd.read_csv("data/datasets/pbmc_perturb/sample2.1/GSE96583_batch1.total.tsne.df.tsv.gz", sep="\t")
    for sample_id, batch_name, barcode_path in [
        ("sampleA", "A", "data/datasets/pbmc_perturb/sampleA/GSM2560245_barcodes.tsv.gz"),
        ("sampleB", "B", "data/datasets/pbmc_perturb/sampleB/GSM2560246_barcodes.tsv.gz"),
        ("sampleC", "C", "data/datasets/pbmc_perturb/sampleC/GSM2560247_barcodes.tsv.gz"),
    ]:
        frame = batch1[batch1["batch"] == batch_name].reset_index(drop=True)
        with gzip.open(barcode_path, "rt") as handle:
            barcodes = [line.strip() for line in handle]
        if len(barcodes) != len(frame):
            raise ValueError(f"Barcode count mismatch for {sample_id}: {len(barcodes)} vs {len(frame)}")
        rows.extend(make_rows(
            dataset_id="pbmc_perturb",
            sample_id=sample_id,
            cell_ids=barcodes,
            cell_types=frame["cell.type"].astype(str).tolist(),
            label_kind="cell_type",
            label_source="data/datasets/pbmc_perturb/sample2.1/GSE96583_batch1.total.tsne.df.tsv.gz (row-order within batch)",
        ))

    batch2 = pd.read_csv("data/datasets/pbmc_perturb/sample2.2/GSE96583_batch2.total.tsne.df.tsv.gz", sep="\t")
    for sample_id, stim_value, barcode_path in [
        ("sample2.1", "ctrl", "data/datasets/pbmc_perturb/sample2.1/GSM2560248_barcodes.tsv.gz"),
        ("sample2.2", "stim", "data/datasets/pbmc_perturb/sample2.2/GSM2560249_barcodes.tsv.gz"),
    ]:
        frame = batch2[batch2["stim"] == stim_value].reset_index(drop=True)
        with gzip.open(barcode_path, "rt") as handle:
            barcodes = [line.strip() for line in handle]
        if len(barcodes) != len(frame):
            raise ValueError(f"Barcode count mismatch for {sample_id}: {len(barcodes)} vs {len(frame)}")
        rows.extend(make_rows(
            dataset_id="pbmc_perturb",
            sample_id=sample_id,
            cell_ids=barcodes,
            cell_types=frame["cell"].astype(str).tolist(),
            label_kind="cell_type",
            label_source="data/datasets/pbmc_perturb/sample2.2/GSE96583_batch2.total.tsne.df.tsv.gz (row-order within stim group)",
        ))

    return rows


def build_brown_mapping():
    barcodes = pd.read_csv("data/datasets/Brown/cell_barcodes.tsv", sep="\t")
    return make_rows(
        dataset_id="Brown",
        sample_id="mouse_spleen",
        cell_ids=barcodes["cell_ID"].astype(str).tolist(),
        cell_barcodes=barcodes["cell_barcode"].astype(str).tolist(),
        label_kind="",
        label_source="data/datasets/Brown/cell_barcodes.tsv",
        label_status="unavailable",
    )


def build_habib_mapping():
    with open("data/datasets/Habib/GSE104525_Mouse_Processed_GTEx_Data.DGE.UMI-Counts.txt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    return make_rows(
        dataset_id="Habib",
        sample_id="GTEx_Mouse",
        cell_ids=header[1:],
        label_kind="",
        label_source="data/datasets/Habib/GSE104525_Mouse_Processed_GTEx_Data.DGE.UMI-Counts.txt",
        label_status="unavailable",
    )


def build_marques_mapping():
    rows = []
    tar_path = "data/datasets/Marques/GSE95194_RAW.tar"
    with tarfile.open(tar_path, "r") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".tab.gz"):
                continue
            raw = archive.extractfile(member).read()
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
                header = handle.readline().decode("utf-8", errors="replace").rstrip("\n").split("\t")
            rows.extend(make_rows(
                dataset_id="Marques",
                sample_id=member.name.replace(".tab.gz", ""),
                cell_ids=[value for value in header[1:] if value],
                label_kind="",
                label_source=f"{tar_path}:{member.name}",
                label_status="unavailable",
            ))
    return rows


def write_outputs(frame):
    RESULT_DIR.mkdir(exist_ok=True)
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)

    combined_path = RESULT_DIR / "standardized_celltype_mappings.tsv"
    frame.to_csv(combined_path, sep="\t", index=False)

    for (dataset_id, sample_id), part in frame.groupby(["dataset_id", "sample_id"], sort=True):
        safe_name = f"{dataset_id}__{sample_id}".replace("/", "_")
        part.to_csv(MAPPING_DIR / f"{safe_name}.tsv", sep="\t", index=False)

    summary = (
        frame.assign(has_label=frame["label_status"].eq("available") & frame["cell_type"].astype(str).ne(""))
        .groupby(["dataset_id", "sample_id", "label_status", "label_kind"], dropna=False)
        .agg(n_cells=("cell_id", "size"), labeled_cells=("has_label", "sum"))
        .reset_index()
        .sort_values(["dataset_id", "sample_id"])
    )
    summary_path = RESULT_DIR / "standardized_celltype_mapping_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"Saved combined mappings to {combined_path}")
    print(f"Saved mapping summary to {summary_path}")


def main():
    rows = []
    rows.extend(build_bjorklund_mapping())
    rows.extend(build_guo_mapping())
    rows.extend(build_pbmc_mapping())
    rows.extend(build_pbmc_perturb_mapping())
    rows.extend(build_brown_mapping())
    rows.extend(build_habib_mapping())
    rows.extend(build_marques_mapping())

    frame = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    write_outputs(frame)


if __name__ == "__main__":
    main()
