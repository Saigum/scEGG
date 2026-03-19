
# scEGG: Exogenous Gene-guided Single-Cell Deep Clustering Framework

![Franework](https://github.com/DayuHuu/scEGA/blob/master/scEGG_framework.png)
**Description:**

[![Python](https://img.shields.io/badge/Python-3.7.0-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-orange.svg)](https://pytorch.org/)
[![Journal](https://img.shields.io/badge/Briefings_in_Bioinformatics-bbae483-darkred)](https://academic.oup.com/bib)

## 📖 Overview

**scEGG** is a novel deep clustering framework designed specifically for single-cell transcriptomic analysis.

Traditional methods often rely solely on endogenous cellular features. **scEGG** advances this by simultaneously integrating **cell features** and **exogenous gene features**. By aligning and fusing these dual sources of information during the clustering process, the model generates a more discriminative representation, leading to superior clustering performance.

### Model Framework
![scEGG Framework](scEGG_framework.png)

> **Paper:** This work is published in *Briefings in Bioinformatics* (2024).


## 🛠 Requirements

Please ensure your environment meets the following dependencies:

* **Python** == 3.7.0
* **Torch** == 1.13.1
* **NumPy** == 1.21.6
* **Pandas** == 1.1.5
* **SciPy** == 1.7.3
* **Scikit-learn** == 0.22.2

### Installation
You can install the required packages using pip:

```bash
pip install torch==1.13.1 numpy==1.21.6 pandas==1.1.5 scipy==1.7.3 scikit-learn==0.22.2
````

-----

## 📂 Data Availability

We evaluated scEGG on several benchmark single-cell datasets. The original data sources can be accessed via the links below:

| Dataset | Source / Accession | Link |
| :--- | :--- | :--- |
| **Darmanis** | PubMed 26060301 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/26060301/) |
| **Bjorklund**| GSE70580 | [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70580) |
| **Sun** | GSE128066 | [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066) |
| **Marques** | PubMed 30078729 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/30078729/) |
| **Zeisel** | PubMed 25700174 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/25700174/) |
| **Fink** | PubMed 35914526 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35914526/) |

-----

## 🧬 Gene Representation Construction

To fully utilize the **scEGG** framework, it is essential to construct exogenous gene representations before training the model. These embeddings introduce external biological knowledge (e.g., from PPI networks) to guide the clustering process.

Detailed instructions and scripts for generating these embeddings can be found in the guide:  
👉 [**Produce_Gene_Emb.md**](https://github.com/DayuHuu/scEGG/blob/master/data/Produce_Gene_Emb.md)

### Example: Generating Embeddings for Bjorklund
To run scEGG on the **Bjorklund** dataset, you must first generate the corresponding gene embedding file (`Bjorklund.emb`).

1.  **Prepare Data:** Ensure the Bjorklund dataset is correctly placed in the data directory.
2.  **Run Generation Script:** Follow the instructions in `Produce_Gene_Emb.md` to process the gene interaction network.
3.  **Output:** The process will yield a `Bjorklund.emb` file.

> **Note:** Ensure the generated `.emb` file matches the dataset name specified in your configuration (e.g., `Bjorklund`) so the model can load it automatically.

-----

## 🚀 Usage

### 1\. Configuration

The model parameters can be configured via command-line arguments. Key arguments (e.g., dataset name, number of clusters) are defined in the parser:
# Add other arguments as needed...
```

**Implement:**

```python
# Example configuration
parser.add_argument('--dataset_str', default='Bjorklund', type=str, help='Name of the target dataset')
parser.add_argument('--n_clusters', default=4, type=int, help='Expected number of clusters')
# ... additional arguments ...
```

### 2\. Execution

To run the scEGG model with the default configuration:

```bash
python run_scEGG.py
```

## New method: AMVF

I added an improved clustering method in `code/run_amvf.py` called **Adaptive Multi-View Fusion (AMVF)**.

- It combines an expression view, a sparse binary-program view, a TF-IDF rarity view, and an optional gene-embedding view.
- It uses a stronger stacked multi-view latent as the default representation, with only light confidence-based calibration.
- It runs with standard `numpy`/`pandas`/`scikit-learn` dependencies.

Example:

```python
python code/run_amvf.py \
  --data_path data/Bjorklund/Bjorklund_data.csv \
  --label_path data/Bjorklund/label.ann \
  --embedding_path data/Bjorklund/Bjorklund.emb \
  --n_clusters 4 \
  --output_path result/bjorklund_amvf_predictions.tsv
```

Benchmark the classical baselines against AMVF on Bjorklund:

```python
python code/benchmark_bjorklund.py
```

Additional notes:

- `code/run_scEGG.py` now runs on CPU.
- `code/Nmetrics.py` no longer depends on the external `munkres` package.
- Method notes are in `docs/literature_review.md` and `docs/amvf_method.md`.

## Standardized cell-type mappings

I added `code/build_standardized_mappings.py` to standardize cell identifier to label metadata across the datasets in this workspace.

It writes:

- `result/standardized_celltype_mappings.tsv` for the combined table.
- `result/standardized_celltype_mapping_summary.tsv` for dataset coverage.
- per-dataset files under `result/standardized_mappings/`.

The standardized columns are:

- `dataset_id`
- `sample_id`
- `cell_id`
- `cell_barcode`
- `cell_type`
- `label_kind`
- `label_source`
- `label_status`

Run it with:

```python
python code/build_standardized_mappings.py
```

## Cross-dataset benchmark

I added `code/benchmark_other_datasets.py` to run the same baseline family used in the Bjorklund benchmark on the other labeled datasets in `data/datasets/`.

- Methods: `KMeans-PCA`, `Agglomerative-PCA`, `Spectral-kNN`, and `AMVF`.
- Output: `result/other_dataset_benchmark.tsv`.
- For larger datasets, the script uses a deterministic stratified subset so all baselines remain comparable and feasible.

Run it with:

```python
python code/benchmark_other_datasets.py --max_cells 2000
```
