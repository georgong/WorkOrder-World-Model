
# WorkOrder-World-Model

A structured world model for work order systems.

## Problem Description

Utility companies like SDG&E coordinates thousands of field tasks across technicians, districts, and time constraints, making daily scheduling a complex system-level problem. However, existing scheduling tools primarily focus on generating feasible plans and provide limited visibility into systemic risks such as workload imbalance, task delays, and regional congestion. 

This project develops a graph-based modeling pipeline that analyzes historical schedules to uncover operational patterns and provide actionable insights into scheduling performance.

## Setup / Deployment


### Environment setup with conda

```bash
conda create -n wow python==3.12
pip install -U pip
pip3 install torch torchvision
pip install -r requirements.txt
```


### Environment setup with docker

```bash
# Build the image
docker build -t workorder-world-model .

# Run the API server
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results workorder-world-model

# Run an interactive shell
docker run -it -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results workorder-world-model /bin/bash
```



## DATA Configuration

All variable-level decisions are centralized in:

```
config/schema.yaml
```


This YAML defines, **for each dataset and each variable**:

### Variable Fields

- **`dtype`**  
  Storage / parsing type (e.g., `Float64`, `string`, `datetime64[ns]`).

- **`key`**  
  Whether the column is a unique identifier or join key  
  (e.g., `ASSIGNEDENGINEERS`).

- **`mask`**  
  Whether the variable is **not available at inference time** or is
  leakage-prone and must be excluded from model-visible features  
  (e.g., `FINISHTIME`).

- **`trait_type`**  
  How the variable is used in the graph pipeline:

  - `node` – node attribute (stored in node feature tables)  
  - `edge` – edge attribute (stored in edge feature tables)  
  - `null` – not used as a graph feature (keys / metadata only)

- **Outlier policy (optional)**

  - **`outlier_type`** – how to interpret outliers (e.g., `datetime`)  
  - **`outlier`** – valid range or thresholds  
    (e.g., `["1980-01-01", "2030-01-01"]`)
  
- **Edge construction policy (optional)**
  - **`edge_group`** – how to group a set of nodes to construct edges (e.g., `weekday`)
  - **`edge_construct`** – how to connect edges within a group (e.g., `context_node`, `neighbor`, `pairwise`)
    - `context_node`: connecting all nodes within a group to a central node
    - `neighbor`: for each node, connect `k` neighbors node 
    - `pairwise`: connect pairwise node within a group 
---

## Data Processing

The raw data is exported from SDG&E CLICK system that contains historical scheduling activity records. Data is classified in compliance with SDG&E data privacy.

Place raw data files under:

```
data/raw/
```
which should contains csv files such as W6ASSIGNMENTS-0.csv, W6TASKS-0.csv, W6ENGINEERS-0.csv, etc.

### Training-data EDA report

```bash
bash scripts/generate_eda_report.sh
```
**Expected output:** `data/analysis/eda_report.txt` — text summary of feature scales, graph statistics, and missing/outlier checks computed by `src.runner.eda`.


---

## Build World Model (Graph)

The world model is represented as a PyTorch Geometric `HeteroData` object.

### Build graph

```bash
bash scripts/generate_graph.sh
```
**Expected output:** `data/graph/sdge.pt` — a serialized PyTorch Geometric `HeteroData` object containing all node types, edge types, and their respective feature tensors.

### Graph Statistic Analysis

#### connectivity
```
bash scripts/graph_eda.sh
```
**Expected output:** `data/analysis/connectivity.count.csv` and `data/analysis/connectivity.ratio.csv`.

### Graph Visualize
```
bash scripts/visualize_graph.sh
```
**Expected output:** An interactive HTML visualization of the heterogeneous graph opened in the browser via the local server.

### Training
```
bash scripts/train_kfold.sh
```
**Expected output:** `data/graph/sdge_pruned.pt` — pruned graph with low-degree nodes removed; `runs/checkpoints/` — saved model checkpoints; training metrics (loss, MAE) logged to W&B.

### Prediction Interpertation(after training)
```
python -m src.runner.interpret_subgraph
bash scripts/visualize_interpretation.sh
```
**Expected output:** `runs/interpret/` — per-assignment subgraph JSONs with feature attribution scores; an interactive HTML visualization of interpretation results served via the local interpret server.

### Hidden-layer PCA by neighbor group (task type / engineer / districts / departments)
After training, run PCA on checkpoint hidden activations over the dataset, grouped by neighbor-derived labels (e.g. engineer, task type, districts, departments):
```
python -m src.runner.pca_weights --pt path/to/graph.pt --ckpt path/to/checkpoint.pt [--split val] [--max_samples 5000] [--out_dir runs/pca_weights]
```
Output: `runs/pca_weights/pca_*_by_*.png` and `pca_summary.json`.

**Interactive Plotly (single HTML with dropdown):**
```
python -m src.runner.pca_weights --pt path/to/graph.pt --ckpt path/to/checkpoint.pt --plotly [--open]
```
Generates `runs/pca_weights/pca_interactive.html`. Use the dropdown to switch layer × group (engineers, task_types, districts, etc.). `--open` opens it in your default browser.

### t-SNE visualization of hidden representations
After training, compute t-SNE embeddings for target-node hidden states and launch an interactive viewer:
```bash
bash scripts/visualize_tsne.sh
```
This runs `src.runner.tsne_weights` to generate `runs/tsne_weights/tsne_nodes.json` and then serves an interactive Plotly UI from `src.runner.render_tsne`.

### Model comparison & prediction analysis
To compare GraphSAGE / MLP / LightGBM performance and analyze hard cases:
```bash
bash scripts/analysis_model.sh
```
**Expected output:** `runs/compare_model/compare_three.png`, `runs/compare_model/predictions.json`, and figures under `runs/analysis_model/` (prediction vs truth, hard-case analysis, metrics bar plot).

## Model Application

See the README for setup and files: [WOW-dashboard/README.md](./WOW-dashboard/README.md)

--- 
## Project Structure

```
WorkOrder-World-Model/
├── configs/
│   ├── data.yaml                        # Dataset variable schema (dtype, mask, outlier policy)
│   └── graph.yaml                       # Graph construction config (nodes, edges, features)
│
├── data/
│   ├── raw/                             # Raw CSVs exported from SDG&E CLICK system
│   │   ├── ...
│   ├── processed/                       # Cleaned & merged parquet tables
│   │   ├── ...
│   ├── features_table/                  # Per-entity feature tables
│   │   ├── ...
│   ├── graph/                           # Serialized PyG HeteroData graphs
│   │   ├── sdge.pt                      # Full constructed graph
│   │   ├── sdge_pruned.pt               # Pruned graph (low-degree nodes removed)
│   │   └── hetero_sdge.pt               # Alternate graph variant
│   └── analysis/                        # EDA & connectivity outputs
│       ├── ...
│
├── src/
│   ├── process/                         # Graph & data pipeline
│   │   ├── structure_graph_builder.py   # Core HeteroData builder from CSVs
│   │   ├── construct_graph.py           # Legacy graph construction
│   │   ├── construct_baseline_graph.py  # Baseline graph variant
│   │   ├── graph_builder.py             # GraphBuilder orchestration
│   │   ├── graph_connectivity.py        # Connectivity heatmaps & metapath analysis
│   │   ├── prune_graph.py               # Prune low-degree nodes
│   │   ├── feature_engineering.py       # Feature extraction & transformation
│   │   ├── feature_schema.py            # Schema parsing utilities
│   │   └── utils/
│   │       ├── convert_columns.py
│   │       ├── filter_raw_data.py
│   │       └── inspect_relation.py
│   ├── model/
│   │   └── gnn.py                       # GNN model definitions
│   └── runner/                          # Experiment entrypoints
│       ├── train.py                     # GNN training loop (W&B logging)
│       ├── train_kfold.py               # K-fold cross-validation training
│       ├── eval.py                      # Checkpoint evaluation
│       ├── interpret_subgraph.py        # Feature attribution (grad×input, IG, occlusion)
│       ├── eda.py                       # EDA report generation
│       └── run_gnn.py                   # Inference runner
│
├── WOW-dashboard/                       # Web application
│   ├── app/                             # Next.js App Router (layout, pages)
│   ├── components/                      # React UI components
│   │   ├── Dashboard.tsx
│   │   ├── MetricsCards.tsx
│   │   ├── RiskTable.tsx
│   │   ├── Charts.tsx
│   │   ├── GraphVisualizer.tsx
│   │   ├── UploadPanel.tsx
│   │   └── HeaderActions.tsx
│   ├── lib/                             # Shared TS utilities & API client
│   │   ├── api.ts
│   │   ├── types.ts
│   │   └── header-context.tsx
│   ├── api/                             # FastAPI Python backend (Vercel serverless)
│   │   ├── index.py                     # API endpoints: /predict, /demo, /health, /graph
│   │   └── inference/
│   │       ├── graph_inference_api.py   # Upload → graph → inference pipeline
│   │       ├── structure_graph_builder.py
│   │       ├── model.py
│   │       ├── feature_engineering.py
│   │       ├── feature_schema.py
│   │       └── schema.py
│
├── test/                                # Pytest unit tests (134 tests)
│   ├── test_prune_graph.py
│   ├── test_graph_connectivity.py
│   ├── test_train_eval_utils.py
│   ├── test_structure_graph_builder.py
│   ├── test_interpret_utils.py
│   └── test_graph_construction.py
│
├── scripts/                             # Shell script entrypoints
│   ├── generate_graph.sh
│   ├── generate_eda_report.sh
│   ├── graph_eda.sh
│   ├── train_gnn.sh
│   ├── train_kfold.sh
│   ├── visualize_graph.sh
│   ├── visualize_interpretation.sh
│   ├── visualize_tsne.sh
│   └── analysis_model.sh
│
├── docs/
│   ├── data_schema.md                   # Field-level data dictionary
│   ├── eda_report_analysis.md
│   └── architecture_mermaid.md          # System architecture diagram
│
├── server/                              # Local graph visualization server
│   ├── app.py
│   └── utils.py
├── interpret_server/                    # Local interpretation visualization server
│   └── app.py
│
├── requirements.txt
├── dockerfile
└── readme.md
```