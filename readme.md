
# WorkOrder-World-Model

A structured world model for work order systems.

This project constructs a heterogeneous graph from tabular work-order data (tasks, assignments, engineers, districts, etc.) to support prediction, simulation, and reasoning tasks. The core goal is to make data semantics explicit, prevent information leakage, and ensure reproducibility when building graph-based models.


## Project Structure

```
├── configs
│   ├── data.yaml
│   └── graph.yaml
├── data
│   ├── analysis
│   │   ├── connectivity.count.csv
│   │   └── connectivity.ratio.csv
│   ├── features_table
│   │   ├── assignment_feat_clean.parquet
│   │   ├── district_feat_clean.parquet
│   │   ├── engineer_feat_clean.parquet
│   │   └── task_feat_clean.parquet
│   ├── graph
│   │   ├── hetero_sdge.pt
│   │   ├── sdge.pt
│   │   └── sdge_pruned.pt
│   ├── processed
│   │   ├── assignments_processed.parquet
│   │   ├── districts_processed.parquet
│   │   ├── engineers_processed.parquet
│   │   └── tasks_processed.parquet
│   └── raw
│       ├── W6ASSIGNMENTS-0.csv
│       ├── W6ASSIGNMENTS-1.csv
│       ├── W6ASSIGNMENTS-10.csv
│       ├── W6ASSIGNMENTS-11.csv
│       ├── W6ASSIGNMENTS-12.csv
│       ├── W6ASSIGNMENTS-13.csv
│       ├── W6ASSIGNMENTS-14.csv
│       ├── W6ASSIGNMENTS-15.csv
│       ├── W6ASSIGNMENTS-16.csv
│       ├── W6ASSIGNMENTS-17.csv
│       ├── W6ASSIGNMENTS-18.csv
│       ├── W6ASSIGNMENTS-19.csv
│       ├── W6ASSIGNMENTS-2.csv
│       ├── W6ASSIGNMENTS-20.csv
│       ├── W6ASSIGNMENTS-21.csv
│       ├── W6ASSIGNMENTS-22.csv
│       ├── W6ASSIGNMENTS-3.csv
│       ├── W6ASSIGNMENTS-4.csv
│       ├── W6ASSIGNMENTS-5.csv
│       ├── W6ASSIGNMENTS-6.csv
│       ├── W6ASSIGNMENTS-7.csv
│       ├── W6ASSIGNMENTS-8.csv
│       ├── W6ASSIGNMENTS-9.csv
│       ├── W6DEPARTMENT-0.csv
│       ├── W6DISTRICTS-0.csv
│       ├── W6ENGINEERS-0.csv
│       ├── W6EQUIPMENT-0.csv
│       ├── W6EQUIPMENT-1.csv
│       ├── W6REGIONS-0.csv
│       ├── W6TASKS-0.csv
│       ├── W6TASKS-1.csv
│       ├── W6TASKS-10.csv
│       ├── W6TASKS-11.csv
│       ├── W6TASKS-12.csv
│       ├── W6TASKS-13.csv
│       ├── W6TASKS-14.csv
│       ├── W6TASKS-15.csv
│       ├── W6TASKS-16.csv
│       ├── W6TASKS-17.csv
│       ├── W6TASKS-18.csv
│       ├── W6TASKS-19.csv
│       ├── W6TASKS-2.csv
│       ├── W6TASKS-20.csv
│       ├── W6TASKS-21.csv
│       ├── W6TASKS-3.csv
│       ├── W6TASKS-4.csv
│       ├── W6TASKS-5.csv
│       ├── W6TASKS-6.csv
│       ├── W6TASKS-7.csv
│       ├── W6TASKS-8.csv
│       ├── W6TASKS-9.csv
│       ├── W6TASK_STATUSES-0.csv
│       └── W6TASK_TYPES-0.csv
├── dockerfile
├── docs
│   └── data_schema.md
├── eda_notbook.ipynb
├── interpret_server
│   ├── app.py
│   └── static
│       └── index.html
├── pipeline_log.txt
├── processing_notebook.ipynb
├── readme.md
├── requirements.txt
├── results
├── scripts
│   ├── generate_graph.sh
│   ├── graph_eda.sh
│   ├── train_gnn.sh
│   ├── visualize_graph.sh
│   └── visualize_interpretation.sh
├── server
│   ├── app.py
│   ├── static
│   │   └── index.html
│   └── utils.py
├── src
│   ├── layer
│   ├── model
│   │   └── gnn.py
│   ├── process
│   │   ├── construct_baseline_graph.py
│   │   ├── construct_graph.py
│   │   ├── feature_engineering.py
│   │   ├── feature_schema.py
│   │   ├── graph_builder.py
│   │   ├── graph_connectivity.py
│   │   ├── prune_graph.py
│   │   ├── structure_graph_builder.py
│   │   └── utils
│   │       ├── convert_columns.py
│   │       ├── filter_raw_data.py
│   │       └── inspect_relation.py
│   └── runner
│       ├── eda.py
│       ├── eval.py
│       ├── interpret_subgraph.py
│       ├── run_gnn.py
│       ├── train.py
│       └── train_kfold.py
├── test
│   ├── __init__.py
│   └── test_graph_construction.py
├── util_function
│   ├── build_schema_from_wow.py
│   ├── how_far_we_go.py
│   └── update_ouliter.py


```
## Setup / Deployment


### 1. Create environment

requirement.txt
```bash
conda create -n wow python==3.12
pip install -U pip
pip3 install torch torchvision
pip install -r requirements.txt
```

dockerfile
```
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
---

## Data Processing

Place raw data files under:

```
data/raw/
```
which should contains csv files named and formated like this:
#### Assignments

- W6ASSIGNMENTS-0.csv
- W6ASSIGNMENTS-1.csv
- W6ASSIGNMENTS-2.csv
- W6ASSIGNMENTS-3.csv
- ...
- W6ASSIGNMENTS-{+d}.csv


#### Tasks

- W6TASKS-0.csv
- W6TASKS-1.csv
- W6TASKS-2.csv
- W6TASKS-3.csv
- ...
- W6TASKS-{+d}.csv


#### Engineers

- W6ENGINEERS-0.csv
- ...
- W6ENGINEERS-{+d}.csv


#### Districts

- W6DISTRICTS-0.csv
- ...
- W6DISTRICTS-{+d}.csv


#### Regions

- W6REGIONS-0.csv
- ...
- W6REGIONS-{+d}.csv


#### Departments

- W6DEPARTMENT-0.csv
- ...
- W6DEPARTMENT-{+d}.csv


#### Equipment

- W6EQUIPMENT-0.csv
- W6EQUIPMENT-1.csv
- ...
- W6EQUIPMENT-{+d}.csv


#### Task Statuses

- W6TASK_STATUSES-0.csv
- ...
- W6TASK_STATUSES-{+d}.csv


#### Task Types

- W6TASK_TYPES-0.csv
- ...
- W6TASK_TYPES-{+d}.csv




---

## Build World Model (Graph)

The world model is represented as a PyTorch Geometric `HeteroData` object.

### Build graph

```bash
bash scripts/generate_graph.sh
```
### Graph Statistic Analysis

#### connectivity
```
bash scripts/graph_eda.sh
```
EDA result will put into data/analysis

### Graph Visualize
```
bash scripts/visualize_graph.sh
```

### Training
```
bash scripts/train_gnn.sh
```

### Prediction Interpertation(after training)
```
python -m src.runner interpert_subgraph
bash scripts/visualize_interpretation.sh
```
