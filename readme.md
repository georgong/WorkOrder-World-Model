
# WorkOrder-World-Model

A structured world model for work order systems.

This project constructs a heterogeneous graph from tabular work-order data (tasks, assignments, engineers, districts, etc.) to support prediction, simulation, and reasoning tasks. The core goal is to make data semantics explicit, prevent information leakage, and ensure reproducibility when building graph-based models.


## Project Structure

```



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
