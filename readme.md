
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
pip install -U pip
pip install -r requirements.txt
```

dockerfile
```
```



## Configuration

All variable-level decisions are centralized in:

```
src/config/schema.yaml
```

This yaml file defines:

* Variable role (feature / label / key / metadata / forbidden)
* Data type and semantic type
* Missing value handling
* Outlier detection and actions
* Cardinality strategies for categorical variables
* Leakage risk and inference-time availability

Code should never hard-code these rules.

---

## Data Processing

Place raw data files under:

```
data/raw/
```



The output in `data/processed/` is the only input allowed for graph construction.

---

## Build World Model (Graph)

The world model is represented as a PyTorch Geometric `HeteroData` object.

### Build graph

```bash

```

---

## Documentation

Documentation is maintained separately from code.

* `docs/data_schema.md`

  * Table-level schema
  * Primary keys and join keys
  * Temporal semantics of fields
  * Label definitions
  * Known leakage fields


