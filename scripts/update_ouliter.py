import yaml
import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")

def concat_parts(files):
    dfs = []
    for f in sorted(files, key=lambda x: x.name):
        try:
            df = pd.read_csv(f, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(f, low_memory=False, encoding="latin1")
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True, sort=True) if dfs else pd.DataFrame()

if __name__ == "__main__":
    yaml_path = "configs/data.yaml"

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    for dataset_name, dataset_info in data["datasets"].items():
        # Find all parts for this dataset
        base_name = Path(dataset_info["file"]).stem
        part_files = list(RAW_DATA_DIR.glob(f"{base_name}-*.csv"))
        if not part_files:
            # Fallback: try the single merged file
            merged_file = RAW_DATA_DIR / f"{base_name}.csv"
            if merged_file.exists():
                part_files = [merged_file]
            else:
                print(f"No CSV parts or merged file found for: {base_name}")
                continue
        
        print(part_files)
        df = concat_parts(part_files)
        print(df)

        for col, meta in dataset_info["variables"].items():
            dtype = meta.get("dtype", "")
            if dtype in ("Float64", "Int64"):
                if col in df.columns:

                    s = pd.to_numeric(df[col], errors="coerce")
                    if s.notna().sum() > 0:
                        low = float(s.quantile(0.001))
                        high = float(s.quantile(0.999))
                        meta["outlier"] = [low, high]
                    else:
                        meta["outlier"] = [None, None]
                else:
                    meta["outlier"] = [None, None]
            else:
                meta["outlier"] = [None, None]

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"Updated outlier bounds in {yaml_path}")

