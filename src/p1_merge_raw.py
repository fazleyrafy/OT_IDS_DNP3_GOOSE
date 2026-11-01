"""Step 1: merge the 13 per protocol raw files into one window aligned master table.

Input : data/raw/*.csv
Output: data/processed/master_table.csv  (one row per window per protocol flow)
Run   : python src/p1_merge_raw.py
"""
import pandas as pd

from config import MASTER_TABLE, PROCESSED_DIR, RAW_DIR, RAW_FILES


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def main() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for file_name, (protocol, src, dst) in RAW_FILES.items():
        path = RAW_DIR / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")
        df = clean_column_names(pd.read_csv(path))
        df["protocol"] = protocol
        df["src"] = src
        df["dst"] = dst
        df["edge_id"] = f"{protocol}_{src}_to_{dst}"
        df["file_name"] = file_name
        frames.append(df)

    master = pd.concat(frames, ignore_index=True, sort=False)

    # Integrity checks: every file spans the same windows, no duplicate (window, flow) rows
    span = master.groupby("file_name")["Window_Num"].agg(["min", "max", "nunique"])
    assert span["nunique"].nunique() == 1, f"Files are not window aligned:\n{span}"
    dups = master.duplicated(subset=["Window_Num", "edge_id"]).sum()
    assert dups == 0, f"{dups} duplicate (Window_Num, edge_id) rows"

    master = master.sort_values(["Window_Num", "protocol", "src", "dst"]).reset_index(drop=True)

    meta_cols = ["Window_Num", "protocol", "src", "dst", "edge_id", "file_name"]
    end_cols = ["Label", "Notes"]
    feature_cols = [c for c in master.columns if c not in meta_cols + end_cols]
    master = master[meta_cols + feature_cols + end_cols]

    master.to_csv(MASTER_TABLE, index=False)

    print(f"Files merged      : {len(frames)}")
    print(f"Master table      : {master.shape}  -> {MASTER_TABLE}")
    print(f"Windows           : {master['Window_Num'].nunique()} "
          f"({master['Window_Num'].min()} to {master['Window_Num'].max()})")
    print(f"Rows per protocol : {master['protocol'].value_counts().to_dict()}")
    print(f"Label counts      : {master['Label'].value_counts().to_dict()}")
    return master


if __name__ == "__main__":
    main()
