"""Step 3: build PyTorch Geometric graph snapshots, T=3 sequences, scaling, and splits.

* One graph per 5 s window: 6 nodes (x: 7 features), 4 directed edges (edge_attr: 24 features).
* Sliding windows of SEQUENCE_LENGTH consecutive graphs, built inside each split.
* Sequence label priority: any dynamic attack (2) > any deterministic attack (1) > normal (0).
* RobustScaler fitted on the TRAIN split only, applied to train/val/test.

Input : data/processed/edge_features.csv, node_features.csv
Output: data/processed/sequences/{train,val,test}_sequences.pt
        data/processed/scalers/{node,edge}_scaler.pkl
        data/processed/scaled_csv/scaled_sequence_{node,edge}_features_{split}.csv (for inspection)
Needs : torch, torch_geometric, scikit-learn, joblib  (see requirements.txt)
Run   : python src/p3_build_sequences.py
"""
import copy
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Data

from config import (EDGE_FEATURE_COLS, EDGE_SCALE_COLS, EDGE_TABLE, NODE_FEATURE_COLS,
                    NODE_SCALE_COLS, NODE_TABLE, SCALED_CSV_DIR, SCALER_DIR,
                    SEQUENCE_DIR, SEQUENCE_LENGTH, TEST_RANGE, TRAIN_RANGE, VAL_RANGE)


def build_graph_for_window(window_num, node_df, edge_df):
    node_win = node_df[node_df["Window_Num"] == window_num].sort_values("node_id").reset_index(drop=True)
    edge_win = edge_df[edge_df["Window_Num"] == window_num].sort_values(["src_id", "dst_id"]).reset_index(drop=True)

    node_id_to_idx = {nid: i for i, nid in enumerate(node_win["node_id"].tolist())}
    x = torch.tensor(node_win[NODE_FEATURE_COLS].to_numpy(), dtype=torch.float)
    src_idx = edge_win["src_id"].map(node_id_to_idx).to_numpy()
    dst_idx = edge_win["dst_id"].map(node_id_to_idx).to_numpy()
    edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)
    edge_attr = torch.tensor(edge_win[EDGE_FEATURE_COLS].to_numpy(), dtype=torch.float)

    # Window label = max node label (node table carries the label on GOOSE publisher rows)
    y = torch.tensor([int(node_win["Label"].max())], dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    graph.window_num = int(window_num)
    return graph


def get_sequence_label(graph_labels):
    if 2 in graph_labels:
        return 2
    if 1 in graph_labels:
        return 1
    return 0


def build_sequences(graph_block, seq_len):
    sequences = []
    for i in range(len(graph_block) - seq_len + 1):
        seq_graphs = graph_block[i:i + seq_len]
        labels = [int(g.y.item()) for g in seq_graphs]
        sequences.append({
            "graphs": seq_graphs,
            "windows": [g.window_num for g in seq_graphs],
            "graph_labels": labels,
            "sequence_label": get_sequence_label(labels),
        })
    return sequences


def apply_scaling(sequences, node_scaler, edge_scaler, node_idx, edge_idx):
    scaled = copy.deepcopy(sequences)
    for seq in scaled:
        for g in seq["graphs"]:
            x_np = g.x.numpy().copy()
            e_np = g.edge_attr.numpy().copy()
            x_np[:, node_idx] = node_scaler.transform(x_np[:, node_idx])
            e_np[:, edge_idx] = edge_scaler.transform(e_np[:, edge_idx])
            g.x = torch.tensor(x_np, dtype=torch.float)
            g.edge_attr = torch.tensor(e_np, dtype=torch.float)
    return scaled


def export_scaled_csv(sequences, split_name):
    node_rows, edge_rows = [], []
    for s_idx, seq in enumerate(sequences):
        for t, g in enumerate(seq["graphs"]):
            base = {"sequence_index": s_idx, "time_step_in_sequence": t,
                    "window_num": seq["windows"][t], "sequence_label": seq["sequence_label"]}
            x_np, e_np, ei = g.x.numpy(), g.edge_attr.numpy(), g.edge_index.numpy()
            for n in range(x_np.shape[0]):
                row = {**base, "node_idx": n}
                row.update({c: x_np[n, j] for j, c in enumerate(NODE_FEATURE_COLS)})
                node_rows.append(row)
            for e in range(e_np.shape[0]):
                row = {**base, "src_idx": ei[0, e], "dst_idx": ei[1, e]}
                row.update({c: e_np[e, j] for j, c in enumerate(EDGE_FEATURE_COLS)})
                edge_rows.append(row)
    pd.DataFrame(node_rows).to_csv(SCALED_CSV_DIR / f"scaled_sequence_node_features_{split_name}.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(SCALED_CSV_DIR / f"scaled_sequence_edge_features_{split_name}.csv", index=False)


def main():
    for d in (SEQUENCE_DIR, SCALER_DIR, SCALED_CSV_DIR):
        d.mkdir(parents=True, exist_ok=True)

    edge_df = pd.read_csv(EDGE_TABLE)
    node_df = pd.read_csv(NODE_TABLE)
    windows = sorted(node_df["Window_Num"].unique())
    assert windows == sorted(edge_df["Window_Num"].unique()), "Node and edge tables are not aligned"

    graph_list = [build_graph_for_window(w, node_df, edge_df) for w in windows]
    print(f"Graphs built: {len(graph_list)} | labels: {dict(Counter(int(g.y.item()) for g in graph_list))}")

    def in_range(g, r):
        return r[0] <= g.window_num <= r[1]

    splits = {
        "train": build_sequences([g for g in graph_list if in_range(g, TRAIN_RANGE)], SEQUENCE_LENGTH),
        "val": build_sequences([g for g in graph_list if in_range(g, VAL_RANGE)], SEQUENCE_LENGTH),
        "test": build_sequences([g for g in graph_list if in_range(g, TEST_RANGE)], SEQUENCE_LENGTH),
    }
    for name, seqs in splits.items():
        print(f"{name:5s}: {len(seqs)} sequences | {dict(Counter(s['sequence_label'] for s in seqs))}")

    # Fit scalers on TRAIN only
    node_idx = [NODE_FEATURE_COLS.index(c) for c in NODE_SCALE_COLS]
    edge_idx = [EDGE_FEATURE_COLS.index(c) for c in EDGE_SCALE_COLS]
    node_data = np.vstack([g.x.numpy()[:, node_idx] for s in splits["train"] for g in s["graphs"]])
    edge_data = np.vstack([g.edge_attr.numpy()[:, edge_idx] for s in splits["train"] for g in s["graphs"]])
    node_scaler = RobustScaler().fit(node_data)
    edge_scaler = RobustScaler().fit(edge_data)
    joblib.dump(node_scaler, SCALER_DIR / "node_scaler.pkl")
    joblib.dump(edge_scaler, SCALER_DIR / "edge_scaler.pkl")

    for name, seqs in splits.items():
        scaled = apply_scaling(seqs, node_scaler, edge_scaler, node_idx, edge_idx)
        torch.save(scaled, SEQUENCE_DIR / f"{name}_sequences.pt")
        export_scaled_csv(scaled, name)
        print(f"saved {name}: {SEQUENCE_DIR / (name + '_sequences.pt')}")


if __name__ == "__main__":
    main()
