"""Step 2: turn the master table into graph ready node and edge feature tables.

* Rows with a real destination (DNP3 + TCP) become EDGE records (device to device links).
* Rows with dst == MULTI (GOOSE multicast) become NODE records (publisher side behavior).
* Raw device names are mapped to 6 canonical nodes; every node appears in every window.

Input : data/processed/master_table.csv
Output: data/processed/edge_features.csv   (4 edges x 1681 windows)
        data/processed/node_features.csv   (6 nodes x 1681 windows)
Run   : python src/p2_build_graph_tables.py
"""
import pandas as pd

from config import EDGE_TABLE, MASTER_TABLE, NODE_NAME_MAP, NODE_TABLE

DNP3_COLS = [
    "Num_Packets", "Avg_DNP3_Length", "F0_Confirm", "F1_Read", "F2_Write", "F4_Operate",
    "F5_DirectOperate", "F6_DirectOperate_NoResp", "F13_ColdRestart", "F14_WarmRestart",
    "F129_Response", "F130_UnsolicitedResponse", "IIN_Val_Change", "Analog_Input_Change",
    "Binary_Input_Change", "Src_Port_Changed", "Dst_Port_Changed",
]
TCP_COLS = [
    "Num_Packets", "Avg_Length", "Flag_Ack", "Flag_Retransmission", "Flag_Reset",
    "Flag_Syn", "Flag_Fin", "Round_Trip_Time",
]
NODE_COLS = [
    "Num_Packets", "Avg_Length", "stNum_Change", "sqNum_Reset", "ConfRev_Change",
    "Boolean_Data_Change", "Num_Events",
]


def main():
    df = pd.read_csv(MASTER_TABLE)

    # 1. split point to point rows (edges) from multicast rows (node events)
    edge_df = df[df["dst"].astype(str) != "MULTI"].copy()
    node_event_df = df[df["dst"].astype(str) == "MULTI"].copy()

    # 2. canonical device names
    edge_df["src_clean"] = edge_df["src"].astype(str).map(NODE_NAME_MAP)
    edge_df["dst_clean"] = edge_df["dst"].astype(str).map(NODE_NAME_MAP)
    node_event_df["src_clean"] = node_event_df["src"].astype(str).map(NODE_NAME_MAP)
    assert not edge_df[["src_clean", "dst_clean"]].isna().any().any(), "Unmapped edge device name"
    assert not node_event_df["src_clean"].isna().any(), "Unmapped node device name"

    # 3. node inventory (alphabetical -> stable integer ids)
    all_nodes = sorted(
        set(edge_df["src_clean"]) | set(edge_df["dst_clean"]) | set(node_event_df["src_clean"])
    )
    node_id_map = {name: i for i, name in enumerate(all_nodes)}
    print("Node ids:", node_id_map)

    edge_df["src_id"] = edge_df["src_clean"].map(node_id_map)
    edge_df["dst_id"] = edge_df["dst_clean"].map(node_id_map)
    node_event_df["node_id"] = node_event_df["src_clean"].map(node_id_map)

    # 4. EDGE table: DNP3 and TCP features side by side, one row per window per directed edge
    dnp3 = edge_df[edge_df["protocol"] == "DNP3"][
        ["Window_Num", "src_id", "dst_id"] + DNP3_COLS + ["Label", "Notes"]
    ].rename(columns={
        c: ("dnp3_Avg_Length" if c == "Avg_DNP3_Length" else f"dnp3_{c}") for c in DNP3_COLS
    })
    tcp = edge_df[edge_df["protocol"] == "TCP"][
        ["Window_Num", "src_id", "dst_id"] + TCP_COLS + ["Label", "Notes"]
    ].rename(columns={c: f"tcp_{c}" for c in TCP_COLS})

    edge_final = pd.merge(dnp3, tcp, on=["Window_Num", "src_id", "dst_id"], how="outer").fillna(0)
    edge_final = edge_final.sort_values(["Window_Num", "src_id", "dst_id"]).reset_index(drop=True)
    edge_final["Label"] = edge_final["Label_x"]
    edge_final["Notes"] = edge_final["Notes_x"]
    edge_final = edge_final.drop(columns=["Label_x", "Label_y", "Notes_x", "Notes_y"])
    edge_final.to_csv(EDGE_TABLE, index=False)

    # 5. NODE table: GOOSE publisher behavior, completed to a full window x node grid
    node_final = node_event_df[["Window_Num", "node_id"] + NODE_COLS + ["Label", "Notes"]].rename(
        columns={c: f"node_{c}" for c in NODE_COLS}
    )
    nodes = pd.DataFrame({"node_id": sorted(node_id_map.values())})
    windows = node_final[["Window_Num"]].drop_duplicates().sort_values("Window_Num")
    grid = windows.merge(nodes, how="cross")
    node_full = pd.merge(grid, node_final, on=["Window_Num", "node_id"], how="left")
    fill_cols = [c for c in node_full.columns if c not in ["Window_Num", "node_id"]]
    node_full[fill_cols] = node_full[fill_cols].fillna(0)
    node_full.to_csv(NODE_TABLE, index=False)

    print(f"Edge table : {edge_final.shape} -> {EDGE_TABLE}")
    print(f"Node table : {node_full.shape} -> {NODE_TABLE}")


if __name__ == "__main__":
    main()
