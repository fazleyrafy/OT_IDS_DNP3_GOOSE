# OT IAS Work: Data and Preprocessing for Attention-Guided Root Cause Diagnosis in Power Grid OT Networks

Clean starting point for the IEEE IAS (Transactions on Industry Applications, TPSO special issue, Cyber-Physical Security and Resilience track) paper *AI-Driven Intrusion Detection and Attention-Guided Root Cause Diagnosis for Cyber-Resilience in Power Grid Operational Technology Networks* (abstract in `docs/ias_abstract.pdf`).

This folder contains **only the data and the preprocessing code**. No trained models, no previous results. The data comes from a hardware-in-the-loop (HIL) IEC 61850 / DNP3 substation testbed built for the earlier OT-SHIELD intrusion detection work. What that earlier work did, and how its model code was structured, is documented in [`docs/previous_work.md`](docs/previous_work.md) so a new person (or a new Claude session) can pick it up without the old notebooks.

---

## 1. Quick start (VS Code)

```bash
# from the folder that contains src/ and data/
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
# install torch for your platform first (https://pytorch.org/get-started/locally/), then:
pip install -r requirements.txt

python src/run_all.py             # rebuilds everything in data/processed/ from data/raw/
```

In VS Code you can also open any file in `src/` and press Run. All paths are relative to the project root (`src/config.py` resolves them from its own location), so nothing needs editing. Steps 1, 2 and 4 need only pandas. Step 3 needs torch, torch_geometric and scikit-learn.

Everything in `data/processed/` is already generated (and verified identical to the original OT-SHIELD preprocessing), so you can start analysis immediately without running anything.

## 2. Folder structure

```
OT IAS Work/
├── README.md                this file
├── requirements.txt
├── .gitignore
├── docs/
│   ├── previous_work.md     what OT-SHIELD did, notebook by notebook, model code explained
│   ├── ias_abstract.pdf     current draft of the new paper
│   └── system_setup.png     testbed and attack vector figure (Fig. 1 of the paper)
├── data/
│   ├── raw/                 13 per protocol engineered feature files (the original data)
│   └── processed/
│       ├── master_table.csv         all 13 files merged, window aligned
│       ├── edge_features.csv        graph edge table (4 edges x 1681 windows)
│       ├── node_features.csv        graph node table (6 nodes x 1681 windows)
│       ├── window_metadata.csv      one row per window: split, label, attack tag, notes (ground truth)
│       ├── sequences/               train/val/test_sequences.pt  (PyG graph sequences, scaled)
│       ├── scalers/                 node_scaler.pkl, edge_scaler.pkl  (RobustScaler, fit on train)
│       └── scaled_csv/              the scaled sequences as CSV (for inspection only)
└── src/
    ├── config.py                    all paths, file registry, feature lists, splits, constants
    ├── p1_merge_raw.py              raw/*.csv           -> master_table.csv
    ├── p2_build_graph_tables.py     master_table.csv    -> edge_features.csv, node_features.csv
    ├── p3_build_sequences.py        edge/node tables    -> sequences/*.pt, scalers/, scaled_csv/
    ├── p4_window_metadata.py        master_table.csv    -> window_metadata.csv
    └── run_all.py                   runs p1, p2, p3, p4 in order
```

Raw files use simple names: `dnp3_<src>_to_<dst>.csv`, `tcp_<src>_to_<dst>.csv`, `goose_<publisher>.csv`. The numbers are device ids from the capture (70 = RTAC/SCADA, 72 and 73 = SEL-451 relays). The mapping from file name to protocol, source and destination lives in `RAW_FILES` in `src/config.py`.

## 3. The testbed in one paragraph

HIL digital substation with a **station bus** (DNP3 over TCP between the SCADA/RTAC SEL-3555 and the relays, HMI, SEL-2488 GPS clock) and a **process bus** (IEC 61850 GOOSE and Sampled Values between merging units, the RTDS power system simulator, and SEL-421 / SEL-451 / SEL-401 relays). Traffic was captured by port mirroring, parsed with PyShark, aggregated into **5 second windows**, and reduced to engineered features per protocol flow (see `docs/system_setup.png`). Attacks were injected with MITRE CALDERA, Scapy, Nmap and mitmproxy.

## 4. Data dictionary

### 4.1 Raw files (`data/raw/`, 1681 rows each, one row per 5 s window)

Common columns in every file: `Window_Num` (2815 to 4495), `Label` (see 4.5), `Notes` (free text, empty for normal windows).

| Files | Extra columns |
|---|---|
| `dnp3_*` (4 files, 20 columns) | `Num_Packets`, `Avg_DNP3_Length`, function code counts `F0_Confirm`, `F1_Read`, `F2_Write`, `F4_Operate`, `F5_DirectOperate`, `F6_DirectOperate_NoResp`, `F13_ColdRestart`, `F14_WarmRestart`, `F129_Response`, `F130_UnsolicitedResponse`, `IIN_Val_Change`, `Analog_Input_Change`, `Binary_Input_Change`, `Src_Port_Changed`, `Dst_Port_Changed` |
| `tcp_*` (4 files, 13 columns) | `Num_Packets`, `Avg_Length`, `Flag_Ack`, `Flag_Retransmission`, `Flag_Reset`, `Flag_Syn`, `Flag_Fin`, `Src_Port_Changed`, `Dst_Port_Changed`, `Round_Trip_Time` |
| `goose_*` (5 files, 10 columns) | `Num_Packets`, `Avg_Length`, `stNum_Change`, `sqNum_Reset`, `ConfRev_Change`, `Boolean_Data_Change`, `Num_Events` |

Directions in the file names: `70_to_72` means SCADA (RTAC 70) to SEL-451 relay 72; `72_to_70` is the reverse. GOOSE is multicast, one file per publisher (RTDS, SEL-401 id 60, SEL-421 id 71, SEL-451 id 72, SEL-451 id 73). No Sampled Value (SV) file exists in this dataset.

### 4.2 Processed tables (`data/processed/`)

| File | Shape | Description |
|---|---|---|
| `master_table.csv` | 21,853 x 37 | All raw files stacked. Added columns: `protocol`, `src`, `dst`, `edge_id`, `file_name`. Features that do not apply to a protocol are blank. |
| `edge_features.csv` | 6,724 x 30 | `Window_Num`, `src_id`, `dst_id`, 17 `dnp3_*` features, 8 `tcp_*` features (7 model features plus `tcp_Round_Trip_Time`), `Label`, `Notes`. 4 directed edges per window. |
| `node_features.csv` | 10,086 x 11 | `Window_Num`, `node_id`, 7 `node_*` GOOSE features, `Label`, `Notes`. 6 nodes per window. |
| `window_metadata.csv` | 1,681 x 10 | `Window_Num`, `split`, `label`, `label_name`, `attack_tag`, `notes`, `protocols_in_notes`, `links_in_notes`, `goose_stream_ids_in_notes`. |

**Graph definition.** Node ids (alphabetical): `0 RTDS, 1 SCADA, 2 SEL_401_60, 3 SEL_421_71, 4 SEL_451_72, 5 SEL_451_73`. Edges: `1->4, 1->5, 4->1, 5->1` (DNP3 plus TCP features). Per window the model sees `x` of shape `[6, 7]`, `edge_index` `[2, 4]`, `edge_attr` `[4, 24]`, label `y`.

### 4.3 Sequences (`data/processed/sequences/*.pt`)

`torch.load(path)` returns a list of dicts, one per sliding window of 3 consecutive graphs (sequence length T = 3, built inside each split):

```python
{"graphs": [Data, Data, Data],      # torch_geometric Data: x, edge_index, edge_attr, y, window_num
 "windows": [w, w+1, w+2],          # Window_Num of each graph
 "graph_labels": [0, 0, 2],         # per window label
 "sequence_label": 2}               # priority rule: any 2 -> 2, else any 1 -> 1, else 0
```

Features are scaled with a `RobustScaler` fitted on the **train split only** for count and length features (`node_Num_Packets`, `node_Avg_Length`, `node_Num_Events`, `dnp3_Num_Packets`, `dnp3_Avg_Length`, `dnp3_F1_Read`, `dnp3_F129_Response`, `tcp_Num_Packets`, `tcp_Avg_Length`). Binary and flag features are left as is. `tcp_Round_Trip_Time` is kept in the tables but is **not** part of the 24 edge model features (the earlier work excluded it).

```python
import torch, sys
sys.path.insert(0, "src")
from config import SEQUENCE_DIR
train = torch.load(SEQUENCE_DIR / "train_sequences.pt", weights_only=False)   # needs torch_geometric installed
```

### 4.4 Splits (chronological, set in `config.py`)

| Split | Window_Num | Windows | Normal (0) | Deterministic (1) | Dynamic (2) | Sequences (T=3) |
|---|---|---|---|---|---|---|
| train | 2815 to 3766 | 952 | 914 | 5 | 33 | 950 (896 / 15 / 39) |
| val | 3767 to 4067 | 301 | 263 | 5 | 33 | 299 (245 / 15 / 39) |
| test | 4068 to 4495 | 428 | 390 | 5 | 33 | 426 (370 / 15 / 41) |

Sequence counts in brackets are normal / label 1 / label 2. The earlier work trained only on normal sequences (label 0) and used the attack sequences for validation and test.

### 4.5 Labels and attack types

`Label`: **0** normal, **1** deterministic attack (protocol invariant violation, detectable by rules), **2** dynamic attack (protocol compliant, needs learning). Totals: 1,567 normal, 15 label 1, 99 label 2 windows (27 contiguous attack events, 6 to 51 windows per type).

| Tag | Attack | Label | Windows | Ground truth in `Notes` |
|---|---|---|---|---|
| D1 | DNP3 invalid operation (request function sent by outstation, IIN value change) | 1 | 9 | DNP3 link, e.g. `72-70`, `73-70` |
| G1 | GOOSE state manipulation (stNum / sqNum inconsistency) | 1 | 6 | GOOSE stream id (`2ca059`, `4f9dcd`) |
| D2 | Unauthorized breaker open from SCADA (attacker connects, sends Direct Operate, disconnects) | 2 | 9 | DNP3 and TCP links `70-72`, `72-70` |
| D3 | DNP3 delay / selective drop (raises RTT) | 2 | 12 | DNP3 and TCP links `70-73`, `73-70` |
| G2 | GOOSE flooding | 2 | 27 | GOOSE stream id `2ca04e` |
| G4 | GOOSE spoofing, injection and suppression | 2 | 51 | GOOSE stream id `33207b` |

The abstract mentions seven classes; **G3** (GOOSE stNum change / false trip) and **S1** (SV time sync) appear in `docs/system_setup.png` but are **not tagged** in this capture's `Notes`, and there is no SV data. Decide how to describe the attack count in the paper.

## 5. Things to know before using the data (quirks)

1. **Labels are window wide, not device level.** `Label` is identical in all 13 raw files for a given window, so it says *that* a window is attacked, not *which* device or flow. Localization ground truth exists only as text in `Notes` (DNP3 links for D1 to D3, GOOSE stream ids for G1, G2, G4). GOOSE stream ids are not mapped to device names anywhere in the data, so device level truth for GOOSE attacks needs a mapping from you (testbed logs).
2. **Isolated nodes.** Only SCADA (1), SEL_451_72 (4), SEL_451_73 (5) have edges. RTDS, SEL_401_60 and SEL_421_71 have GOOSE node features but no edges, so message passing does not reach them. SCADA has no GOOSE features (its node features are all zero). Keep this in mind when ranking devices by node level contribution for root cause.
3. **Zero filling.** Missing values are filled with 0 in `edge_features.csv` and `node_features.csv`, including the `Notes` column, so `Notes` contains the string `0` for non-attack rows. In `node_features.csv` the `Label` of nodes without GOOSE data (SCADA) is 0 even in attack windows. The window label is the **max over nodes**, which is what `p3` uses.
4. **Small data.** 114 attack windows in total (38 per split). Report counts next to percentages and expect wide uncertainty per attack type.
5. **File names inside the master table** (`file_name` column) use the new simple names; `edge_id` keeps the original style (`DNP3_RTAC_70_to_72`, `GOOSE_SEL_451_72_to_MULTI`).
6. **Windows are 5 s, ordered in time.** Do not shuffle across the train/val/test boundaries.

## 6. Suggested next steps for the IAS paper

Map to the abstract's stages:

1. **Stage 1 detection.** Re-implement the GNN + GRU + attention reconstruction model in a new `src/model.py` (architecture is documented in `docs/previous_work.md`), train on label 0 train sequences, choose the threshold on validation.
2. **Stage 2 root cause.** Needs node level embeddings before pooling, per sequence attention weights over the 3 windows, a normal baseline per node (mean embedding over normal train sequences), and ground truth from `window_metadata.csv`.
3. **Stage 3 resilience metric driven mitigation.** Uses the resilience metric from the PhD dissertation (Hussain, WVU ETD 13305). No code or data for it is in this folder yet.
4. Put new code in new files (`src/model.py`, `src/train.py`, ...) and write outputs to `models/` and `results/` (listed in `.gitignore` as optional exclusions).

## 7. Citation and acknowledgments

Hussain, Mohammed Mustafa, "Data-driven Analytics and Decision Support for Cyber-Physical Grid Resilience" (2026). Graduate Theses, Dissertations, and Problem Reports (ETD). 13305. https://researchrepository.wvu.edu/etd/13305

Supported partly by the NSF and the U.S. DARPA AI-CRAFT AWD16069 grant. SEL is acknowledged for support. Lane Department of Computer Science and Electrical Engineering, West Virginia University.
