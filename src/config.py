"""Central configuration for the OT IAS Work data pipeline.

All paths are relative to the project root (the folder that contains `src/` and `data/`),
so the scripts run from VS Code, a terminal, or any machine without edits.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SEQUENCE_DIR = PROCESSED_DIR / "sequences"
SCALER_DIR = PROCESSED_DIR / "scalers"
SCALED_CSV_DIR = PROCESSED_DIR / "scaled_csv"

MASTER_TABLE = PROCESSED_DIR / "master_table.csv"
EDGE_TABLE = PROCESSED_DIR / "edge_features.csv"
NODE_TABLE = PROCESSED_DIR / "node_features.csv"
WINDOW_METADATA = PROCESSED_DIR / "window_metadata.csv"

# ---------------------------------------------------------------------------
# Raw file registry: file name -> (protocol, source, destination)
# Order matters only for column order of the master table (DNP3, GOOSE, TCP).
# DNP3 / TCP files are point to point (RTAC <-> outstation). GOOSE is multicast.
# `src` / `dst` are the raw device names used by the original capture.
# ---------------------------------------------------------------------------
RAW_FILES = {
    "dnp3_70_to_72.csv":   ("DNP3", "RTAC_70", "72"),
    "dnp3_70_to_73.csv":   ("DNP3", "RTAC_70", "73"),
    "dnp3_72_to_70.csv":   ("DNP3", "RTAC_72", "70"),
    "dnp3_73_to_70.csv":   ("DNP3", "RTAC_73", "70"),
    "goose_rtds.csv":      ("GOOSE", "RTDS", "MULTI"),
    "goose_sel401_60.csv": ("GOOSE", "SEL_401_60", "MULTI"),
    "goose_sel421_71.csv": ("GOOSE", "SEL_421_71", "MULTI"),
    "goose_sel451_72.csv": ("GOOSE", "SEL_451_72", "MULTI"),
    "goose_sel451_73.csv": ("GOOSE", "SEL_451_73", "MULTI"),
    "tcp_70_to_72.csv":    ("TCP",  "RTAC_70", "72"),
    "tcp_70_to_73.csv":    ("TCP",  "RTAC_70", "73"),
    "tcp_72_to_70.csv":    ("TCP",  "RTAC_72", "70"),
    "tcp_73_to_70.csv":    ("TCP",  "RTAC_73", "70"),
}

# Canonical device (graph node) names. Node ids are assigned alphabetically in p2:
# RTDS=0, SCADA=1, SEL_401_60=2, SEL_421_71=3, SEL_451_72=4, SEL_451_73=5
NODE_NAME_MAP = {
    "70": "SCADA", "RTAC_70": "SCADA",
    "72": "SEL_451_72", "RTAC_72": "SEL_451_72", "SEL_451_72": "SEL_451_72",
    "73": "SEL_451_73", "RTAC_73": "SEL_451_73", "SEL_451_73": "SEL_451_73",
    "SEL_401_60": "SEL_401_60",
    "SEL_421_71": "SEL_421_71",
    "RTDS": "RTDS",
}

# ---------------------------------------------------------------------------
# Graph / sequence settings
# ---------------------------------------------------------------------------
NODE_FEATURE_COLS = [
    "node_Num_Packets", "node_Avg_Length", "node_stNum_Change", "node_sqNum_Reset",
    "node_ConfRev_Change", "node_Boolean_Data_Change", "node_Num_Events",
]

EDGE_FEATURE_COLS = [
    "dnp3_Num_Packets", "dnp3_Avg_Length", "dnp3_F0_Confirm", "dnp3_F1_Read",
    "dnp3_F2_Write", "dnp3_F4_Operate", "dnp3_F5_DirectOperate",
    "dnp3_F6_DirectOperate_NoResp", "dnp3_F13_ColdRestart", "dnp3_F14_WarmRestart",
    "dnp3_F129_Response", "dnp3_F130_UnsolicitedResponse", "dnp3_IIN_Val_Change",
    "dnp3_Analog_Input_Change", "dnp3_Binary_Input_Change", "dnp3_Src_Port_Changed",
    "dnp3_Dst_Port_Changed", "tcp_Num_Packets", "tcp_Avg_Length", "tcp_Flag_Ack",
    "tcp_Flag_Retransmission", "tcp_Flag_Reset", "tcp_Flag_Syn", "tcp_Flag_Fin",
]  # tcp_Round_Trip_Time is kept in the tables but intentionally not used as a model feature

# Only count / length features are scaled (RobustScaler, fit on TRAIN only);
# binary indicator features are left untouched.
NODE_SCALE_COLS = ["node_Num_Packets", "node_Avg_Length", "node_Num_Events"]
EDGE_SCALE_COLS = [
    "dnp3_Num_Packets", "dnp3_Avg_Length", "dnp3_F1_Read", "dnp3_F129_Response",
    "tcp_Num_Packets", "tcp_Avg_Length",
]

SEQUENCE_LENGTH = 3  # consecutive 5 s windows per sequence

# Chronological split by Window_Num (inclusive)
TRAIN_RANGE = (2815, 3766)
VAL_RANGE = (3767, 4067)
TEST_RANGE = (4068, 4495)

# Label convention: 0 = normal, 1 = deterministic attack (rule detectable),
# 2 = dynamic attack (protocol compliant, needs learning based detection)
LABEL_NAMES = {0: "normal", 1: "deterministic_attack", 2: "dynamic_attack"}
