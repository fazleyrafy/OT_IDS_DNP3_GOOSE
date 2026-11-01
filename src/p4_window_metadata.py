"""Step 4 (new): window level ground truth for diagnosis / root cause work.

Parses the free text `Notes` column of the master table into one row per window:
attack tag (D1, D2, D3, G1, G2, G4), attack description, protocols / links / GOOSE stream ids named in the notes, label, split.
This is the localization ground truth for the attention guided root cause study.

Input : data/processed/master_table.csv
Output: data/processed/window_metadata.csv
Run   : python src/p4_window_metadata.py
"""
import re

import pandas as pd

from config import (LABEL_NAMES, MASTER_TABLE, TEST_RANGE, TRAIN_RANGE, VAL_RANGE,
                    WINDOW_METADATA)


def split_of(window):
    if TRAIN_RANGE[0] <= window <= TRAIN_RANGE[1]:
        return "train"
    if VAL_RANGE[0] <= window <= VAL_RANGE[1]:
        return "val"
    if TEST_RANGE[0] <= window <= TEST_RANGE[1]:
        return "test"
    return "unassigned"


def main():
    m = pd.read_csv(MASTER_TABLE, low_memory=False)
    m["Notes"] = m["Notes"].fillna("").astype(str)

    rows = []
    for window, g in m.groupby("Window_Num"):
        notes = sorted({n for n in g["Notes"] if n and n != "0"})
        tags = sorted({re.match(r"([A-Z]\d)", n).group(1) for n in notes if re.match(r"[A-Z]\d", n)})
        label = int(g["Label"].max())
        rows.append({
            "Window_Num": window,
            "split": split_of(window),
            "label": label,
            "label_name": LABEL_NAMES[label],
            "attack_tag": "+".join(tags) if tags else "",
            "notes": " | ".join(notes),
            # Protocols and DNP3 links named in the notes. NOTE: the Label column is window wide
            # (identical in every raw file), so it does NOT say which device/flow is attacked.
            # Localization ground truth only exists as text in Notes (see README).
            "protocols_in_notes": ";".join(sorted(set(re.findall(r"\b(DNP3|TCP|GOOSE|SV)\b", " ".join(notes))))),
            "links_in_notes": ";".join(sorted(set(re.findall(r"\b(\d{2}-\d{2})\b", " ".join(notes))))),
            "goose_stream_ids_in_notes": ";".join(sorted(set(re.findall(r"\b(?:TAP|RTDS|Mirror)\s+([0-9a-f]{6})\b", " ".join(notes))))),
        })

    meta = pd.DataFrame(rows)
    meta.to_csv(WINDOW_METADATA, index=False)

    print(f"Windows: {len(meta)} -> {WINDOW_METADATA}")
    print(meta.groupby(["split", "label_name"]).size().unstack(fill_value=0))
    print("Windows per attack tag:", meta[meta["attack_tag"] != ""]["attack_tag"].value_counts().to_dict())


if __name__ == "__main__":
    main()
