# Previous Work: OT-SHIELD (what was done, and how the old code worked)

Read this to understand the earlier work that produced the data in this repository. The old notebooks, trained models and results are **not** copied here on purpose (they live in the parent folder `OT_SHIELD/` of the author's research workspace). The new IAS paper reuses only the testbed, the data, and the preprocessing; the model is to be re-implemented and re-trained. Published metric values are intentionally not repeated here as targets.

## 1. The earlier paper

**SHIELD-OT / OT-SHIELD**: *A Hybrid Rule-Based and GNN-GRU-Attention Framework for Intrusion Detection in Multi-Protocol OT Systems* (H. M. Mustafa, G. Butler, A. K. Srivastava, West Virginia University; manuscript under review). Background and the full pipeline are also in the author's dissertation: Hussain, Mohammed Mustafa, "Data-driven Analytics and Decision Support for Cyber-Physical Grid Resilience" (2026), WVU ETD 13305.

Idea: OT attacks come in two kinds. **Deterministic** attacks break a protocol invariant (invalid state transition, wrong function code) and a rule engine catches them with no false positives. **Dynamic** attacks keep every message protocol compliant but distort behavior across devices, protocols and time (unauthorized but valid breaker command, delay, flooding, spoofed but well formed GOOSE) and need a learned model. OT-SHIELD runs both and fuses them as `D(x) = R(x) + (1 - R(x)) * G(x)` (rule decision wins, learned model scores whatever the rules pass).

Limitations that motivate the new IAS paper: the output is one binary flag per sequence, with no device, protocol or time localization and no response guidance.

### 1.1 Rule engine (not in this repository)

Protocol invariants for DNP3 (function code consistency, request/response association, IIN stability), GOOSE (identity, publisher binding, stNum/sqNum transitions) and SV (stream identity, sync, sampling counter continuity), written as YAML rules, evaluated per packet against the previous state held in SQLite. The code and YAML files were not part of the preserved workspace, so the rule engine cannot be reproduced from here. In the dataset its coverage is represented by `Label = 1` windows.

### 1.2 Learned model (GNN + GRU + attention reconstruction)

Per 5 s window a graph with 6 nodes and 4 directed edges is built (node features 7 dims, edge features 24 dims). The model is trained only on normal sequences to reconstruct a *sequence aware target embedding*; reconstruction error is the anomaly score.

| Component | Detail (from the old `P4_Seq_Multi_OT_IDS` notebook) |
|---|---|
| Encoder `SharedGNNEncoder` | `NNConv` (edge conditioned convolution, `aggr="mean"`), `in=7`, `out=32`, edge network = a single `Linear(24, 32*7)`; ReLU; graph readout = concat(mean pool, max pool) giving a 64 dim graph embedding `z_t` |
| Temporal module `GRUAttention` | `GRU(64 -> 32, batch_first)`, then `Linear(32, 1)` score per time step, softmax over the T = 3 steps, context vector = attention weighted sum of GRU outputs (32 dims). Attention weights have shape `[1, 3, 1]` |
| Decoder `GraphEmbeddingDecoder` | `Linear(32, 64)`, ReLU, `Linear(64, 64)` maps the context vector to a reconstructed 64 dim embedding |
| Target | mean of the T graph embeddings of the sequence (`sequence_tensor.mean(0)`), "sequence aware" |
| Loss and score | `smooth_l1_loss(beta=1.0)` (Huber) between reconstruction and target; the same quantity is the anomaly score |
| Training | Adam, lr 1e-3, no weight decay, 20 epochs, batch size 1 (one sequence per optimizer step), seed 42, **only sequences with `sequence_label == 0`** |
| Thresholding | candidate thresholds from quantiles of validation scores (normal scores quantile 0.90 to 1.00 and all scores 0.80 to 1.00, 101 points each); pick the threshold with the best F1 on validation, ties by precision/recall; apply unchanged to test |
| Evaluation convention | sequence level precision, recall, F1; coverage = fraction of attack events with at least one flagged window; sequences with `sequence_label == 1` are excluded when scoring the learned branch (rules own them) |

Ablation variants that were trained for comparison: `graph_only_reconstruction` (no temporal module), `gnn_gru`, `gnn_lstm`, `gnn_temporal_attention` (attention without GRU), `gnn_gru_attention` (final), and both "last snapshot target" and "sequence aware target" reconstruction modes; plus a loss sweep (MSE, MAE, Huber).

Reported conclusions from that work (context only): the hybrid detected 78 of 81 attack sequences, and the three missed sequences each had a single attacked window at the first or last position of the 3 window sequence ("boundary dilution" of the mean based target).

## 2. Old notebook catalog (all were Jupyter notebooks with hard coded Windows paths)

The data preprocessing notebooks `P1` to `P3` are what `src/p1_merge_raw.py`, `src/p2_build_graph_tables.py` and `src/p3_build_sequences.py` now implement as relative path scripts, with identical outputs.

| Old notebook | Role | Reads | Writes |
|---|---|---|---|
| `P1_OT_IDS` | Parse file names into protocol/src/dst, merge 13 files, integrity checks | raw CSVs | `master_edge_table.csv` |
| `P2_OT_IDS` | Canonical device naming, split into edge rows (DNP3 + TCP) and node rows (GOOSE multicast), full window x node grid | master table | `final_edge_features.csv`, `final_node_features.csv` |
| `P3_OT_IDS` | Build PyG graph per window, T=3 sequences per split, RobustScaler on train, export | edge/node tables | `final_dataset/*.pt`, `scalers/*.pkl`, `scaled_outputs/*.csv` |
| `P4_Multi` / `P5_Multi` | Train / evaluate 4 ablation models, last snapshot target, MSE | sequences | per model checkpoints, scores, thresholds, error analysis CSVs |
| `P4_Tune_Seq` / `P5_Tune_Seq` | GRU only and attention only models, sequence aware target, MSE/MAE/Huber sweep | sequences | loss ablation results |
| `P4_Seq_Multi` / `P5_Seq_Multi` | **Final combined GNN+GRU+Attention**: train; then threshold search, test evaluation, false positive/negative tables, inference timing with `time.perf_counter()` | sequences | final checkpoint, score tables, metrics |
| `Diagnosis_OT_IDS` | Retrain the final model with per epoch train and validation loss tracking | sequences | loss curves, checkpoints at epochs 1 and 20 |
| `Extraction_OT_IDS` | Load checkpoints, extract per sequence context vectors, anomaly scores and attention weights for train(normal)/val/test; compare epoch 0, 1, 20 (PCA, centroid distances, cosine similarity); control model with a frozen random initialized GNN (`static_gnn_gru_attention`) | checkpoints | `analysis_artifacts/` (`.npy`, `.pt`, metadata CSVs) |
| `Plot_OT_IDS` | Figures: loss curves, score distributions with threshold, attention profiles and heatmaps (normal vs anomalous, TP vs FN), PCA of latent space | Diagnosis + Extraction outputs | PNG files |

Active figure chain was `P1 -> P2 -> P3 -> P4_Seq_Multi -> Diagnosis -> Extraction -> Plot`; the Multi and Tune_Seq pairs were side branches for baseline and loss tables.

The attention weights `alpha_i` and context vectors that Stage 2 of the new paper wants to reuse were produced by `Extraction_OT_IDS` (per sequence, T = 3 weights). Node level embeddings (before pooling) were never saved and must be extracted from a newly trained encoder.

## 3. Known facts and open items inherited from the earlier work

* The earlier paper reports a different number of attack classes than the data tags: D1, D2, D3, G1, G2, G4 are tagged in `Notes`; G3 and S1 are not. No SV traffic file exists.
* Sequence length T = 3 was never swept; only one seed (42) was used; only one 6 node topology exists; no external baselines (for example Isolation Forest, plain autoencoder) were run.
* The rule engine code, raw packet captures (pcap), and the latency timing script for the whole pipeline are not available.
* The earlier work's remaining reviewer-facing ideas (holdout attack generalization, scalability, transformer temporal module) are not part of this repository's scope.

## 4. Where the old material lives

Parent folder `OT_IDS_IAS_Special Issue/OT_SHIELD/` of the author's workspace: the 13 notebooks, `Final_Files_Noisy/` (data plus all old model results and plots), `AI_and_Rule_base_for_Cyber_IDS.pdf` (the paper), `background.md`, and `OT-SHIELD_Pipeline_Audit_and_Reviewer_Response_Strategy.md` (a notebook by notebook audit with reviewer response notes). Not needed to start the new paper.
