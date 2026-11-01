"""Run the whole preprocessing pipeline: python src/run_all.py  (or press Run in VS Code)."""
import p1_merge_raw
import p2_build_graph_tables
import p3_build_sequences
import p4_window_metadata

if __name__ == "__main__":
    print("=== P1: merge raw files ===");           p1_merge_raw.main()
    print("\n=== P2: node / edge tables ===");      p2_build_graph_tables.main()
    print("\n=== P3: graph sequences + scaling ==="); p3_build_sequences.main()
    print("\n=== P4: window metadata ===");         p4_window_metadata.main()
    print("\nDone. Outputs are in data/processed/")
