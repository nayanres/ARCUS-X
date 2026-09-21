import post_run_analyzer as p

entry = ("\n=== RAW STREAM ENTRY (Tokens: 10) ===\n"
         "Parameters: z=3, gravity=1.0, tier=0, grid=8x8\n"
         "Grid: 8x8\nInitialState: [0,0]\nActions: [[0,0],[1,0]]\n"
         "TransitionRules: step=1\nEnvironmentMetadata: gravity=1.0\n"
         "[MODEL OUTPUT]\n[[0,0],[1,0],[2,0]]\n[GROUND TRUTH EXPECTED]\n[[0,0],[1,0],[2,0]]\n")

parsed = p.parse_raw_log_text(entry)
print("records:", len(parsed["records"]), "errors:", parsed["parse_errors"])
if parsed["records"]:
    r = parsed["records"][0]
    print("keys:", sorted(r.keys()))
    print("step_accuracy:", r.get("step_accuracy"))
    print("model_path_len:", r.get("model_path_len"), "truth_path_len:", r.get("truth_path_len"))
    print("_model_coords:", r.get("_model_coords"))
    print("_truth_coords:", r.get("_truth_coords"))
