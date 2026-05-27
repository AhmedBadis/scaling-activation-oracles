"""Run: python experiments/summarize_eval.py"""
import json, glob, collections

results = collections.defaultdict(dict)  # [oracle_label/target/layer] -> {ds: acc}

for path in sorted(glob.glob("experiments/classification_oracle_eval/**/results.json", recursive=True)):
    with open(path) as f:
        data = json.load(f)
    meta = data["meta"]
    key = f"{meta['oracle_label']} | target={meta['target_model'].split('/')[-1]} | layer={meta['layer_percent']}"
    ds_correct = collections.defaultdict(list)
    for r in data["records"]:
        ds_correct[r["dataset_id"]].append(int(r["ground_truth"].strip().lower() == r["target"].strip().lower()))
    for ds, vals in sorted(ds_correct.items()):
        results[key][ds] = sum(vals) / len(vals)

for key, ds_accs in results.items():
    print(f"\n=== {key} ===")
    avg = sum(ds_accs.values()) / len(ds_accs)
    for ds, acc in ds_accs.items():
        print(f"  {ds:<40} {acc:.4f}")
    print(f"  {'AVERAGE':<40} {avg:.4f}")
