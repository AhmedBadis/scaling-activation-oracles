"""Run: python experiments/summarize_ood_classification_eval.py"""
import json
import glob
import collections
import os

# oracle_id -> layer_percent -> dataset_id -> [correct bools]
results: dict[str, dict[int, dict[str, list[int]]]] = collections.defaultdict(
    lambda: collections.defaultdict(lambda: collections.defaultdict(list))
)
meta_by_oracle: dict[str, dict] = {}

for path in sorted(glob.glob("experiments/ood_classification_eval/**/results.json", recursive=True)):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data["meta"]
    oid = meta["oracle_label"]
    meta_by_oracle[oid] = meta

    for r in data["records"]:
        gt = str(r["ground_truth"]).strip().lower()
        pred = str(r["target"]).strip().lower()
        lp = int(r["layer_percent"])
        ds = r["dataset_id"]
        results[oid][lp][ds].append(int(gt == pred))

for oid, layers in sorted(results.items()):
    meta = meta_by_oracle.get(oid, {})
    cross = meta.get("cross_model", False)
    t_model = meta.get("target_model", "?").split("/")[-1]
    o_model = meta.get("oracle_model", "?").split("/")[-1]
    tag = f"[cross: oracle={o_model} act={t_model}]" if cross else f"[same: {o_model}]"

    print(f"\n{'=' * 70}")
    print(f"Oracle: {oid}  {tag}")
    print(f"{'=' * 70}")

    for lp, datasets in sorted(layers.items()):
        print(f"\n  --- layer={lp} ---")
        accs = []
        for ds, vals in sorted(datasets.items()):
            acc = sum(vals) / len(vals)
            accs.append(acc)
            print(f"    {ds:<45} {acc:.4f}  (n={len(vals)})")
        if accs:
            print(f"    {'AVERAGE':<45} {sum(accs)/len(accs):.4f}")