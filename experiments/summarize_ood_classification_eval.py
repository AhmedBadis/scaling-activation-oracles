"""Run: python experiments/summarize_ood_classification_eval.py"""
import json
import glob
import collections
import os

# (oracle_id, target_model) -> layer_percent -> dataset_id -> [correct bools]
#
# Keyed by the PAIR, not just oracle_id, because a single oracle_id can now
# produce multiple results.json files -- one per target it was co-trained
# against (dual/triple-target runs) -- living under
# experiments/ood_classification_eval/{oracle_id}/{target_tag}/results.json.
# Keying by oid alone would silently pool two different targets' correctness
# bools into one bucket and let the second-loaded file's meta clobber the
# first's, hiding exactly the per-target trade-off this eval is meant to show.
results: dict[tuple[str, str], dict[int, dict[str, list[int]]]] = collections.defaultdict(
    lambda: collections.defaultdict(lambda: collections.defaultdict(list))
)
meta_by_key: dict[tuple[str, str], dict] = {}

for path in sorted(glob.glob("experiments/ood_classification_eval/**/results.json", recursive=True)):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data["meta"]
    oid = meta["oracle_label"]
    t_model = meta.get("target_model", "?")
    key = (oid, t_model)
    meta_by_key[key] = meta

    for r in data["records"]:
        gt = str(r["ground_truth"]).strip().lower()
        pred = str(r["target"]).strip().lower()
        lp = int(r["layer_percent"])
        ds = r["dataset_id"]
        results[key][lp][ds].append(int(gt == pred))

# Group (oid, target_model) keys back under their shared oracle_id so
# multi-target oracles print as one block with a per-target sub-section each.
by_oracle: dict[str, list[str]] = collections.defaultdict(list)
for (oid, t_model) in results:
    by_oracle[oid].append(t_model)

for oid in sorted(by_oracle):
    t_models = sorted(by_oracle[oid])
    multi_target = len(t_models) > 1

    header = f"Oracle: {oid}"
    if multi_target:
        header += f"  (multi-target: {len(t_models)} targets)"
    print(f"\n{'#' * 70}")
    print(f"# {header}")
    print(f"{'#' * 70}")

    per_target_grand_avg: dict[str, float] = {}

    for t_model in t_models:
        key = (oid, t_model)
        layers = results[key]
        meta = meta_by_key.get(key, {})
        cross = meta.get("cross_model", False)
        t_short = t_model.split("/")[-1]
        o_short = meta.get("oracle_model", "?").split("/")[-1]
        tag = f"[cross: oracle={o_short} act={t_short}]" if cross else f"[same: {o_short}]"

        print(f"\n{'=' * 70}")
        print(f"Oracle: {oid}  ->  target: {t_short}  {tag}")
        print(f"{'=' * 70}")

        layer_avgs = []
        for lp, datasets in sorted(layers.items()):
            print(f"\n  --- layer={lp} ---")
            accs = []
            for ds, vals in sorted(datasets.items()):
                acc = sum(vals) / len(vals)
                accs.append(acc)
                print(f"    {ds:<45} {acc:.4f}  (n={len(vals)})")
            if accs:
                layer_avg = sum(accs) / len(accs)
                layer_avgs.append(layer_avg)
                print(f"    {'AVERAGE':<45} {layer_avg:.4f}")

        if layer_avgs:
            target_grand_avg = sum(layer_avgs) / len(layer_avgs)
            per_target_grand_avg[t_short] = target_grand_avg
            print(f"\n  {'GRAND AVERAGE (all layers)':<45} {target_grand_avg:.4f}")

    # For dual/triple-target oracles, also report the cross-target average --
    # mirrors the thesis's "grand average across both targets" comparison
    # (Table 5.6) and generalizes to any number of co-trained targets.
    if multi_target and per_target_grand_avg:
        print(f"\n  {'-' * 66}")
        for t_short, avg in per_target_grand_avg.items():
            print(f"    per-target grand average  {t_short:<30} {avg:.4f}")
        overall = sum(per_target_grand_avg.values()) / len(per_target_grand_avg)
        print(f"    {'GRAND AVERAGE ACROSS TARGETS':<45} {overall:.4f}")