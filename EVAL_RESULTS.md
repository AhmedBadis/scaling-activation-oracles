# Evaluation results snapshot (training-time classification eval)

This file aggregates **`eval_*` metrics from `wandb-summary.json`** for local runs under `wandb/`. These are **in-training classification benchmarks** included in the default mixture (not the same as secret-keeping taboo/SSC eval scripts under `experiments/`, which you run separately).

**Convention**: run folder suffix hints oracle/target shorthand (`o` = Qwen3 oracle, `O` = Qwen3.5 oracle, `t`/`T` = target family).

| WandB run folder (local) | Inferred combo | Final `eval_ans_correct` (selection) |
|--------------------------|----------------|----------------------------------------|
| `run-20260511_120445-7x9wx1ev-o1.7t4` | o1.7 → t4 (Qwen3-1.7B oracle, Qwen3-4B target) | geometry_of_truth **0.783**, md_gender **0.795**, snli **0.781**, sst2 **0.775**, ner **0.805**, tense **0.873**, relations **0.624**, ag_news **0.747**, language_id **0.476**, singular_plural **0.515** |
| `run-20260430_163007-u23rgit2-o1.7t1.7` | o1.7 → t1.7 (self-family) | geometry **0.924**, md_gender **0.933**, snli **0.929**, sst2 **0.865**, ner **0.937**, tense **0.972**, relations **0.739**, ag_news **0.824**, language_id **0.763**, singular_plural **0.749** |
| `run-20260429_172741-n81quo9s-o1.7t4` | o1.7 → t4 (alternate run) | geometry **0.520**, md_gender **0.461**, snli **0.631**, sst2 **0.500**, ner **0.673**, tense **0.509**, relations **0.477**, ag_news **0.547**, language_id **0.429**, singular_plural **0.503** |
| `run-20260423_163231-kxr8bb6v-o4t4` | o4 → t4 | geometry **0.949**, md_gender **0.920**, snli **0.940**, sst2 **0.887**, ner **0.957**, tense **0.980**, relations **0.781**, ag_news **0.757**, language_id **0.823**, singular_plural **0.797** |
| `run-20260508_230824-xlu5xhex-O4t4` | O3.5-4B → t4 | geometry **0.839**, md_gender **0.913**, snli **0.903**, sst2 **0.872**, ner **0.929**, tense **0.967**, relations **0.733**, ag_news **0.847**, language_id **0.652**, singular_plural **0.896** |

All listed `eval_format_correct/classification_*` values in those summaries were **1.0** except run `O4t4`, where several format-correct metrics are **~0.99** (see raw JSON).

**How to refresh this table**

```bash
python3 -c "import json,glob; import pathlib; \
root=pathlib.Path('wandb'); \
[print(p.parent.parent.name, json.load(open(p))['eval_ans_correct/classification_geometry_of_truth']) \
 for p in sorted(root.glob('**/wandb-summary.json'))]"
```

Then copy the relevant keys into this Markdown.
