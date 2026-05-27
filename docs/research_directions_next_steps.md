# Research directions (Activation Oracles fork)

This note lists **novel experiment axes** that fit the paper’s framing—training an LLM to verbalize another model’s internal states under steering—while leveraging this repo’s cross-model setup, learned projections, phased dual-target training, and OOD / secret-keeping evals.

## 1. Projection geometry and identifiability

- **Rank / singular-value structure** of the learned activation map (target → oracle width): does a low-rank subspace carry most of the “answerable” information for LatentQA vs classification vs past-lens tasks?
- **Orthogonal / semi-orthogonal constraints** on the projector vs unconstrained linear maps: effect on OOD taboo, SSC, and gender leakage.
- **Shared projector, shifting LoRA**: freeze the projector from one run and only adapt the oracle LoRA on a new target (or the reverse) to separate “where activations live” from “how the oracle speaks about them.”

## 2. Multi-target curricula (beyond two halves)

- Compare **phased dual-target** (this fork) to **interleaved minibatches** from two targets with identical hidden size, controlling for total tokens.
- **Target-skill transfer matrix**: after training on A then B, measure whether the oracle’s accuracy on held-out A drops (“catastrophic cross-target forgetting”) and whether a short A-only finetune recovers it.

## 3. Layer, coefficient, and norm-matching ablations

- The stack steers with **norm-matched additive injections** at chosen layers. Systematically sweep **injection layer**, **steering coefficient**, and **layer_percent** grids on the same checkpoint to see whether secret-keeping failures are localized mis-scalings vs representational limits.
- **Multi-layer readout**: instead of a single hook layer, train or evaluate with a small mixture of layers (or a learned per-layer weighting) injected into the oracle—closer to “distributed introspection” hypotheses.

## 4. Tokenizer and format mismatch (oracle ≠ target text pipeline)

- You already tokenize prompts with the **oracle** tokenizer while reading **target** activations. Quantify **alignment noise** (e.g., same underlying string tokenized differently) vs pure cross-width projection effects by holding the target fixed and swapping only the oracle tokenizer (where architecturally possible) or by inserting a controlled “formatting shim” in the prompt.

## 5. Stronger OOD and adaptive evaluation

- **Adaptive adversaries** that optimize short context strings to maximize leakage under fixed oracle checkpoints (not in-distribution taboo lists only).
- **Compositional secrets**: multiple constraints (SSC + persona fields) in one context to test whether the oracle’s failures are additive or super-additive.

## 6. Baselines that isolate “oracle” from “steering interface”

- **Linear decoder-only probes** on the same activation batches vs the full oracle LoRA, matched for data and layer, to report an “oracle gain” curve vs compute.
- **Frozen oracle backbone, train only projector** (or only LoRA) in a 2×2 factorial design to see which component drives cross-family generalization.

## 7. Scaling and data mixture laws

- Fix oracle size and vary **target size** (and vice versa) with matched token budgets; fit simple scaling curves for in-mixture vs OOD evals from `oracle_ood_eval` and training-side logs.

## 8. Dual-target oracles and activation distribution shift

- The dual-target run trains one LoRA on two activation sources. Evaluate **each target’s OOD suite** with the same `final/` adapter: does the oracle become a **compromise explainer** (middling on both) or retain **specialist** behavior on one target? Compare to two single-target checkpoints ensembled at inference (e.g., LoRA switching heuristics).

## 9. Thesis Qwen bases and secret-keeping LoRAs

- Hub target LoRAs for thesis activation bases live in `nl_probes/utils/lora_utils.py` (taboo partial for 1.7B/4B; gender/ssc/personaqa skipped until you train or publish LoRAs on those bases). To add coverage, extend that module and mirror in `experiments/*_open_ended_eval.py`.
- To run full **personaqa / gender / SSC** on Qwen3-1.7B, Qwen3-4B, or Qwen3.5-4B, train adapters with `nl_probes/trl_training/personaqa_train.py` (and analogous taboo/gender pipelines), push to Hub, then register IDs in `lora_utils.py`.

---

These directions are meant to be **paper-sized**: each bullet can anchor a section with 1–2 main figures (OOD matrix, scaling curve, or projector analysis) without rewriting the whole training stack.
