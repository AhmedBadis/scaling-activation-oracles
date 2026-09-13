"""Deterministic fixed-budget sampling for Activation Oracle SFT.

Counts are applied to materialized TrainingDataPoint objects. This matters because
loaders may expand one source row into several training examples. The paper's
full SFT run is one million materialized examples: 60% context prediction,
33.6% classification, and 6.4% SPQA. CoT is an optional fourth family that
reallocates the fixed budget; it is never silently added on top.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
import random
from typing import Any, Iterable, Mapping, Sequence

FAMILY_COT = "cot_oracle"
FAMILY_PAST = "past_lens"
FAMILY_CLASSIFICATION = "classification"
FAMILY_LATENTQA = "legacy_latentqa"
FAMILY_ORDER = (FAMILY_COT, FAMILY_PAST, FAMILY_CLASSIFICATION, FAMILY_LATENTQA)

ORIGINAL_TOTAL_SAMPLES = 1_000_000
ORIGINAL_WEIGHTS = {
    FAMILY_COT: 0.0,
    FAMILY_PAST: 0.60,
    FAMILY_CLASSIFICATION: 0.336,
    FAMILY_LATENTQA: 0.064,
}


def datapoint_family(datapoint: Any) -> str:
    kind = str(getattr(datapoint, "datapoint_type", ""))
    if kind.startswith("cot_oracle"):
        return FAMILY_COT
    if kind.startswith("past_lens"):
        return FAMILY_PAST
    if kind.startswith("classification_"):
        return FAMILY_CLASSIFICATION
    if kind.startswith("latentqa"):
        return FAMILY_LATENTQA
    return kind or "unknown"


def _source_key(datapoint: Any) -> str:
    return str(getattr(datapoint, "datapoint_type", "unknown"))


def normalise_weights(weights: Mapping[str, float] | None = None) -> dict[str, float]:
    values = {family: 0.0 for family in FAMILY_ORDER}
    if weights is not None:
        unknown = set(weights) - set(FAMILY_ORDER)
        if unknown:
            raise ValueError(f"Unknown mixture families: {sorted(unknown)}")
        values.update({str(key): float(value) for key, value in weights.items()})
    if any(value < 0 for value in values.values()):
        raise ValueError(f"Mixture weights must be non-negative: {values}")
    total = sum(values.values())
    if abs(total - 100.0) < 1e-6:
        values = {key: value / 100.0 for key, value in values.items()}
        total = 1.0
    if total <= 0 or abs(total - 1.0) > 1e-6:
        raise ValueError(f"Mixture weights must sum to 1.0 or 100.0; got {total}")
    return values


def weights_from_args(
    *,
    cot_pct: float = 0.0,
    past_pct: float = 60.0,
    classification_pct: float = 33.6,
    latentqa_pct: float = 6.4,
) -> dict[str, float]:
    return normalise_weights({
        FAMILY_COT: cot_pct,
        FAMILY_PAST: past_pct,
        FAMILY_CLASSIFICATION: classification_pct,
        FAMILY_LATENTQA: latentqa_pct,
    })


def mixture_weights(default: Mapping[str, float] | None = None) -> dict[str, float]:
    raw = os.environ.get("AO_MIXTURE_WEIGHTS")
    if not raw:
        return normalise_weights(default or ORIGINAL_WEIGHTS)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("AO_MIXTURE_WEIGHTS must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("AO_MIXTURE_WEIGHTS must be a JSON object")
    return normalise_weights(parsed)


def mixture_total_samples(default: int = ORIGINAL_TOTAL_SAMPLES) -> int:
    raw = os.environ.get("AO_MIXTURE_TOTAL_SAMPLES")
    total = default if raw is None or not raw.strip() else int(raw)
    if total < 1:
        raise ValueError("AO_MIXTURE_TOTAL_SAMPLES must be positive")
    return total


def _largest_remainder_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    raw = {family: total * float(weights.get(family, 0.0)) for family in FAMILY_ORDER}
    counts = {family: int(value) for family, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(
        FAMILY_ORDER,
        key=lambda family: (raw[family] - counts[family], -FAMILY_ORDER.index(family)),
        reverse=True,
    )
    for family in order[:remaining]:
        counts[family] += 1
    return counts


def mixture_target_counts(*, total: int | None = None, weights: Mapping[str, float] | None = None) -> dict[str, int]:
    return _largest_remainder_counts(
        mixture_total_samples() if total is None else total,
        mixture_weights() if weights is None else normalise_weights(weights),
    )


def _sample_balanced(datapoints: Sequence[Any], count: int, rng: random.Random) -> list[Any]:
    if count <= 0:
        return []
    if count > len(datapoints):
        raise ValueError(f"Requested {count} examples from a pool of {len(datapoints)}")
    if count == len(datapoints):
        result = list(datapoints)
        rng.shuffle(result)
        return result

    pools: dict[str, list[Any]] = defaultdict(list)
    for datapoint in datapoints:
        pools[_source_key(datapoint)].append(datapoint)
    for pool in pools.values():
        rng.shuffle(pool)
    source_keys = list(pools)
    rng.shuffle(source_keys)
    result: list[Any] = []
    while len(result) < count:
        progressed = False
        for source_key in source_keys:
            pool = pools[source_key]
            if pool:
                result.append(pool.pop())
                progressed = True
                if len(result) == count:
                    break
        if not progressed:
            raise RuntimeError("Mixture sampler stalled")
    rng.shuffle(result)
    return result


def _counts(datapoints: Iterable[Any]) -> Counter[str]:
    return Counter(datapoint_family(dp) for dp in datapoints)


def select_training_mixture(
    datapoints: Sequence[Any],
    *,
    weights: Mapping[str, float] | None = None,
    seed: int = 42,
    total_examples: int | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Select exactly the requested count and fail on any family shortfall."""
    rng = random.Random(seed)
    all_data = list(datapoints)
    by_family: dict[str, list[Any]] = defaultdict(list)
    for datapoint in all_data:
        by_family[datapoint_family(datapoint)].append(datapoint)

    unknown = set(by_family) - set(FAMILY_ORDER)
    if unknown:
        raise ValueError(f"Unclassified datapoint families would be dropped: {sorted(unknown)}")

    target = mixture_total_samples() if total_examples is None else int(total_examples)
    if target < 1:
        raise ValueError("total_examples must be positive")
    normalised = mixture_weights() if weights is None else normalise_weights(weights)
    desired = _largest_remainder_counts(target, normalised)

    selected: list[Any] = []
    for family in FAMILY_ORDER:
        requested = desired[family]
        pool = by_family.get(family, [])
        if requested > len(pool):
            raise RuntimeError(
                f"Mixture quota shortfall for {family}: requested {requested}, "
                f"available {len(pool)}. Increase the corresponding loader pool."
            )
        selected.extend(_sample_balanced(pool, requested, rng))

    rng.shuffle(selected)
    manifest = {
        "seed": seed,
        "target_examples": target,
        "selected_examples": len(selected),
        "weights": normalised,
        "desired_by_family": desired,
        "candidate_by_family": dict(_counts(all_data)),
        "selected_by_family": dict(_counts(selected)),
    }
    return selected, manifest


def apply_training_mixture(datapoints: Sequence[Any], *, seed: int = 42) -> list[Any]:
    selected, manifest = select_training_mixture(datapoints, seed=seed)
    print(
        "AO training mixture: "
        f"selected={manifest['selected_examples']} "
        f"weights={manifest['weights']} "
        f"families={manifest['selected_by_family']}"
    )
    return selected


def mixture_seed(default: int = 42) -> int:
    raw = os.environ.get("AO_MIXTURE_SEED")
    return default if raw is None or not raw.strip() else int(raw)


def write_mixture_manifest(manifest: Mapping[str, Any], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "FAMILY_COT", "FAMILY_PAST", "FAMILY_CLASSIFICATION", "FAMILY_LATENTQA",
    "ORIGINAL_TOTAL_SAMPLES", "ORIGINAL_WEIGHTS", "normalise_weights", "weights_from_args",
    "mixture_seed", "mixture_target_counts", "mixture_total_samples", "mixture_weights",
    "select_training_mixture", "apply_training_mixture", "write_mixture_manifest",
]