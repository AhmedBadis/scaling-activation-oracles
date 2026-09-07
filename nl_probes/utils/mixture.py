"""Deterministic, fixed-budget mixture selection for Activation Oracle training.

The selector operates on materialized ``TrainingDataPoint`` objects rather than
loader configuration counts.  This is intentional: one source example can
expand into multiple datapoints because of QA variants, layer variants, or
activation windows.  Selecting after expansion makes the sample-count ablation
honest.

Profiles
--------
``legacy``
    Preserve the pre-mixture behavior and exclude optional COT-conversation data.

``diverse_fixed`` (default)
    Keep the number of examples equal to the legacy candidate pool while
    allocating fixed quotas across qualitatively different task families.  The
    classification and past/future pools are balanced across their individual
    dataset variants using deterministic round-robin sampling.

``context_heavy_fixed``
    A fixed-budget replication-oriented profile with more past/future data.
    This is useful for comparing the original paper's context-prediction-heavy
    ablation against the COT-conversation profile.

The profile can be selected without changing the training runner:

    AO_MIXTURE_PROFILE=diverse_fixed
    AO_MIXTURE_SEED=42
    AO_MIXTURE_TOTAL_SAMPLES=400000  # optional exact override

Weights can be overridden with a JSON object in ``AO_MIXTURE_WEIGHTS``.  The
keys are ``cot_oracle``, ``past_lens``, ``classification``, and
``legacy_latentqa``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
import random
from typing import Any, Iterable, Mapping, Sequence


FAMILY_COT = "cot_oracle"
FAMILY_PAST = "past_lens"
FAMILY_CLASSIFICATION = "classification"
FAMILY_LATENTQA = "legacy_latentqa"

_FAMILY_ORDER = (FAMILY_COT, FAMILY_PAST, FAMILY_CLASSIFICATION, FAMILY_LATENTQA)


@dataclass(frozen=True)
class MixtureProfile:
    name: str
    weights: Mapping[str, float] | None
    fixed_to_legacy_pool: bool
    include_optional_cot: bool


PROFILES: dict[str, MixtureProfile] = {
    "legacy": MixtureProfile(
        name="legacy",
        weights=None,
        fixed_to_legacy_pool=False,
        include_optional_cot=False,
    ),
    # The default gives the new solvability-oriented conversation task the
    # largest single quota, while retaining enough self-supervision to scale
    # and enough property diversity to prevent a conversational-only shortcut.
    "diverse_fixed": MixtureProfile(
        name="diverse_fixed",
        weights={
            FAMILY_COT: 0.40,
            FAMILY_PAST: 0.35,
            FAMILY_CLASSIFICATION: 0.20,
            FAMILY_LATENTQA: 0.05,
        },
        fixed_to_legacy_pool=True,
        include_optional_cot=True,
    ),
    "context_heavy_fixed": MixtureProfile(
        name="context_heavy_fixed",
        weights={
            FAMILY_COT: 0.30,
            FAMILY_PAST: 0.50,
            FAMILY_CLASSIFICATION: 0.15,
            FAMILY_LATENTQA: 0.05,
        },
        fixed_to_legacy_pool=True,
        include_optional_cot=True,
    ),
}


def datapoint_family(datapoint: Any) -> str:
    """Map a datapoint type to a stable high-level task family."""
    kind = str(getattr(datapoint, "datapoint_type", ""))
    if kind.startswith("cot_oracle"):
        return FAMILY_COT
    if kind.startswith("past_lens"):
        return FAMILY_PAST
    if kind.startswith("classification_"):
        return FAMILY_CLASSIFICATION
    if kind.startswith("latentqa"):
        return FAMILY_LATENTQA
    # Keep unknown future datasets useful instead of silently dropping them.
    return kind or "unknown"


def _source_key(datapoint: Any) -> str:
    """Return the finest stable source key available for balancing."""
    return str(getattr(datapoint, "datapoint_type", "unknown"))


def _largest_remainder_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights:
        return {}
    if any(value < 0 for value in weights.values()):
        raise ValueError(f"Mixture weights must be non-negative: {weights}")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError(f"Mixture weights must sum to a positive value: {weights}")

    raw = {family: total * float(weight) / weight_sum for family, weight in weights.items()}
    counts = {family: int(value) for family, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(
        weights,
        key=lambda family: (raw[family] - counts[family], -_FAMILY_ORDER.index(family) if family in _FAMILY_ORDER else 0),
        reverse=True,
    )
    for family in order[:remaining]:
        counts[family] += 1
    return counts


def _sample_balanced(
    datapoints: Sequence[Any],
    count: int,
    rng: random.Random,
) -> list[Any]:
    """Sample without replacement while covering each source type fairly."""
    if count <= 0 or not datapoints:
        return []
    if count >= len(datapoints):
        result = list(datapoints)
        rng.shuffle(result)
        return result

    pools: dict[str, list[Any]] = defaultdict(list)
    for datapoint in datapoints:
        pools[_source_key(datapoint)].append(datapoint)
    for pool in pools.values():
        rng.shuffle(pool)

    # Shuffle source order once, then rotate through it.  This avoids the
    # largest classification dataset consuming the whole family quota.
    source_keys = list(pools)
    rng.shuffle(source_keys)
    result: list[Any] = []
    while len(result) < count and source_keys:
        next_keys: list[str] = []
        for source_key in source_keys:
            pool = pools[source_key]
            if pool:
                result.append(pool.pop())
                if len(result) >= count:
                    break
            if pool:
                next_keys.append(source_key)
        source_keys = next_keys
    rng.shuffle(result)
    return result


def _parse_profile() -> MixtureProfile:
    name = os.environ.get("AO_MIXTURE_PROFILE", "diverse_fixed").strip().lower()
    if name in {"off", "none", "all"}:
        return MixtureProfile(name="all", weights=None, fixed_to_legacy_pool=False, include_optional_cot=True)
    if name not in PROFILES:
        valid = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown AO_MIXTURE_PROFILE={name!r}; choose one of: {valid}, all")
    return PROFILES[name]


def _parse_weight_override(defaults: Mapping[str, float] | None) -> Mapping[str, float] | None:
    raw = os.environ.get("AO_MIXTURE_WEIGHTS")
    if not raw:
        return defaults
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("AO_MIXTURE_WEIGHTS must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("AO_MIXTURE_WEIGHTS must be a JSON object")
    result = {str(key): float(value) for key, value in parsed.items()}
    unknown = set(result) - set(_FAMILY_ORDER)
    if unknown:
        raise ValueError(f"Unknown mixture families in AO_MIXTURE_WEIGHTS: {sorted(unknown)}")
    return result


def _parse_total(default: int | None) -> int | None:
    raw = os.environ.get("AO_MIXTURE_TOTAL_SAMPLES")
    if raw is None or raw.strip() == "":
        return default
    total = int(raw)
    if total < 1:
        raise ValueError("AO_MIXTURE_TOTAL_SAMPLES must be positive")
    return total


def _counts(datapoints: Iterable[Any]) -> Counter[str]:
    return Counter(datapoint_family(dp) for dp in datapoints)


def select_training_mixture(
    datapoints: Sequence[Any],
    *,
    profile: MixtureProfile | str | None = None,
    seed: int = 42,
    total_examples: int | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Select a deterministic mixture and return it with an audit manifest.

    ``total_examples`` is applied after dataset/layer/QA expansion.  If omitted
    for a fixed profile, the selector uses the size of the candidate pool that
    excludes optional COT data, making the default an apples-to-apples
    replacement rather than an accidental data-quantity increase.
    """
    if profile is None:
        selected_profile = _parse_profile()
    elif isinstance(profile, str):
        if profile in {"off", "none", "all"}:
            selected_profile = MixtureProfile("all", None, False, True)
        else:
            selected_profile = PROFILES[profile]
    else:
        selected_profile = profile

    rng = random.Random(seed)
    all_data = list(datapoints)
    by_family: dict[str, list[Any]] = defaultdict(list)
    for datapoint in all_data:
        by_family[datapoint_family(datapoint)].append(datapoint)

    legacy_candidates = [dp for dp in all_data if datapoint_family(dp) != FAMILY_COT]

    if selected_profile.name == "legacy":
        result = list(legacy_candidates)
        rng.shuffle(result)
        return result, {
            "profile": selected_profile.name,
            "seed": seed,
            "candidate_examples": len(all_data),
            "target_examples": len(result),
            "selected_examples": len(result),
            "candidate_by_family": dict(_counts(all_data)),
            "selected_by_family": dict(_counts(result)),
            "shortfall_by_family": {},
        }

    if selected_profile.name == "all":
        result = list(all_data)
        rng.shuffle(result)
        return result, {
            "profile": selected_profile.name,
            "seed": seed,
            "candidate_examples": len(all_data),
            "target_examples": len(result),
            "selected_examples": len(result),
            "candidate_by_family": dict(_counts(all_data)),
            "selected_by_family": dict(_counts(result)),
            "shortfall_by_family": {},
        }

    weights = _parse_weight_override(selected_profile.weights)
    if not weights:
        raise ValueError(f"Profile {selected_profile.name!r} needs mixture weights")

    legacy_total = len(legacy_candidates)
    target = total_examples
    if target is None:
        target = legacy_total if selected_profile.fixed_to_legacy_pool else len(all_data)
    target = _parse_total(target)
    assert target is not None

    desired = _largest_remainder_counts(target, weights)
    selected: list[Any] = []
    shortfall: dict[str, int] = {}
    for family in _FAMILY_ORDER:
        requested = desired.get(family, 0)
        pool = by_family.get(family, [])
        chosen = _sample_balanced(pool, requested, rng)
        selected.extend(chosen)
        if len(chosen) < requested:
            shortfall[family] = requested - len(chosen)

    # If a requested family is under-populated, fill from unused examples in a
    # deterministic, diversity-preserving round robin.  Usually this is only
    # relevant when a remote COT dataset is smaller than requested.
    if len(selected) < target:
        selected_ids = {id(dp) for dp in selected}
        remaining_by_family: dict[str, list[Any]] = {}
        for family, pool in by_family.items():
            remaining_by_family[family] = [dp for dp in pool if id(dp) not in selected_ids]
            rng.shuffle(remaining_by_family[family])
        fill_families = list(remaining_by_family)
        rng.shuffle(fill_families)
        while len(selected) < target and fill_families:
            next_families: list[str] = []
            for family in fill_families:
                pool = remaining_by_family[family]
                if pool:
                    selected.append(pool.pop())
                    if len(selected) >= target:
                        break
                if pool:
                    next_families.append(family)
            fill_families = next_families

    rng.shuffle(selected)
    manifest = {
        "profile": selected_profile.name,
        "seed": seed,
        "candidate_examples": len(all_data),
        "legacy_candidate_examples": legacy_total,
        "target_examples": target,
        "selected_examples": len(selected),
        "weights": dict(weights),
        "desired_by_family": desired,
        "candidate_by_family": dict(_counts(all_data)),
        "selected_by_family": dict(_counts(selected)),
        "shortfall_by_family": shortfall,
    }
    return selected, manifest


def apply_training_mixture(
    datapoints: Sequence[Any],
    *,
    seed: int = 42,
) -> list[Any]:
    """Apply the environment-configured mixture and print a compact manifest."""
    selected, manifest = select_training_mixture(datapoints, seed=seed)
    print(
        "AO training mixture: "
        f"profile={manifest['profile']} "
        f"selected={manifest['selected_examples']}/{manifest['candidate_examples']} "
        f"families={manifest.get('selected_by_family', {})}"
    )
    if manifest.get("shortfall_by_family"):
        print(f"AO mixture family shortfalls: {manifest['shortfall_by_family']}")
    return selected

def mixture_seed(default: int = 42) -> int:
    raw = os.environ.get("AO_MIXTURE_SEED")
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def write_mixture_manifest(manifest: Mapping[str, Any], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")

__all__ = [
    "FAMILY_COT",
    "FAMILY_PAST",
    "FAMILY_CLASSIFICATION",
    "FAMILY_LATENTQA",
    "MixtureProfile",
    "PROFILES",
    "apply_training_mixture",
    "datapoint_family",
    "select_training_mixture",
]