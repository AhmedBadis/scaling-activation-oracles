"""CoT-oracle conversational QA dataset for Activation Oracle training.

This loader implements the most directly relevant follow-up to the original
Activation Oracles paper: use target-model chain-of-thought prefixes as the
activation source, and ask a question whose answer is explicitly present in
that prefix.  The task is substantially different from ordinary LatentQA while
remaining solvable from the injected activations.

The default public dataset is ``cds-jb/cot-oracle-convqa-chunked``.  It is
expected to contain at least ``prompt``, ``cot_prefix``, and
``target_response`` columns.  Rows are cycled with replacement when the
requested materialized size is larger than the source split, so the requested
sample budget is exact.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm

from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint


@dataclass
class CotOracleDatasetConfig(BaseDatasetConfig):
    hf_dataset_repo: str = "cds-jb/cot-oracle-convqa-chunked"
    hf_split: str = "train"
    cot_prefix_field: str = "cot_prefix"
    prompt_field: str = "prompt"
    target_field: str = "target_response"
    stochastic_max_k: int = 100
    max_cot_prefix_tokens: int = 2048



def sample_cot_oracle_token_positions(
    num_tokens: int,
    rng: random.Random,
    *,
    max_k: int = 100,
) -> list[int]:
    """Sample sparse/broad CoT positions while preserving prefix boundaries.

    Small boundary-heavy samples encourage the oracle to work with the compact
    evidence pattern used in practical activation probing.  The log-uniform
    branch supplies broad coverage over activation-set sizes without letting
    long CoTs dominate the dataset.
    """
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    if max_k < 2:
        raise ValueError("max_k must be at least 2")
    if num_tokens == 1:
        return [0]
    if num_tokens == 2:
        return [0, 1]

    r = rng.random()
    if r < 0.20:
        return [0, num_tokens - 1]
    if r < 0.35:
        return [0, num_tokens - 2, num_tokens - 1]
    if r < 0.50:
        return [0, num_tokens - 3, num_tokens - 2, num_tokens - 1]

    hi = min(max_k, num_tokens)
    k = int(round(math.exp(rng.uniform(math.log(2), math.log(hi)))))
    k = max(2, min(k, num_tokens))
    positions = set(rng.sample(range(num_tokens), k))
    positions.update((0, num_tokens - 1))
    return sorted(positions)


class CotOracleDatasetLoader(ActDatasetLoader):
    DATASET_NAME = "cot_oracle_convqa"

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = self.DATASET_NAME
        self.dataset_params: CotOracleDatasetConfig = dataset_config.custom_dataset_params
        if self.dataset_config.save_acts:
            raise ValueError("CotOracleDatasetLoader requires save_acts=False for on-the-fly activation extraction")
        if not self.dataset_config.layer_percents:
            raise ValueError("CotOracleDatasetLoader requires non-empty layer_percents")

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)
        source = load_dataset(self.dataset_params.hf_dataset_repo, split=self.dataset_params.hf_split)
        if len(source) == 0:
            raise RuntimeError(f"No rows found in {self.dataset_params.hf_dataset_repo}:{self.dataset_params.hf_split}")

        layer_combinations = [
            [layer_percent_to_layer(self.dataset_config.model_name, p) for p in combo]
            for combo in self.dataset_config.layer_combinations
        ]
        rng = random.Random(self.dataset_config.seed)

        for split in self.dataset_config.splits:
            target_n = self.dataset_config.num_train if split == "train" else self.dataset_config.num_test
            if target_n <= 0:
                continue
            data = self._build_split(source, tokenizer, layer_combinations, rng, target_n)
            self.save_dataset(data, split)

    def _build_split(
        self,
        source,
        tokenizer,
        layer_combinations: list[list[int]],
        rng: random.Random,
        target_n: int,
    ) -> list[TrainingDataPoint]:
        order = list(range(len(source)))
        rng.shuffle(order)
        cursor = 0
        out: list[TrainingDataPoint] = []
        attempts = 0
        max_attempts = target_n * 10 + 1000
        pbar = tqdm(total=target_n, desc=f"{self.DATASET_NAME}/{self.dataset_params.hf_split}")

        while len(out) < target_n and attempts < max_attempts:
            attempts += 1
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            row = source[int(order[cursor])]
            cursor += 1

            prompt = row.get(self.dataset_params.prompt_field)
            cot_prefix = row.get(self.dataset_params.cot_prefix_field)
            target = row.get(self.dataset_params.target_field)
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            if not isinstance(cot_prefix, str) or not cot_prefix.strip():
                continue
            if not isinstance(target, str) or not target.strip():
                continue

            context_ids = tokenizer(
                cot_prefix,
                add_special_tokens=False,
                truncation=True,
                max_length=self.dataset_params.max_cot_prefix_tokens,
                return_tensors=None,
            )["input_ids"]
            if not context_ids:
                continue

            layers = list(rng.choice(layer_combinations))
            context_positions = sample_cot_oracle_token_positions(
                len(context_ids), rng, max_k=self.dataset_params.stochastic_max_k
            )
            meta_info: dict[str, Any] = {
                "cot_id": row.get("cot_id"),
                "source": row.get("source"),
                "split_index": row.get("split_index"),
                "bb_correct": row.get("bb_correct"),
                "n_positions_sampled": len(context_positions),
                "position_sampler": "cot_oracle_stochastic",
                "stochastic_max_k": self.dataset_params.stochastic_max_k,
                "cot_prefix_len_tokens": len(context_ids),
            }
            dp = create_training_datapoint(
                datapoint_type=self.DATASET_NAME,
                prompt=prompt,
                target_response=target,
                layers=layers,
                num_positions=len(context_positions),
                tokenizer=tokenizer,
                acts_BD=None,
                feature_idx=-1,
                context_input_ids=context_ids,
                context_positions=context_positions,
                ds_label=None,
                meta_info=meta_info,
            )
            out.append(dp)
            pbar.update(1)

        pbar.close()
        if len(out) != target_n:
            raise RuntimeError(
                f"{self.DATASET_NAME}: collected {len(out)}/{target_n} valid examples "
                f"after {attempts} attempts from {self.dataset_params.hf_dataset_repo}"
            )
        return out


__all__ = [
    "CotOracleDatasetConfig",
    "CotOracleDatasetLoader",
    "sample_cot_oracle_token_positions",
]