import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Literal

import torch

from nl_probes.utils.dataset_utils import TrainingDataPoint


@dataclass
class BaseDatasetConfig:
    pass


@dataclass
class DatasetLoaderConfig:
    custom_dataset_params: BaseDatasetConfig
    num_train: int
    num_test: int
    splits: list[str]
    model_name: str
    layer_percents: list[int]
    save_acts: bool
    batch_size: int
    prompt_model_name: str | None = None
    dataset_name: str = ""
    dataset_folder: str = "sft_training_data"
    seed: int = 42


def _config_hash(cfg: DatasetLoaderConfig, split: str, exclude: tuple[str, ...] = ("batch_size",)) -> str:
    """
    Stable short hash over the full config + split.
    Excludes path-like fields so moving folders does not change the filename.
    """

    def _strip(obj):
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items() if k not in exclude}
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        return obj

    payload = {"config": _strip(asdict(cfg)), "split": split}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2s(blob, digest_size=6).hexdigest()  # 12 hex chars


class ActDatasetLoader:
    def __init__(
        self,
        dataset_config: DatasetLoaderConfig,
    ):
        self.valid_splits = set(["train", "test"])
        self.dataset_config = dataset_config

        for split in self.dataset_config.splits:
            assert split in self.valid_splits, f"Invalid split: {split}"

    def create_dataset(self) -> None:
        """
        Note: Will always make all split(s) at the same time.
        This is so we ensure that train / test splits have no overlap.
        """
        raise NotImplementedError

    def load_dataset(
        self,
        split: Literal["train", "test"],
    ) -> list[TrainingDataPoint]:
        assert split in self.valid_splits, f"Invalid split: {split}"

        dataset_name = self.get_dataset_filename(split)
        filepath = os.path.join(self.dataset_config.dataset_folder, dataset_name)
        if not os.path.exists(filepath):
            os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
            self.create_dataset()

        saved_object = torch.load(filepath)
        data_dicts = saved_object["data"]
        data = [TrainingDataPoint(**d) for d in data_dicts]

        print(f"Loaded {len(data)} datapoints from {filepath}")
        return data

    def save_dataset(self, data: list[TrainingDataPoint], split: Literal["train", "test"]) -> None:
        data_filename = self.get_dataset_filename(split)
        data_path = os.path.join(self.dataset_config.dataset_folder, data_filename)
        torch.save(
            {
                "config": asdict(self.dataset_config),
                "data": [dp.model_dump() for dp in data],
            },
            data_path,
        )
        print(f"Saved {len(data)} {split} datapoints to {data_path}")

    def get_dataset_filename(self, split: Literal["train", "test"]) -> str:
        num_datapoints = self.dataset_config.num_train if split == "train" else self.dataset_config.num_test

        model_str = self.dataset_config.model_name.split("/")[-1]

        config_hash = _config_hash(self.dataset_config, split)

        filename = f"{self.dataset_config.dataset_name}_model_{model_str}_n_{num_datapoints}_save_acts_{self.dataset_config.save_acts}_{split}_{config_hash}"
        filename = filename.replace("/", "_").replace(".", "_").replace(" ", "_")
        return f"{filename}.pt"
