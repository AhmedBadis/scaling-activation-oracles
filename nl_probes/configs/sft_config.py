import datetime
from dataclasses import asdict, dataclass, field
from typing import Any

from huggingface_hub import login, whoami

from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, DatasetLoaderConfig
from nl_probes.utils.common import layer_percent_to_layer


@dataclass
class SelfInterpTrainingConfig:
    # --- Models ---
    oracle_model_name: str = "Qwen/Qwen3-1.7B"
    target_model_name: str = "Qwen/Qwen3-1.7B"
    hook_onto_layer: int = 1
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])
    act_layers: list[int] = field(default_factory=list)  # derived if empty
    
    # Backward compatibility
    @property
    def model_name(self) -> str:
        """For backward compatibility - returns oracle model name"""
        return self.oracle_model_name
    
    # Handle legacy single model_name for backward compatibility
    def __post_init__(self):
        # If only one model is set (legacy behavior), use it for both
        if hasattr(self, '_original_model_name'):
            # Legacy mode: single model_name was set
            self.oracle_model_name = self._original_model_name
            self.target_model_name = self._original_model_name
            delattr(self, '_original_model_name')

    # --- Data / experiment ---
    dataset_configs: list[dict] = field(default_factory=list)
    use_decoder_vectors: bool = True
    generation_kwargs: dict[str, Any] = field(default_factory=lambda: {"do_sample": False, "max_new_tokens": 20})
    steering_coefficient: float = 1.0
    dataset_folder: str = "sft_training_data"

    # --- Batching ---
    train_batch_size: int = 16
    eval_batch_size: int = 128
    activation_collection_batch_size: int = 128

    # --- LoRA ---
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # --- Training ---
    num_epochs: int = 1
    lr: float = 1e-5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    eval_steps: int = 9_999_999  # effectively off by default
    eval_on_start: bool = False
    gradient_checkpointing: bool = False
    window_mult: int = 20
    save_steps: int = 5_000
    save_dir: str = "checkpoints"
    use_learned_activation_projection: bool = False
    activation_projection_bias: bool = False
    seed: int = 42
    eval_logs_path: str = "eval_logs.json"
    load_lora_path: str | None = None

    # --- Tracking ---
    wandb_project: str = "sae_introspection"
    wandb_run_name: str = ""  # derived if empty
    wandb_suffix: str = ""

    # --- Hub ---
    hf_push_to_hub: bool = False
    hf_private_repo: bool = False
    hf_repo_name: str = ""  # optional short name, used to compute repo_id
    hf_repo_id: str = ""  # derived if empty and push is on

    # --- Misc experiment options ---
    positive_negative_examples: bool = False

    def finalize(self, dataset_loaders: list[ActDatasetLoader]) -> "SelfInterpTrainingConfig":
        self.dataset_configs = [asdict(dataset_loader.dataset_config) for dataset_loader in dataset_loaders]
        # act_layers from percents if caller did not set them directly
        if not self.act_layers:
            self.act_layers = [layer_percent_to_layer(self.oracle_model_name, p) for p in self.layer_percents]

        # run name - stable and readable
        layers_str = "-".join(map(str, self.act_layers))
        default_run = f"{self.oracle_model_name}-layers_{layers_str}-decoder-{self.use_decoder_vectors}{self.wandb_suffix}"
        if not self.wandb_run_name:
            self.wandb_run_name = default_run

        # save dir namespacing
        self.save_dir = namespaced_save_dir(self.save_dir, self.wandb_suffix)

        # repo id if pushing
        if self.hf_push_to_hub and not self.hf_repo_id:
            self.hf_repo_id = get_hf_repo_id(self.hf_repo_name)
        return self


def namespaced_save_dir(save_dir: str = "checkpoints", wandb_suffix: str = "") -> str:
    """Same path rule as ``SelfInterpTrainingConfig.finalize`` (without building loaders)."""
    if wandb_suffix and not save_dir.endswith(wandb_suffix):
        return f"{save_dir}{wandb_suffix}"
    return save_dir


def get_hf_repo_id(hf_repo_name: str) -> str:
    print("Setting up Hugging Face authentication...")
    # check if already logged in
    if whoami() is None:
        print("Not logged in to Hugging Face. Attempting to log in...")
        login()
    else:
        print("Already logged in to Hugging Face.")

    # Determine default HF repo name if not provided
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    if not hf_repo_name:
        hf_repo_name = f"gemma-introspection-{date_str}"

    # Compose full repo_id with current username
    user_info = whoami()
    owner = user_info.get("name") if isinstance(user_info, dict) else None
    hf_repo_id_computed = f"{owner}/{hf_repo_name}" if owner else hf_repo_name

    return hf_repo_id_computed
