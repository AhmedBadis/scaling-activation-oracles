import os
import tempfile
from pathlib import Path
from pickle import UnpicklingError
from typing import Any

import torch


def atomic_save(obj: dict, path: str) -> None:
    """Save atomically to avoid partial checkpoint files."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    os.close(tmp_fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_latest_checkpoint_meta(save_dir: str) -> dict[str, Any]:
    """``global_step`` and ``extra`` from the latest checkpoint (no weight restore)."""
    ckpt_path = find_latest_checkpoint(save_dir)
    if ckpt_path is None:
        return {}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "global_step": int(ckpt.get("global_step", 0)),
        "extra": ckpt.get("extra") or {},
    }


def read_latest_checkpoint_extra(save_dir: str) -> dict[str, Any]:
    """Load ``extra`` from the latest checkpoint without restoring weights."""
    return read_latest_checkpoint_meta(save_dir).get("extra", {})


def _checkpoint_step_number(path: Path) -> int:
    stem = path.stem  # checkpoint_step_32075
    return int(stem.rsplit("_", 1)[-1])


def find_latest_checkpoint(save_dir: str) -> str | None:
    """Find the latest checkpoint by step number (step checkpoints), else newest epoch ckpt."""
    p = Path(save_dir)
    if not p.exists():
        return None
    step_ckpts = list(p.glob("checkpoint_step_*.pth"))
    if step_ckpts:
        return str(max(step_ckpts, key=_checkpoint_step_number))
    epoch_ckpts = sorted(p.glob("checkpoint_epoch_*.pth"), key=lambda x: x.stat().st_mtime)
    if epoch_ckpts:
        return str(epoch_ckpts[-1])
    return None


def save_checkpoint(
    save_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    global_step: int,
    epoch: int,
    rng_state: dict | None = None,
    extra: dict | None = None,
    tag: str = "step",
    module_states: dict[str, dict[str, Any]] | None = None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    filename = f"checkpoint_{tag}_{global_step if tag == 'step' else epoch}.pth"
    path = os.path.join(save_dir, filename)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "global_step": global_step,
        "epoch": epoch,
        "extra": extra or {},
        "module_states": module_states or {},
    }
    if rng_state is not None:
        ckpt["rng_state"] = rng_state

    atomic_save(ckpt, path)
    return path


def load_checkpoint_if_exists(
    save_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    modules_to_restore: dict[str, torch.nn.Module] | None = None,
) -> tuple[int | None, int | None, dict[str, Any]]:
    ckpt_path = find_latest_checkpoint(save_dir)
    if ckpt_path is None:
        return None, None, {}

    ckpt = None
    load_kwargs = {"map_location": "cpu"}

    try:
        ckpt = torch.load(ckpt_path, **load_kwargs)
    except UnpicklingError:
        try:
            ckpt = torch.load(ckpt_path, **{**load_kwargs, "weights_only": False})
        except TypeError:
            raise
    except Exception as e:
        msg = str(e)
        if "numpy.core.multiarray._reconstruct" in msg or "WeightsUnpickler error" in msg:
            import numpy as np
            from torch.serialization import add_safe_globals

            add_safe_globals([np.core.multiarray._reconstruct])
            ckpt = torch.load(ckpt_path, **{**load_kwargs, "weights_only": False})
        else:
            raise

    if ckpt is None:
        raise RuntimeError(f"Failed to load checkpoint from {ckpt_path}")

    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if modules_to_restore:
        module_states = ckpt.get("module_states", {})
        for module_name, module in modules_to_restore.items():
            state_dict = module_states.get(module_name)
            if state_dict is not None:
                module.load_state_dict(state_dict)

    if "rng_state" in ckpt:
        rng = ckpt["rng_state"]
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if "python" in rng:
            import random

            random.setstate(rng["python"])
        if "numpy" in rng:
            import numpy as np

            np.random.set_state(rng["numpy"])

    extra = ckpt.get("extra") or {}
    return int(ckpt.get("global_step", 0)), int(ckpt.get("epoch", 0)), extra


def capture_rng_state() -> dict[str, Any]:
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    import random
    import numpy as np

    state["python"] = random.getstate()
    state["numpy"] = np.random.get_state()
    return state
