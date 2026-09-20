import copy
import os
import tempfile
from pathlib import Path
from pickle import UnpicklingError
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = 2


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


def _load_torch(path: str | Path) -> Any:
    load_kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, **load_kwargs, weights_only=False)
    except TypeError:
        return torch.load(path, **load_kwargs)
    except UnpicklingError:
        return torch.load(path, **load_kwargs, weights_only=False)
    except Exception as exc:
        msg = str(exc)
        if "numpy.core.multiarray._reconstruct" in msg or "WeightsUnpickler error" in msg:
            import numpy as np
            from torch.serialization import add_safe_globals

            add_safe_globals([np.core.multiarray._reconstruct])
            return torch.load(path, **load_kwargs, weights_only=False)
        raise


def read_latest_checkpoint_meta(save_dir: str) -> dict[str, Any]:
    """Return ``global_step`` and ``extra`` from the latest checkpoint."""
    ckpt_path = find_latest_checkpoint(save_dir)
    if ckpt_path is None:
        return {}
    ckpt = _load_torch(ckpt_path)
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
    """Find the latest checkpoint by step number, else newest epoch checkpoint."""
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


def _cleanup_old_checkpoints(
    save_dir: str,
    tag: str,
    just_saved_path: str,
    keep_last_n: int = 1,
) -> None:
    """Delete older checkpoints sharing ``tag``, keeping the newest ones."""
    if keep_last_n < 1:
        keep_last_n = 1

    p = Path(save_dir)
    candidates = list(p.glob(f"checkpoint_{tag}_*.pth"))
    try:
        candidates.sort(key=_checkpoint_step_number)
    except ValueError:
        candidates.sort(key=lambda x: x.stat().st_mtime)

    just_saved_resolved = str(Path(just_saved_path).resolve())
    others = [c for c in candidates if str(c.resolve()) != just_saved_resolved]

    n_old_to_keep = keep_last_n - 1
    to_delete = others[:-n_old_to_keep] if n_old_to_keep > 0 else others

    for old_ckpt in to_delete:
        try:
            old_ckpt.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"[checkpointing] Warning: failed to delete old checkpoint {old_ckpt}: {exc}")


def _normalise_param_names(
    optimizer: torch.optim.Optimizer,
    optimizer_param_names: list[list[str]] | None,
) -> list[list[str]] | None:
    """Validate the parameter-name layout against the live optimizer."""
    if optimizer_param_names is None:
        optimizer_param_names = getattr(optimizer, "_checkpoint_param_names", None)
    if optimizer_param_names is None:
        return None

    names = [list(group) for group in optimizer_param_names]
    if len(names) != len(optimizer.param_groups):
        raise ValueError(
            "optimizer_param_names has a different number of parameter groups: "
            f"names={len(names)}, optimizer={len(optimizer.param_groups)}"
        )
    for index, (group_names, group) in enumerate(zip(names, optimizer.param_groups)):
        if len(group_names) != len(group["params"]):
            raise ValueError(
                "optimizer_param_names does not match optimizer group size: "
                f"group={index}, names={len(group_names)}, params={len(group['params'])}"
            )
        if any(not isinstance(name, str) or not name for name in group_names):
            raise ValueError(f"optimizer_param_names contains an invalid name in group {index}")
    flattened = [name for group in names for name in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("optimizer_param_names must contain each parameter name exactly once")
    return names


def _optimizer_shapes(optimizer: torch.optim.Optimizer) -> list[list[tuple[int, ...]]]:
    return [
        [tuple(parameter.shape) for parameter in group["params"]]
        for group in optimizer.param_groups
    ]


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
    keep_last_n: int = 1,
    optimizer_param_names: list[list[str]] | None = None,
) -> str:
    """Save a complete, resumable training checkpoint.

    Optimizer state is accompanied by the parameter names corresponding to each
    optimizer parameter ID. This makes resume independent of harmless parameter
    additions/reordering, such as adding a separately trained projector module.
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"checkpoint_{tag}_{global_step if tag == 'step' else epoch}.pth"
    path = os.path.join(save_dir, filename)

    merged_module_states: dict[str, dict[str, Any]] = {}
    prev_ckpt_path = find_latest_checkpoint(save_dir)
    if prev_ckpt_path is not None and os.path.abspath(prev_ckpt_path) != os.path.abspath(path):
        try:
            prev_ckpt = _load_torch(prev_ckpt_path)
            merged_module_states.update(prev_ckpt.get("module_states") or {})
        except Exception as exc:
            print(f"[checkpointing] Warning: failed to load previous module_states for merge: {exc}")
    merged_module_states.update(module_states or {})

    optimizer_state = None
    names = None
    shapes = None
    if optimizer is not None:
        optimizer_state = optimizer.state_dict()
        names = _normalise_param_names(optimizer, optimizer_param_names)
        if names is not None:
            shapes = _optimizer_shapes(optimizer)

    ckpt = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "global_step": global_step,
        "epoch": epoch,
        "extra": extra or {},
        "module_states": merged_module_states,
    }
    if names is not None:
        ckpt["optimizer_param_names"] = names
        ckpt["optimizer_param_shapes"] = shapes
    if rng_state is not None:
        ckpt["rng_state"] = rng_state

    atomic_save(ckpt, path)
    _cleanup_old_checkpoints(save_dir, tag, path, keep_last_n=keep_last_n)
    return path


def _state_shape_error(
    parameter_name: str,
    parameter: torch.nn.Parameter,
    state_key: str,
    state_value: Any,
) -> ValueError | None:
    if not isinstance(state_value, torch.Tensor) or state_value.ndim == 0:
        return None
    if tuple(state_value.shape) != tuple(parameter.shape):
        return ValueError(
            "Optimizer state shape mismatch for "
            f"{parameter_name!r}, state={state_key!r}: "
            f"checkpoint={tuple(state_value.shape)}, current={tuple(parameter.shape)}"
        )
    return None


def _strip_wrapper_prefix(name: str) -> str:
    """Remove wrappers that are not part of a module's parameter identity."""
    while name.startswith("module."):
        name = name[len("module.") :]
    return name


def _state_dict_tensor_keys(state_dict: Any) -> set[str]:
    if not isinstance(state_dict, dict):
        return set()
    return {
        _strip_wrapper_prefix(key)
        for key, value in state_dict.items()
        if isinstance(key, str) and isinstance(value, torch.Tensor)
    }


def _checkpoint_has_current_model_parameters(
    checkpoint_model_state: Any,
    model: torch.nn.Module,
) -> bool:
    """Check that a legacy checkpoint has the current model parameter layout."""
    state_keys = _state_dict_tensor_keys(checkpoint_model_state)
    if not state_keys:
        return False

    for name, parameter in model.named_parameters():
        normalized = _strip_wrapper_prefix(name)
        exact = state_keys & {normalized}
        if exact:
            key = next(iter(exact))
            value = checkpoint_model_state.get(key)
            if value is None:
                value = next(
                    candidate_value
                    for candidate_key, candidate_value in checkpoint_model_state.items()
                    if isinstance(candidate_key, str)
                    and _strip_wrapper_prefix(candidate_key) == key
                )
            if tuple(value.shape) != tuple(parameter.shape):
                return False
            continue

        # PEFT/DDP wrappers can add a prefix. Require one unambiguous suffix
        # match, and still validate its shape.
        candidates = [
            key for key in state_keys
            if key.endswith("." + normalized)
        ]
        if len(candidates) != 1:
            return False
        candidate_key = candidates[0]
        value = next(
            candidate_value
            for candidate_key_raw, candidate_value in checkpoint_model_state.items()
            if isinstance(candidate_key_raw, str)
            and _strip_wrapper_prefix(candidate_key_raw) == candidate_key
        )
        if tuple(value.shape) != tuple(parameter.shape):
            return False
    return True


def _map_legacy_module_parameter(
    state_key: str,
    current_extra_names: list[str],
) -> str | None:
    """Map old standalone-projector state names to current optimizer names."""
    if not current_extra_names:
        return None

    # The original projector was torch.nn.Linear, whose state keys were simply
    # ``weight`` and ``bias``.  The current projector keeps that path under
    # ``base`` and adds a residual branch.
    if state_key in {"weight", "bias"}:
        preferred_suffix = ".base." + state_key
        candidates = [
            name for name in current_extra_names
            if name.endswith(preferred_suffix)
        ]
        if len(candidates) == 1:
            return candidates[0]

    candidates = [
        name for name in current_extra_names
        if name.endswith("." + state_key) or name == state_key
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _infer_legacy_optimizer_names(
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    current_names: list[list[str]] | None,
) -> list[list[str]] | None:
    """Reconstruct names for pre-v2 checkpoints that stored only numeric IDs.

    The trainer historically built one optimizer group as:

        model parameters, followed by activation-projector parameters.

    The model state dict and ``module_states`` preserve the order and identities
    needed to recover that layout.  Projector keys ``weight``/``bias`` are also
    migrated to the current ``base.weight``/``base.bias`` names.  If the layout
    cannot be recovered unambiguously, this returns ``None`` and the caller can
    apply the strict legacy fallback rather than guessing.
    """
    if current_names is None:
        return None

    current_names = [list(group) for group in current_names]
    current_flat_names = [name for group in current_names for name in group]
    current_flat_params = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(current_flat_names) != len(current_flat_params):
        return None

    saved_optimizer = checkpoint.get("optimizer_state_dict")
    if not isinstance(saved_optimizer, dict):
        return None
    saved_groups = saved_optimizer.get("param_groups")
    if not isinstance(saved_groups, list):
        return None
    saved_total = sum(
        len(group.get("params", []))
        for group in saved_groups
        if isinstance(group, dict)
    )
    if saved_total == 0:
        return None

    model_parameters = list(model.parameters())
    model_count = len(model_parameters)
    if model_count > len(current_flat_names) or saved_total < model_count:
        return None
    if not _checkpoint_has_current_model_parameters(
        checkpoint.get("model_state_dict"), model
    ):
        return None

    # The model portion is unchanged and is always the prefix used when the
    # optimizer was constructed.  Only the projector tail may differ.
    reconstructed = current_flat_names[:model_count]
    saved_extra_count = saved_total - model_count
    current_extra_names = current_flat_names[model_count:]

    if saved_extra_count:
        module_states = checkpoint.get("module_states") or {}
        if not isinstance(module_states, dict):
            return None

        candidates: list[tuple[str, list[str]]] = []
        current_prefixes = {
            name.split(".", 1)[0]
            for name in current_extra_names
            if "." in name
        }
        for module_key, module_state in module_states.items():
            if not isinstance(module_key, str) or not isinstance(module_state, dict):
                continue
            state_parameter_keys = [
                key
                for key, value in module_state.items()
                if isinstance(key, str) and isinstance(value, torch.Tensor)
            ]
            mapped = []
            for state_key in state_parameter_keys:
                mapped_name = _map_legacy_module_parameter(
                    state_key, current_extra_names
                )
                if mapped_name is not None:
                    mapped.append(mapped_name)
            if len(mapped) != saved_extra_count:
                continue
            if len(set(mapped)) != len(mapped):
                continue
            candidates.append((module_key, mapped))

        # Prefer a target-specific current key.  Otherwise a single legacy
        # projector state is safe; multiple indistinguishable targets are not.
        exact_candidates = [
            mapped
            for module_key, mapped in candidates
            if module_key in current_prefixes
        ]
        if len(exact_candidates) == 1:
            reconstructed.extend(exact_candidates[0])
        elif len(candidates) == 1:
            reconstructed.extend(candidates[0][1])
        else:
            return None

    if len(reconstructed) != saved_total:
        return None

    saved_group_sizes = [len(group.get("params", [])) for group in saved_groups]
    result = []
    offset = 0
    for size in saved_group_sizes:
        result.append(reconstructed[offset : offset + size])
        offset += size
    return result


def _match_groups_by_names(
    saved_names: list[list[str]],
    current_names: list[list[str]],
) -> list[int]:
    """Match current optimizer groups to saved groups by parameter-name overlap."""
    saved_sets = [set(group) for group in saved_names]
    used: set[int] = set()
    matches: list[int] = []
    for current_group in current_names:
        current_set = set(current_group)
        candidates = [
            (len(current_set & saved_sets[index]), index)
            for index in range(len(saved_sets))
            if index not in used
        ]
        if not candidates or max(candidates)[0] == 0:
            raise ValueError(
                "Could not match an optimizer parameter group to the checkpoint "
                f"for parameters beginning with {current_group[:3]!r}"
            )
        _, match = max(candidates)
        used.add(match)
        matches.append(match)
    return matches


def _remap_optimizer_state(
    optimizer: torch.optim.Optimizer,
    saved_optimizer_state: dict[str, Any],
    *,
    current_names: list[list[str]] | None,
    saved_names: list[list[str]] | None,
    allow_legacy_positional_extension: bool,
) -> None:
    """Restore optimizer state onto the current parameter objects by name.

    For pre-v2 checkpoints, parameter names were not stored. Those checkpoints
    are restored positionally only after validating the model layout and every
    tensor state shape. New checkpoints always use name-based restoration.
    """
    if not isinstance(saved_optimizer_state, dict):
        raise ValueError("Checkpoint optimizer_state_dict is not a dictionary")

    current_state = optimizer.state_dict()
    current_groups = current_state.get("param_groups", [])
    saved_groups = saved_optimizer_state.get("param_groups", [])
    saved_state = saved_optimizer_state.get("state", {})

    if saved_names is not None:
        if current_names is None:
            raise ValueError(
                "Checkpoint contains optimizer parameter names but the current optimizer "
                "does not provide them"
            )
        current_names = [list(group) for group in current_names]
        if len(current_names) != len(current_groups):
            raise ValueError("Current optimizer parameter-name layout is invalid")
        saved_names = [list(group) for group in saved_names]
        if len(saved_names) != len(saved_groups):
            raise ValueError("Checkpoint optimizer parameter-name layout is invalid")
        group_matches = _match_groups_by_names(saved_names, current_names)
    else:
        # Legacy checkpoints predate optimizer_param_names.  Restore by order
        # only when the group structure is unchanged; a newly appended module
        # (for example a separately optimized projector) is allowed and starts
        # with fresh optimizer moments.  Tensor shapes are validated below.
        if len(saved_groups) != len(current_groups):
            raise ValueError(
                "Legacy checkpoint optimizer groups are ambiguous: "
                f"checkpoint={len(saved_groups)}, current={len(current_groups)}"
            )
        group_matches = list(range(len(current_groups)))

    current_parameters = [
        list(group["params"])
        for group in optimizer.param_groups
    ]
    current_parameter_ids = [
        list(group["params"])
        for group in current_groups
    ]

    saved_by_name: dict[str, tuple[Any, int]] = {}
    if saved_names is not None:
        for saved_group_index, (group, group_names) in enumerate(zip(saved_groups, saved_names)):
            if len(group["params"]) != len(group_names):
                raise ValueError(
                    "Checkpoint optimizer parameter names do not match its parameter group size: "
                    f"group={saved_group_index}"
                )
            for parameter_id, name in zip(group["params"], group_names):
                if name in saved_by_name:
                    raise ValueError(f"Duplicate optimizer parameter name in checkpoint: {name!r}")
                saved_by_name[name] = (saved_state.get(parameter_id), saved_group_index)

    remapped_state: dict[Any, Any] = {}
    matched_count = 0
    for current_group_index, (current_group, current_names_group) in enumerate(
        zip(current_groups, current_names or [[] for _ in current_groups])
    ):
        saved_group_index = group_matches[current_group_index]
        saved_group = saved_groups[saved_group_index]
        current_params = current_parameters[current_group_index]
        current_ids = current_parameter_ids[current_group_index]

        if saved_names is None:
            if len(saved_group["params"]) > len(current_params):
                raise ValueError(
                    "Legacy checkpoint has more optimizer parameters than the current optimizer: "
                    f"checkpoint={len(saved_group['params'])}, current={len(current_params)}"
                )
            pairs = zip(
                saved_group["params"],
                current_ids[: len(saved_group["params"])],
                current_params[: len(saved_group["params"])],
                [f"group{current_group_index}[{index}]" for index in range(len(saved_group["params"]))],
            )
        else:
            pairs = []
            for current_id, parameter, name in zip(current_ids, current_params, current_names_group):
                saved_entry = saved_by_name.get(name)
                if saved_entry is None:
                    continue
                saved_parameter_state, _ = saved_entry
                saved_parameter_id = next(
                    parameter_id
                    for parameter_id, candidate_name in zip(
                        saved_groups[saved_entry[1]]["params"], saved_names[saved_entry[1]]
                    )
                    if candidate_name == name
                )
                pairs.append((saved_parameter_id, current_id, parameter, name))

        for saved_parameter_id, current_id, parameter, parameter_name in pairs:
            parameter_state = saved_state.get(saved_parameter_id)
            if parameter_state is None:
                continue
            if not isinstance(parameter_state, dict):
                raise ValueError(
                    f"Optimizer state for {parameter_name!r} is not a dictionary"
                )
            for state_key, state_value in parameter_state.items():
                error = _state_shape_error(parameter_name, parameter, state_key, state_value)
                if error is not None:
                    raise error
            remapped_state[current_id] = parameter_state
            matched_count += 1

    remapped_groups = [dict(group) for group in current_groups]
    if saved_names is not None:
        # Restore optimizer hyperparameters from the corresponding saved group,
        # while retaining the current parameter IDs and newly added parameters.
        for current_index, saved_index in enumerate(group_matches):
            for key, value in saved_groups[saved_index].items():
                if key != "params":
                    remapped_groups[current_index][key] = copy.deepcopy(value)
    else:
        for current_index, saved_index in enumerate(group_matches):
            for key, value in saved_groups[saved_index].items():
                if key != "params":
                    remapped_groups[current_index][key] = copy.deepcopy(value)

    if saved_names is None and not allow_legacy_positional_extension:
        saved_count = sum(len(group["params"]) for group in saved_groups)
        current_count = sum(len(group["params"]) for group in current_groups)
        if saved_count != current_count:
            raise ValueError(
                "Legacy checkpoint optimizer layout differs from the current layout "
                "and cannot be safely restored without parameter names"
            )

    optimizer.load_state_dict({"state": remapped_state, "param_groups": remapped_groups})
    if saved_names is not None and matched_count == 0 and saved_state:
        raise ValueError("No checkpoint optimizer states matched the current parameter names")


def _load_scheduler_state(scheduler: Any, scheduler_state: dict[str, Any] | None) -> None:
    if scheduler is None or scheduler_state is None:
        return
    try:
        scheduler.load_state_dict(scheduler_state)
        return
    except (RuntimeError, ValueError) as original_error:
        # Schedulers store one value per optimizer group. If a checkpoint was
        # written with a different number of groups, preserve the scalar
        # scheduler progress and fit the per-group lists to the live optimizer.
        adjusted = copy.deepcopy(scheduler_state)
        current_state = scheduler.state_dict()
        group_count = len(scheduler.optimizer.param_groups)
        for key, current_value in current_state.items():
            saved_value = adjusted.get(key)
            if not isinstance(current_value, list) or not isinstance(saved_value, list):
                continue
            if len(saved_value) == group_count:
                continue
            if len(saved_value) > group_count:
                adjusted[key] = saved_value[:group_count]
            else:
                adjusted[key] = saved_value + current_value[len(saved_value) :]
        try:
            scheduler.load_state_dict(adjusted)
        except Exception as exc:
            raise ValueError(
                "Checkpoint scheduler state is incompatible with the current optimizer "
                f"groups: {original_error}"
            ) from exc


def load_checkpoint_if_exists(
    save_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    modules_to_restore: dict[str, torch.nn.Module] | None = None,
    optimizer_param_names: list[list[str]] | None = None,
) -> tuple[int | None, int | None, dict[str, Any]]:
    """Load a checkpoint and restore model, modules, optimizer, scheduler, and RNG."""
    ckpt_path = find_latest_checkpoint(save_dir)
    if ckpt_path is None:
        return None, None, {}

    ckpt = _load_torch(ckpt_path)
    if ckpt is None or not isinstance(ckpt, dict):
        raise RuntimeError(f"Failed to load checkpoint from {ckpt_path}")

    model_state = ckpt.get("model_state_dict")
    if model_state is None:
        raise KeyError(f"Checkpoint {ckpt_path} has no model_state_dict")
    model.load_state_dict(model_state, strict=False)

    current_names = None
    if optimizer is not None:
        current_names = _normalise_param_names(optimizer, optimizer_param_names)
        if current_names is None:
            current_names = getattr(optimizer, "_checkpoint_param_names", None)

        saved_optimizer_state = ckpt.get("optimizer_state_dict")
        if saved_optimizer_state is not None:
            saved_names = ckpt.get("optimizer_param_names")
            legacy_extension = saved_names is None
            if legacy_extension:
                saved_names = _infer_legacy_optimizer_names(
                    ckpt,
                    model,
                    optimizer,
                    current_names,
                )
                if saved_names is not None:
                    print(
                        "[checkpointing] Reconstructed legacy optimizer parameter names "
                        "from model/module state; restoring optimizer moments by name.",
                        flush=True,
                    )
                else:
                    print(
                        "[checkpointing] Loading a legacy checkpoint without optimizer "
                        "parameter names; restoring by validated parameter order. "
                        "New checkpoints store names.",
                        flush=True,
                    )
            _remap_optimizer_state(
                optimizer,
                saved_optimizer_state,
                current_names=current_names,
                saved_names=saved_names,
                allow_legacy_positional_extension=legacy_extension and saved_names is None,
            )

    _load_scheduler_state(scheduler, ckpt.get("scheduler_state_dict"))

    if modules_to_restore:
        module_states = ckpt.get("module_states", {}) or {}
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