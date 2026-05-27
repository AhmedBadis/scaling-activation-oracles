"""Hugging Face ``AutoConfig`` helpers (composite configs, family keys, memory heuristics)."""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


def pretrained_hidden_size(model_id: str) -> int:
    """
    Return the text tower ``hidden_size`` for ``model_id``.

    ``Qwen3_5Config`` wraps ``text_config`` and does not expose ``hidden_size`` on the root object
    (unlike ``Qwen3Config``), which breaks naive ``AutoConfig...hidden_size`` access.
    """
    cfg: PretrainedConfig = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if getattr(cfg, "hidden_size", None) is not None:
        return int(cfg.hidden_size)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "hidden_size", None) is not None:
        return int(text_cfg.hidden_size)
    raise AttributeError(
        f"No hidden_size on root or text_config for {model_id!r} ({type(cfg).__name__}). "
        "Extend nl_probes.utils.hf_config.pretrained_hidden_size."
    )


def pretrained_model_type_key(model_id: str) -> str:
    """
    Stable **architecture family** key from ``transformers`` config (``model_type``).

    Same key ⇒ same modeling code family (e.g. ``llama``, ``gemma3``, ``qwen3``, ``qwen3_5``),
    even when parameter counts differ. Used for gradient-checkpointing heuristics.
    """
    cfg: PretrainedConfig = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    mt = getattr(cfg, "model_type", None)
    if mt:
        return str(mt)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        mt = getattr(text_cfg, "model_type", None)
        if mt:
            return str(mt)
    raise ValueError(
        f"Cannot read model_type from config for {model_id!r} ({type(cfg).__name__}). "
        "Extend nl_probes.utils.hf_config.pretrained_model_type_key."
    )


def same_pretrained_model_family(model_id_a: str, model_id_b: str) -> bool:
    return pretrained_model_type_key(model_id_a) == pretrained_model_type_key(model_id_b)


def gradient_checkpointing_same_pretrained_family(oracle_model_id: str, target_model_id: str) -> bool:
    """Enable GC when oracle and target share the same HF ``model_type`` (same code family)."""
    #return same_pretrained_model_family(oracle_model_id, target_model_id) #careful
    return False


def dual_target_auto_gradient_checkpointing(oracle_model_id: str, target_a: str, target_b: str) -> bool:
    """GC on only when oracle and both targets share one ``model_type``."""
    o = pretrained_model_type_key(oracle_model_id)
    a = pretrained_model_type_key(target_a)
    b = pretrained_model_type_key(target_b)
    return o == a == b
