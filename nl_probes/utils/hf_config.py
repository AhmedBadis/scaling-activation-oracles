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