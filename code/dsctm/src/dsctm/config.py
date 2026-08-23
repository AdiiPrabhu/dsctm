"""Config load / deep-merge / resolve / hash. Configs are plain YAML dicts;
the resolved dict is written to each run's config_resolved.yaml and hashed."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path) -> dict:
    if yaml is None:
        raise RuntimeError("pyyaml not installed")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_config(*layers) -> dict:
    """Merge an ordered list of dicts and/or YAML paths (later overrides earlier)."""
    cfg: dict = {}
    for layer in layers:
        if layer is None:
            continue
        if isinstance(layer, (str, Path)):
            layer = load_yaml(layer)
        cfg = _deep_merge(cfg, layer)
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def dump_yaml(cfg: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        Path(path).write_text(json.dumps(cfg, indent=2, default=str))
        return
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
