# -*- coding: utf-8 -*-
"""Dataset helpers for YOLO train/validation paths."""
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


SPLIT_ALIASES = {
    "valid": "val",
    "validation": "val",
}


def load_dataset_config(data: str) -> Tuple[Path, Dict[str, Any]]:
    data_path = Path(data)
    yaml_path = data_path / "data.yaml" if data_path.is_dir() else data_path
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到数据集配置: {yaml_path}")

    with yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"数据集配置格式无效: {yaml_path}")
    return yaml_path.resolve(), cfg


def normalize_names(names: Any) -> List[str]:
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def resolve_dataset_root(yaml_path: Path, cfg: Dict[str, Any]) -> Path:
    raw_path = cfg.get("path")
    if not raw_path:
        return yaml_path.parent

    root = Path(str(raw_path)).expanduser()
    if root.is_absolute():
        return root
    return (yaml_path.parent / root).resolve()


def resolve_split_images(data: str, split: str = "val") -> Path:
    yaml_path, cfg = load_dataset_config(data)
    root = resolve_dataset_root(yaml_path, cfg)
    split_key = SPLIT_ALIASES.get(split, split)
    raw_split = cfg.get(split_key)
    if raw_split is None:
        raw_split = _first_existing(root, (f"{split}/images", f"images/{split}"))
    if raw_split is None:
        raise ValueError(f"data.yaml 缺少 split: {split}")
    return _resolve_one_split_path(root, raw_split)


def infer_label_dir(image_dir: Path) -> Path:
    parts = image_dir.parts
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "images":
            return Path(*parts[:idx], "labels", *parts[idx + 1:])
    raise ValueError(f"无法从图片目录推导标签目录: {image_dir}")


def _resolve_one_split_path(root: Path, raw_split: Any) -> Path:
    if isinstance(raw_split, (list, tuple)):
        if not raw_split:
            raise ValueError("split 路径列表为空")
        raw_split = raw_split[0]

    split_path = Path(str(raw_split)).expanduser()
    if split_path.is_absolute():
        return split_path
    return (root / split_path).resolve()


def _first_existing(root: Path, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if (root / candidate).exists():
            return candidate
    return None
