# -*- coding: utf-8 -*-

"""BCPT-Med: Bounded-Consensus Partial Transport for medical clustering.

This is an offline-first, label-free clustering implementation for imbalanced
medical image collections. Missing MedMNIST files may be acquired once from
the official API and are then reused from ``offline_pack``. The proposed
BCPT-Med objective combines four mechanisms:

1. cross-foundation neighbour consensus estimates sample reliability;
2. a credal (interval-valued) cluster marginal is estimated from view-specific
   teacher predictions with a finite-sample uncertainty term; and
3. reject-augmented partial optimal transport assigns only a scheduled fraction
   of the source mass while retaining the accepted mass in the training loss;
   and
4. a label-free anchor-retention and graph/cohesion guard prevents stochastic
   self-training from replacing a stronger initialization partition.

Labels are never supplied to feature extraction, graph construction, head
training, head selection, or hyper-parameter selection.  They are loaded only
for post-hoc Hungarian-matched evaluation.  The number of clusters K is assumed
known from dataset metadata, which is the standard known-K clustering setting.

The former TANGO command aliases are retained only for CLI compatibility.  The
method is called BCPT-Med to avoid collision with the previously published
TANGO clustering algorithm.
"""

# =============================================================================
# 0. Optional dependency installer for Colab/Jupyter
# =============================================================================

import os
import sys
import math
import time
import gc
import csv
import json
import shutil
import random
import subprocess
import tempfile
import shlex
import platform
import resource
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Any


def install_colab_dependencies(include_biomedclip: bool = True):
    """Run once at the top of a Colab/Jupyter notebook.

    Do not place shell magic like `!pip install ...` inside a normal .py file.
    This function is safe in both notebooks and scripts.
    """
    pkgs = [
        "medmnist>=3.0.2",
        "scikit-learn",
        "scipy",
        "pandas",
        "matplotlib",
        "tqdm",
        "pillow",
    ]
    if include_biomedclip:
        pkgs += ["open_clip_torch", "huggingface_hub", "timm"]
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    print("[install]", " ".join(cmd))
    subprocess.check_call(cmd)


# =============================================================================
# 1. Imports that require installed packages
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    recall_score,
)
from scipy.stats import wilcoxon


# =============================================================================
# 2. Reproducibility, offline paths, and utilities
# =============================================================================


# All offline assets are resolved relative to this script by default:
#
#   BCPT_Med_AAAI.py
#   offline_pack/
#       medmnist/*.npz
#       torch_hub/dinov2/
#       torch_hub/checkpoints/*.pth
#       hf_home/local_models/BiomedCLIP-.../
#       hf_home/local_models/BiomedNLP-BiomedBERT-.../
#
# Override the root with either:
#   export BCPT_OFFLINE_PACK=/absolute/path/to/offline_pack
# or:
#   python BCPT_Med_AAAI.py --offline-pack-dir /absolute/path/to/offline_pack ...

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()

OFFLINE_PACK_DIR: Path
MEDMNIST_DIR: Path
DINO_REPO_DIR: Path
DINO_CHECKPOINTS_DIR: Path
HF_HOME_DIR: Path
BIOMEDCLIP_DIR: Path
BIOMEDBERT_DIR: Path
AUTO_DOWNLOAD_MEDMNIST = True


def configure_offline_paths(root: Optional[str] = None, verbose: bool = True) -> Path:
    """Configure all local assets and force supported libraries into offline mode."""
    global OFFLINE_PACK_DIR
    global MEDMNIST_DIR
    global DINO_REPO_DIR
    global DINO_CHECKPOINTS_DIR
    global HF_HOME_DIR
    global BIOMEDCLIP_DIR
    global BIOMEDBERT_DIR

    if root is None:
        root = os.environ.get(
            "BCPT_OFFLINE_PACK",
            os.environ.get("TANGO_OFFLINE_PACK", str(SCRIPT_DIR / "offline_pack")),
        )

    OFFLINE_PACK_DIR = Path(root).expanduser().resolve()
    os.environ["TANGO_OFFLINE_PACK"] = str(OFFLINE_PACK_DIR)
    os.environ["BCPT_OFFLINE_PACK"] = str(OFFLINE_PACK_DIR)
    MEDMNIST_DIR = OFFLINE_PACK_DIR / "medmnist"
    DINO_REPO_DIR = OFFLINE_PACK_DIR / "torch_hub" / "dinov2"
    DINO_CHECKPOINTS_DIR = OFFLINE_PACK_DIR / "torch_hub" / "checkpoints"
    HF_HOME_DIR = OFFLINE_PACK_DIR / "hf_home"
    BIOMEDCLIP_DIR = HF_HOME_DIR / "local_models" / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    BIOMEDBERT_DIR = HF_HOME_DIR / "local_models" / "BiomedNLP-BiomedBERT-base-uncased-abstract"

    # Prevent accidental network access by Hugging Face / Transformers.
    os.environ["HF_HOME"] = str(HF_HOME_DIR)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    if verbose:
        print(f"[offline] pack root: {OFFLINE_PACK_DIR}")
        print(f"[offline] MedMNIST: {MEDMNIST_DIR}")
        print(f"[offline] DINOv2 repo: {DINO_REPO_DIR}")
        print(f"[offline] DINOv2 checkpoints: {DINO_CHECKPOINTS_DIR}")
        print(f"[offline] BiomedCLIP: {BIOMEDCLIP_DIR}")
        print(f"[offline] BiomedBERT: {BIOMEDBERT_DIR}")

    return OFFLINE_PACK_DIR


def require_path(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing offline {what}: {path}\n"
            f"Expected offline pack root: {OFFLINE_PACK_DIR}\n"
            "Place the asset in the expected relative path or pass "
            "--offline-pack-dir /path/to/offline_pack."
        )
    return path


def available_medmnist_size(dataset: str, requested_size: int) -> Tuple[int, Path]:
    """Return an available local MedMNIST size, preferring the requested size."""
    candidates = [requested_size]
    if 224 not in candidates:
        candidates.append(224)
    if 28 not in candidates:
        candidates.append(28)

    for size in candidates:
        suffix = "" if size == 28 else f"_{size}"
        path = MEDMNIST_DIR / f"{dataset}{suffix}.npz"
        if path.is_file():
            return size, path

    matches = sorted(MEDMNIST_DIR.glob(f"{dataset}*.npz")) if MEDMNIST_DIR.exists() else []
    found = ", ".join(p.name for p in matches) if matches else "none"
    raise FileNotFoundError(
        f"No offline MedMNIST file found for dataset '{dataset}' in {MEDMNIST_DIR}. "
        f"Available matching files: {found}"
    )


def dinov2_checkpoint_path(name: str) -> Path:
    """Resolve the exact local DINOv2 checkpoint for a Hub entrypoint name."""
    return DINO_CHECKPOINTS_DIR / f"{name}_pretrain.pth"


def _read_biomedclip_config() -> Dict[str, Any]:
    cfg_path = require_path(BIOMEDCLIP_DIR / "open_clip_config.json", "BiomedCLIP config")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "model_cfg" not in cfg:
        raise ValueError(f"Invalid BiomedCLIP config; missing 'model_cfg': {cfg_path}")
    return cfg


def _patched_biomedclip_text_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Point BiomedCLIP's HF text tower to the local BiomedBERT directory."""
    require_path(BIOMEDBERT_DIR, "BiomedBERT model directory")
    text_cfg = dict(cfg.get("model_cfg", {}).get("text_cfg", {}))
    text_cfg["hf_model_name"] = str(BIOMEDBERT_DIR)
    text_cfg["hf_tokenizer_name"] = str(BIOMEDBERT_DIR)
    return text_cfg


def build_biomedclip_preprocess(cfg: Dict[str, Any]):
    """Build the local BiomedCLIP validation transform from its saved config."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    pp = cfg.get("preprocess_cfg", {})
    mean = pp.get("mean", [0.48145466, 0.4578275, 0.40821073])
    std = pp.get("std", [0.26862954, 0.26130258, 0.27577711])
    image_size = int(cfg.get("model_cfg", {}).get("vision_cfg", {}).get("image_size", 224))

    return transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_biomedclip_offline(device: torch.device):
    """Load BiomedCLIP and its text tower without any Hub/network access.

    Newer OpenCLIP versions support the ``local-dir:`` schema directly. A
    compatibility fallback registers a temporary local architecture config and
    loads the saved checkpoint path explicitly for older OpenCLIP versions.
    """
    import open_clip

    require_path(BIOMEDCLIP_DIR, "BiomedCLIP model directory")
    weight_path = require_path(
        BIOMEDCLIP_DIR / "open_clip_pytorch_model.bin",
        "BiomedCLIP checkpoint",
    )
    cfg = _read_biomedclip_config()
    text_cfg = _patched_biomedclip_text_cfg(cfg)
    preprocess = build_biomedclip_preprocess(cfg)

    local_identifier = f"local-dir:{BIOMEDCLIP_DIR}"
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            local_identifier,
            text_cfg=text_cfg,
            cache_dir=str(HF_HOME_DIR),
        )
        print(f"[offline] loaded BiomedCLIP via {local_identifier}")
    except Exception as local_dir_error:
        print(
            "[offline] OpenCLIP local-dir loader failed; trying compatibility fallback. "
            f"Reason: {local_dir_error}"
        )
        model_cfg = dict(cfg["model_cfg"])
        model_cfg["text_cfg"] = text_cfg

        with tempfile.TemporaryDirectory(prefix="bcpt_biomedclip_cfg_") as tmp:
            cfg_path = Path(tmp) / "biomedclip_offline.json"
            with cfg_path.open("w", encoding="utf-8") as f:
                json.dump(model_cfg, f, indent=2)

            factory = getattr(open_clip, "factory", None)
            if factory is None or not hasattr(factory, "add_model_config"):
                raise RuntimeError(
                    "Installed open_clip version cannot load local-dir models and does not expose "
                    "factory.add_model_config for the offline compatibility fallback."
                ) from local_dir_error

            factory.add_model_config(str(cfg_path))
            model, _, _ = open_clip.create_model_and_transforms(
                "biomedclip_offline",
                pretrained=str(weight_path),
            )
        print(f"[offline] loaded BiomedCLIP checkpoint: {weight_path}")

    model = model.eval().to(device)
    return model, preprocess


# Configure defaults at import time so notebook helper functions also work.
configure_offline_paths(verbose=False)


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def to_numpy_int(x) -> np.ndarray:
    return np.asarray(x).astype(int).reshape(-1)


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def start_resource_measurement(device: torch.device) -> Dict[str, float]:
    """Start a lightweight wall-time and peak-memory measurement."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return {"wall_start": time.perf_counter()}


def finish_resource_measurement(mark: Dict[str, float], device: torch.device) -> Dict[str, float]:
    """Return elapsed seconds plus CUDA and process peak memory in MiB."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        cuda_peak = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    else:
        cuda_peak = 0.0
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    rss_mib = rss / (1024.0 ** 2) if platform.system() == "Darwin" else rss / 1024.0
    return {
        "RuntimeSec": float(time.perf_counter() - mark["wall_start"]),
        "PeakCudaMiB": float(cuda_peak),
        "PeakProcessMiB": float(rss_mib),
    }


# =============================================================================
# 3. Metrics: Hungarian-matched clustering evaluation
# =============================================================================


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict[int, int]]:
    y_true = to_numpy_int(y_true)
    y_pred = to_numpy_int(y_pred)
    pred_ids = np.unique(y_pred)
    true_ids = np.unique(y_true)
    mat = np.zeros((len(pred_ids), len(true_ids)), dtype=np.int64)
    pred_to_row = {p: i for i, p in enumerate(pred_ids)}
    true_to_col = {t: j for j, t in enumerate(true_ids)}
    for p, t in zip(y_pred, y_true):
        mat[pred_to_row[p], true_to_col[t]] += 1
    row_ind, col_ind = linear_sum_assignment(-mat)
    matched = mat[row_ind, col_ind].sum()
    mapping = {int(pred_ids[r]): int(true_ids[c]) for r, c in zip(row_ind, col_ind)}
    return float(matched / max(len(y_true), 1)), mapping


def apply_mapping(y_pred: np.ndarray, mapping: Dict[int, int], fallback: int = 0) -> np.ndarray:
    return np.asarray([mapping.get(int(p), fallback) for p in y_pred], dtype=int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> Dict[str, float]:
    y_true = to_numpy_int(y_true)
    y_pred = to_numpy_int(y_pred)
    acc, mapping = clustering_accuracy(y_true, y_pred)
    fallback = int(np.bincount(y_true).argmax()) if len(y_true) else 0
    y_map = apply_mapping(y_pred, mapping, fallback)

    counts = np.bincount(y_pred, minlength=k)
    fracs = counts / max(counts.sum(), 1)
    active = int(np.sum(fracs >= 0.01))

    labels = np.unique(y_true)
    recalls = recall_score(y_true, y_map, labels=labels, average=None, zero_division=0)
    true_counts = np.array([(y_true == c).sum() for c in labels])
    rare_idx = int(np.argmin(true_counts)) if len(true_counts) else 0

    # Frequency-ranked class groups are more stable and informative than a
    # single rare class.  They are defined only for evaluation and never enter
    # training or label-free model selection.
    order = np.argsort(-true_counts)
    groups = np.array_split(order, 3)

    def group_recall(g: np.ndarray) -> float:
        return float(recalls[g].mean()) if len(g) else float("nan")

    metrics = {
        "ACC": acc,
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "MacroF1": float(f1_score(y_true, y_map, average="macro", zero_division=0)),
        "BalancedACC": float(balanced_accuracy_score(y_true, y_map)),
        "RareRecall": float(recalls[rare_idx]) if len(recalls) else 0.0,
        "WorstRecall": float(recalls.min()) if len(recalls) else 0.0,
        "HeadRecall": group_recall(groups[0]),
        "MediumRecall": group_recall(groups[1]),
        "TailRecall": group_recall(groups[2]),
        "ActiveClusters": float(active),
        "MinClusterFrac": float(fracs.min()) if len(fracs) else 0.0,
        "MaxClusterFrac": float(fracs.max()) if len(fracs) else 0.0,
        "CollapseFlag": float(active < k),
    }
    for label, value in zip(labels, recalls):
        metrics[f"RecallClass{int(label)}"] = float(value)
    return metrics


def summarize_rows(rows: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    if not rows:
        return {}
    keys = [kk for kk in rows[0].keys() if kk not in {"seed", "method", "dataset", "encoders"}]
    out = {}
    for kk in keys:
        vals = np.asarray([float(r[kk]) for r in rows if kk in r], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if len(finite):
            out[kk] = (float(np.mean(finite)), float(np.std(finite)))
        else:
            out[kk] = (float("nan"), float("nan"))
    return out


# =============================================================================
# 4. Feature extraction: MedMNIST + frozen foundation encoders
# =============================================================================


def load_medmnist_dataset(dataset: str, split: str, img_size: int):
    """Load MedMNIST locally, downloading once into ``offline_pack`` if absent.

    Existing files are always reused.  When automatic acquisition is enabled,
    the official MedMNIST API downloads the requested resolution to
    ``offline_pack/medmnist``.  If a large-resolution artifact is unavailable,
    the canonical 28x28 artifact is downloaded and encoder preprocessing still
    resizes it to ``img_size``.  No labels enter clustering or model selection.
    """
    import medmnist
    from medmnist import INFO

    dataset = dataset.lower()
    if dataset not in INFO:
        raise ValueError(
            f"Unknown MedMNIST dataset: {dataset}. "
            f"Available keys include: {list(INFO.keys())[:10]} ..."
        )

    MEDMNIST_DIR.mkdir(parents=True, exist_ok=True)
    info = INFO[dataset]
    DataClass = getattr(medmnist, info["python_class"])
    try:
        native_size, npz_path = available_medmnist_size(dataset, img_size)
    except FileNotFoundError:
        if not AUTO_DOWNLOAD_MEDMNIST:
            raise
        errors = []
        requested_sizes = [img_size] + ([28] if img_size != 28 else [])
        for candidate_size in requested_sizes:
            print(
                f"[download] {dataset} size={candidate_size} is missing; "
                f"downloading with the official MedMNIST API to {MEDMNIST_DIR}"
            )
            download_kwargs = {
                "split": split,
                "download": True,
                "root": str(MEDMNIST_DIR),
            }
            if candidate_size != 28:
                download_kwargs["size"] = candidate_size
            try:
                DataClass(**download_kwargs)
                native_size, npz_path = available_medmnist_size(dataset, candidate_size)
                break
            except Exception as exc:
                errors.append(f"size={candidate_size}: {type(exc).__name__}: {exc}")
        else:
            raise RuntimeError(
                f"Automatic download failed for {dataset}. Tried: " + " | ".join(errors)
            )

    kwargs = {
        "split": split,
        "download": False,
        "root": str(MEDMNIST_DIR),
    }
    if native_size != 28:
        kwargs["size"] = native_size

    try:
        ds = DataClass(**kwargs)
    except TypeError as e:
        # Compatibility with older MedMNIST releases lacking `size` support.
        if native_size != 28:
            raise RuntimeError(
                f"Installed MedMNIST version does not support loading native size={native_size}. "
                "Install medmnist>=3.0.2, but keep the dataset itself offline."
            ) from e
        ds = DataClass(split=split, download=False, root=str(MEDMNIST_DIR))

    print(f"[offline] MedMNIST file: {npz_path}")
    if native_size != img_size:
        print(
            f"[offline] requested img_size={img_size}, using local native size={native_size}; "
            "the encoder preprocessing will resize images."
        )

    labels = np.asarray(ds.labels).reshape(-1).astype(int)
    k = len(info["label"])
    return ds, labels, k, info, native_size


def build_preprocess(img_size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def pil_batch_from_dataset(ds, start: int, end: int):
    images = []
    for i in range(start, end):
        item = ds[i]
        img = item[0] if isinstance(item, (tuple, list)) else item
        images.append(img)
    return images


def build_encoder(name: str, device: torch.device):
    """Return (kind, encoder, out_dim, preprocess_or_None), strictly offline."""
    name = name.lower()

    if name == "resnet50":
        from torchvision import models

        weights_enum = models.ResNet50_Weights.DEFAULT
        checkpoint_name = os.path.basename(weights_enum.url)
        checkpoint_path = DINO_CHECKPOINTS_DIR / checkpoint_name
        require_path(
            checkpoint_path,
            "ResNet-50 checkpoint (not included in the supplied offline pack by default)",
        )

        net = models.resnet50(weights=None)
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")
        net.load_state_dict(state, strict=True)
        net = nn.Sequential(*list(net.children())[:-1]).eval().to(device)
        print(f"[offline] loaded ResNet-50 checkpoint: {checkpoint_path}")
        return "resnet", net, 2048, None

    if name.startswith("dinov2"):
        supported_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        if name not in supported_dims:
            raise ValueError(
                f"Unsupported DINOv2 encoder '{name}'. "
                f"Supported offline names: {list(supported_dims.keys())}"
            )

        repo = require_path(DINO_REPO_DIR, "DINOv2 repository")
        checkpoint = require_path(dinov2_checkpoint_path(name), f"{name} checkpoint")

        # PyTorch Hub source='local' imports hubconf.py from the local clone. The
        # DINOv2 entrypoint accepts a local `weights` path and loads it directly.
        enc = torch.hub.load(
            str(repo),
            name,
            source="local",
            pretrained=True,
            weights=str(checkpoint),
        ).eval().to(device)
        print(f"[offline] loaded {name}: {checkpoint}")
        return "dino", enc, supported_dims[name], None

    if name == "biomedclip":
        model, preprocess = load_biomedclip_offline(device)
        return "biomedclip", model, 512, preprocess

    raise ValueError(
        f"Unknown encoder '{name}'. Supported: resnet50, dinov2_vits14, "
        "dinov2_vitb14, dinov2_vitl14, dinov2_vitg14, biomedclip"
    )


@torch.no_grad()
def extract_features_for_encoder(
    ds,
    enc_name: str,
    img_size: int,
    batch_size: int,
    device: torch.device,
    amp: bool = True,
) -> np.ndarray:
    kind, enc, dim, external_preprocess = build_encoder(enc_name, device)
    tf = build_preprocess(img_size)
    feats = []
    n = len(ds)
    use_amp = amp and device.type == "cuda"

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        pil = pil_batch_from_dataset(ds, s, e)
        if kind == "biomedclip":
            x = torch.stack([external_preprocess(im.convert("RGB")) for im in pil], 0).to(device)
        else:
            x = torch.stack([tf(im) for im in pil], 0).to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if kind == "biomedclip":
                f = enc.encode_image(x)
            elif kind == "dino":
                f = enc(x)
            else:
                f = enc(x).flatten(1)
        feats.append(f.float().cpu().numpy())

        if (s // batch_size) % 20 == 0:
            print(f"    [{enc_name}] {e}/{n}")

    del enc
    clean_memory()
    return np.concatenate(feats, axis=0).astype(np.float32)


def load_medmnist_views(
    dataset: str,
    split: str,
    encoders: List[str],
    img_size: int,
    batch_size: int,
    device: torch.device,
    cache_dir: str,
    amp: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """Extract one frozen feature view per encoder and cache each as .npz."""
    ensure_dir(cache_dir)
    ds, labels, k, info, native_size = load_medmnist_dataset(dataset, split, img_size)
    print(f"[MedMNIST] {dataset}/{split}: N={len(ds)}, classes={k}, native_size={native_size}")
    print(f"[MedMNIST] task={info.get('task', 'unknown')} modality={info.get('modality', 'unknown')}")

    views = []
    for enc_name in encoders:
        cache = os.path.join(cache_dir, f"{dataset.lower()}_{split}_{enc_name.lower()}_{img_size}.npz")
        if os.path.exists(cache):
            X = np.load(cache)["X"].astype(np.float32)
            print(f"[cache] {cache}: {X.shape}")
        else:
            print(f"[extract] {dataset}/{split} using {enc_name}")
            X = extract_features_for_encoder(ds, enc_name, img_size, batch_size, device, amp=amp)
            np.savez_compressed(cache, X=X)
            print(f"[saved] {cache}: {X.shape}")
        views.append(l2norm(X))
    return views, labels, k


def load_precomputed_views(view_paths: List[str], label_path: str):
    views = []
    for p in view_paths:
        arr = np.load(p, allow_pickle=True)
        X = arr["X"] if "X" in arr else arr[arr.files[0]]
        views.append(l2norm(np.asarray(X, dtype=np.float32)))
    y = np.load(label_path).astype(int).reshape(-1)
    k = int(len(np.unique(y)))
    return views, y, k


def make_long_tailed_indices(
    labels: np.ndarray,
    k: int,
    imbalance_ratio: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a deterministic exponential long-tail benchmark subset.

    Ground-truth labels are used only for benchmark construction.  The returned
    indices are applied before training, after which all learning and model
    selection remain label-free.  Class order is seeded so results should be
    averaged over multiple ``imbalance_seed`` values when category difficulty
    may be confounded with head/tail status.
    """
    if imbalance_ratio < 1.0:
        raise ValueError("imbalance_ratio must be >= 1")
    labels = np.asarray(labels, dtype=int).reshape(-1)
    counts = np.bincount(labels, minlength=k)
    if np.any(counts == 0):
        raise ValueError("Controlled long-tail construction requires every class to be present")
    rng = np.random.default_rng(seed)
    class_order = rng.permutation(k)
    if k == 1:
        relative = np.ones(1, dtype=np.float64)
    else:
        relative = imbalance_ratio ** (-np.arange(k, dtype=np.float64) / (k - 1))
    # Largest feasible scale preserves the requested ratio without oversampling.
    available = counts[class_order].astype(np.float64)
    scale = float(np.min(available / relative))
    targets_ordered = np.maximum(1, np.floor(scale * relative).astype(int))
    targets = np.zeros(k, dtype=int)
    targets[class_order] = targets_ordered

    selected = []
    for c in range(k):
        ids = np.flatnonzero(labels == c)
        chosen = rng.choice(ids, size=min(targets[c], len(ids)), replace=False)
        selected.append(chosen)
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return indices.astype(np.int64), targets


STRESS_TESTS = [
    "noisy_encoder", "weak_anchor", "corrupted_graph", "reduced_sample",
    "wrong_k", "distribution_shift",
]


def apply_feature_stress(
    views: List[np.ndarray],
    labels: np.ndarray,
    k: int,
    stress_test: Optional[str],
    strength: float,
    sample_fraction: float,
    wrong_k_delta: int,
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray, int, Dict[str, Any]]:
    """Apply a deterministic stress protocol without using labels for selection."""
    if stress_test in {None, "none", "clean"}:
        return views, labels, k, {"stress_test": "clean"}
    if stress_test not in STRESS_TESTS:
        raise ValueError(f"Unknown stress test '{stress_test}'. Valid: {STRESS_TESTS}")
    rng = np.random.default_rng(seed + 1717)
    out = [np.asarray(v, dtype=np.float32).copy() for v in views]
    y = np.asarray(labels, dtype=int).copy()
    k_used = int(k)
    meta: Dict[str, Any] = {"stress_test": stress_test, "strength": float(strength)}

    if stress_test == "noisy_encoder":
        target = len(out) - 1
        noise = rng.normal(0.0, strength / math.sqrt(out[target].shape[1]), out[target].shape)
        out[target] = l2norm(out[target] + noise.astype(np.float32))
        meta["affected_view"] = int(target)
    elif stress_test == "reduced_sample":
        keep_n = max(2 * k, int(round(len(y) * float(np.clip(sample_fraction, 0.05, 1.0)))))
        keep = np.sort(rng.choice(len(y), size=min(keep_n, len(y)), replace=False))
        out = [v[keep] for v in out]
        y = y[keep]
        meta["kept_samples"] = int(len(y))
        meta["sample_fraction"] = float(len(y) / max(len(labels), 1))
    elif stress_test == "wrong_k":
        k_used = max(2, int(k + wrong_k_delta))
        if k_used == k:
            k_used = k + 1
        meta.update({"true_k": int(k), "used_k": int(k_used)})
    elif stress_test == "distribution_shift":
        affected = rng.random(len(y)) < 0.5
        for vi, v in enumerate(out):
            direction = rng.normal(size=v.shape[1]).astype(np.float32)
            direction /= max(np.linalg.norm(direction), 1e-8)
            noise = rng.normal(0.0, 0.25 * strength / math.sqrt(v.shape[1]), (affected.sum(), v.shape[1]))
            v[affected] += strength * direction[None, :] + noise.astype(np.float32)
            out[vi] = l2norm(v)
        meta["affected_fraction"] = float(affected.mean())
    # weak_anchor is applied after the label-free anchor is constructed;
    # corrupted_graph is applied after graph construction.
    return out, y, k_used, meta


@torch.no_grad()
def corrupt_graph_tensors(
    nbr_idx: torch.Tensor,
    nbr_w: torch.Tensor,
    graph_details: Dict[str, torch.Tensor],
    fraction: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Randomly rewire a fraction of graph edges for safeguard stress testing."""
    frac = float(np.clip(fraction, 0.0, 1.0))
    if frac <= 0:
        return nbr_idx, nbr_w, graph_details
    gen = torch.Generator(device="cpu").manual_seed(seed + 2881)
    idx = nbr_idx.clone()
    mask = torch.rand(idx.shape, generator=gen).to(idx.device) < frac
    replacement = torch.randint(0, idx.shape[0], idx.shape, generator=gen).to(idx.device)
    idx[mask] = replacement[mask]
    weights = nbr_w.clone()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    details = {key: value.clone() for key, value in graph_details.items()}
    view_idx = details["view_idx"]
    view_mask = torch.rand(view_idx.shape, generator=gen).to(view_idx.device) < frac
    view_repl = torch.randint(0, idx.shape[0], view_idx.shape, generator=gen).to(view_idx.device)
    view_idx[view_mask] = view_repl[view_mask]
    details["view_idx"] = view_idx
    details["agreement"] = details["agreement"] * (1.0 - frac)
    return idx, weights, details


# =============================================================================
# 5. Consensus neighbor graph
# =============================================================================


@torch.no_grad()
def cosine_knn(X: torch.Tensor, k: int, chunk: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return top-k cosine neighbors excluding self. X must be L2 normalized."""
    N = X.shape[0]
    k = min(k, max(1, N - 1))
    idx = torch.empty(N, k, dtype=torch.long, device=X.device)
    sim = torch.empty(N, k, dtype=X.dtype, device=X.device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        S = X[s:e] @ X.t()
        rows = torch.arange(s, e, device=X.device)
        S[torch.arange(e - s, device=X.device), rows] = -2.0
        tv, ti = torch.topk(S, k, dim=1)
        idx[s:e] = ti
        sim[s:e] = tv
        del S
    return idx, sim


@torch.no_grad()
def build_consensus_graph(
    views: List[torch.Tensor],
    k_nn: int,
    mutual: bool = True,
    knn_chunk: int = 1024,
    return_details: bool = False,
):
    """Fuse per-view kNN lists into a consensus weighted neighbor list.

    Candidate score = (#views where j is neighbor of i / #views) * mean normalized cosine.
    If mutual=True, keep only reciprocal neighbors within each view. For one view,
    mutual=False is usually safer because reciprocal filtering can over-prune.
    """
    V = len(views)
    N = views[0].shape[0]
    device = views[0].device
    per_view = [cosine_knn(x, k_nn, chunk=knn_chunk) for x in views]

    recip_sets = []
    if mutual:
        for vi, _ in per_view:
            vi_cpu = vi.cpu().numpy()
            rs = [set() for _ in range(N)]
            for i in range(N):
                for j in vi_cpu[i]:
                    rs[int(j)].add(i)
            recip_sets.append(rs)

    cpu_idx = [vi.cpu().numpy() for vi, _ in per_view]
    cpu_sim = [vs.cpu().numpy() for _, vs in per_view]
    nbr_idx_list, nbr_w_list = [], []
    max_k = 1

    for i in range(N):
        agg: Dict[int, List[float]] = {}
        cnt: Dict[int, int] = {}
        for v in range(V):
            for slot, j in enumerate(cpu_idx[v][i]):
                j = int(j)
                if mutual and i not in recip_sets[v][j]:
                    continue
                agg.setdefault(j, []).append(float(cpu_sim[v][i][slot]))
                cnt[j] = cnt.get(j, 0) + 1

        # fallback prevents empty rows after mutual-kNN filtering
        if not agg:
            for slot, j in enumerate(cpu_idx[0][i]):
                j = int(j)
                agg.setdefault(j, []).append(float(cpu_sim[0][i][slot]))
                cnt[j] = cnt.get(j, 0) + 1

        js = list(agg.keys())
        w = np.array([(cnt[j] / V) * (np.mean(agg[j]) * 0.5 + 0.5) for j in js], dtype=np.float32)
        order = np.argsort(-w)
        js = [js[o] for o in order]
        w = w[order]
        nbr_idx_list.append(js)
        nbr_w_list.append(w)
        max_k = max(max_k, len(js))

    nbr_idx = np.zeros((N, max_k), dtype=np.int64)
    nbr_w = np.zeros((N, max_k), dtype=np.float32)
    for i in range(N):
        m = len(nbr_idx_list[i])
        if m == 0:
            nbr_idx[i, 0] = i
            nbr_w[i, 0] = 1.0
        else:
            nbr_idx[i, :m] = nbr_idx_list[i]
            nbr_w[i, :m] = nbr_w_list[i]
            ssum = nbr_w[i, :m].sum()
            if ssum > 0:
                nbr_w[i, :m] /= ssum

    out_idx = torch.from_numpy(nbr_idx).to(device)
    out_w = torch.from_numpy(nbr_w).to(device)
    if not return_details:
        return out_idx, out_w

    # Preserve view-specific neighbourhoods for uncertainty estimation.  Their
    # weights are positive, normalized cosine affinities.  Agreement is a
    # label-free reliability score: pairwise neighbour-set Jaccard for multiple
    # encoders, modulated by local similarity quality.  In the single-view case
    # it reduces to local similarity quality rather than pretending that
    # cross-view agreement was observed.
    view_idx = np.stack(cpu_idx, axis=0).astype(np.int64)
    view_w = np.stack(
        [np.clip((s + 1.0) * 0.5, 1e-6, None) for s in cpu_sim], axis=0
    ).astype(np.float32)
    view_w /= np.clip(view_w.sum(axis=2, keepdims=True), 1e-8, None)
    sim_quality = np.mean(
        np.stack([np.clip((s + 1.0) * 0.5, 0.0, 1.0) for s in cpu_sim], axis=0),
        axis=(0, 2),
    )
    if V == 1:
        agreement = sim_quality
    else:
        pair_scores = []
        for va in range(V):
            for vb in range(va + 1, V):
                scores = np.zeros(N, dtype=np.float32)
                for i in range(N):
                    a = set(int(x) for x in cpu_idx[va][i])
                    b = set(int(x) for x in cpu_idx[vb][i])
                    scores[i] = len(a & b) / max(len(a | b), 1)
                pair_scores.append(scores)
        set_agreement = np.mean(np.stack(pair_scores, axis=0), axis=0)
        agreement = np.sqrt(np.clip(set_agreement * sim_quality, 0.0, 1.0))

    details = {
        "view_idx": torch.from_numpy(view_idx).to(device),
        "view_w": torch.from_numpy(view_w).to(device),
        "agreement": torch.from_numpy(agreement.astype(np.float32)).to(device),
    }
    return out_idx, out_w, details


# =============================================================================
# 6. Initialization
# =============================================================================


def spherical_kmeans_init(Xcat: np.ndarray, k: int, n_init: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """KMeans on L2-normalized features, selected by its label-free inertia."""
    best = None
    for r in range(n_init):
        km = KMeans(n_clusters=k, n_init=1, random_state=seed + r, max_iter=300, algorithm="lloyd").fit(Xcat)
        objective = float(km.inertia_)
        if best is None or objective < best[0]:
            best = (objective, km.labels_.astype(int), km.cluster_centers_.astype(np.float32))
    return best[1], best[2]


# =============================================================================
# 7. BCPT-Med model and losses
# =============================================================================


class CredalConsensusMarginal:
    """EMA marginal with label-free, view-derived uncertainty intervals.

    Each encoder supplies a graph-smoothed teacher histogram.  Their dispersion
    estimates cross-foundation epistemic disagreement, while an effective-sample
    term prevents overconfident bounds when only a few reliable samples support
    a cluster.  The EMA centre is always feasible inside the returned box.
    """

    def __init__(
        self,
        k: int,
        device: torch.device,
        floor_frac: float = 0.05,
        eta: float = 0.10,
        bound_scale: float = 2.0,
        min_radius_frac: float = 0.05,
    ):
        self.k = k
        self.device = device
        self.eta = eta
        self.floor = floor_frac / k
        self.bound_scale = bound_scale
        self.min_radius = min_radius_frac / k
        self.pi = torch.ones(k, device=device) / k
        self.lower = self.pi.clone()
        self.upper = self.pi.clone()
        self.radius = torch.zeros(k, device=device)
        self.effective_n = torch.tensor(0.0, device=device)

    @torch.no_grad()
    def update(
        self,
        probs: torch.Tensor,
        view_idx: torch.Tensor,
        view_w: torch.Tensor,
        agreement: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if probs.ndim != 2 or probs.shape[1] != self.k:
            raise ValueError(f"Expected probabilities [N,{self.k}], got {tuple(probs.shape)}")
        if view_idx.ndim != 3 or view_w.shape != view_idx.shape:
            raise ValueError("view_idx/view_w must have shape [V,N,k_nn]")

        confidence = probs.max(dim=1).values
        chance = 1.0 / self.k
        calibrated_conf = ((confidence - chance) / max(1.0 - chance, 1e-6)).clamp(0.0, 1.0)
        sample_w = (agreement.clamp(0.0, 1.0) * calibrated_conf).clamp_min(1e-4)
        hists = []
        for v in range(view_idx.shape[0]):
            local = probs[view_idx[v]]
            smooth = (view_w[v].unsqueeze(-1) * local).sum(dim=1)
            hist = (sample_w.unsqueeze(1) * smooth).sum(dim=0) / sample_w.sum().clamp_min(1e-8)
            hist = hist.clamp_min(self.floor)
            hists.append(hist / hist.sum())

        H = torch.stack(hists, dim=0)
        estimate = H.mean(dim=0)
        estimate = estimate.clamp_min(self.floor)
        estimate = estimate / estimate.sum()
        self.pi = (1.0 - self.eta) * self.pi + self.eta * estimate
        self.pi = self.pi.clamp_min(self.floor)
        self.pi = self.pi / self.pi.sum()

        w_sum = sample_w.sum()
        n_eff = w_sum.square() / sample_w.square().sum().clamp_min(1e-8)
        sampling_se = torch.sqrt(
            self.pi * (1.0 - self.pi) / n_eff.clamp_min(1.0)
        )
        if H.shape[0] > 1:
            view_se = H.std(dim=0, unbiased=False)
        else:
            # A single encoder cannot estimate cross-view epistemic spread.
            # Retain a conservative slack determined by its predictive entropy.
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
            entropy = (entropy / math.log(max(self.k, 2))).mean()
            view_se = torch.full_like(self.pi, float(entropy) * self.min_radius)

        radius = self.bound_scale * (view_se + sampling_se)
        radius = radius.clamp_min(self.min_radius)
        lower = torch.maximum(self.pi - radius, torch.full_like(self.pi, self.floor))
        upper = torch.minimum(self.pi + radius, torch.ones_like(self.pi))
        # The centre remains component-wise feasible, hence sum(lower)<=1<=sum(upper).
        self.lower = torch.minimum(lower, self.pi)
        self.upper = torch.maximum(upper, self.pi)
        self.radius = radius
        self.effective_n = n_eff
        return self.pi.detach(), self.lower.detach(), self.upper.detach()

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pi.detach(), self.lower.detach(), self.upper.detach()


def project_box_simplex(
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    total: float = 1.0,
    iters: int = 80,
) -> torch.Tensor:
    """Euclidean projection onto {x: sum(x)=total, lower<=x<=upper}."""
    # Feasibility is defined by bounds that are usually produced in float32.
    # In particular, K copies of the representable value ``1 / K`` need not
    # reduce to exactly one (e.g. K=11 on common CPU/CUDA reductions).  Preserve
    # genuine infeasibility, but repair round-off-sized boundary violations
    # before solving the projection in float64.
    input_dtype = value.dtype
    value = value.double()
    lower = lower.double().clamp_min(0.0)
    upper = torch.maximum(upper.double(), lower)
    z = torch.as_tensor(total, dtype=value.dtype, device=value.device)
    source_eps = (
        torch.finfo(input_dtype).eps
        if input_dtype.is_floating_point
        else torch.finfo(torch.float64).eps
    )
    feasibility_tol = max(
        1e-12,
        32.0 * float(source_eps) * max(1, lower.numel()) * max(1.0, abs(float(z))),
    )
    lower_sum = lower.sum()
    upper_sum = upper.sum()
    if lower_sum > z and lower_sum <= z + feasibility_tol:
        lower = lower * (z / lower_sum.clamp_min(1e-30))
        lower_sum = lower.sum()
    if upper_sum < z and upper_sum >= z - feasibility_tol:
        upper = upper * (z / upper_sum.clamp_min(1e-30))
        upper_sum = upper.sum()
    upper = torch.maximum(upper, lower)
    if lower_sum > z or upper_sum < z:
        raise ValueError(
            f"Infeasible bounded simplex: sum(lower)={float(lower_sum):.12g}, "
            f"sum(upper)={float(upper_sum):.12g}, total={float(z):.12g}, "
            f"tolerance={feasibility_tol:.3g}"
        )
    lo = torch.min(value - upper) - 1.0
    hi = torch.max(value - lower) + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        x = torch.clamp(value - mid, min=lower, max=upper)
        if x.sum() > z:
            lo = mid
        else:
            hi = mid
    x = torch.clamp(value - 0.5 * (lo + hi), min=lower, max=upper)
    residual = z - x.sum()
    if residual > 0:
        slack = (upper - x).clamp_min(0.0)
        x = x + residual * slack / slack.sum().clamp_min(1e-12)
    elif residual < 0:
        slack = (x - lower).clamp_min(0.0)
        x = x + residual * slack / slack.sum().clamp_min(1e-12)
    return x.to(dtype=value.dtype).float()


@torch.no_grad()
def bounded_partial_sinkhorn(
    logits: torch.Tensor,
    prior: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    reliability: torch.Tensor,
    transported_mass: float,
    epsilon: float = 0.07,
    iters: int = 40,
    evidence_mix: float = 0.50,
    reject_margin: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Credal target selection followed by genuine partial OT.

    First, batch evidence is projected onto the interval-valued target-marginal
    set.  Second, a reject column receives exactly ``1-transported_mass``.  The
    returned real-cluster matrix is *not* row-normalized: its row sum is the
    sample's accepted mass and must be retained by the learning objective.
    """
    B, K = logits.shape
    rho = float(np.clip(transported_mass, 1e-4, 1.0))
    reliability = reliability.reshape(-1).clamp(0.0, 1.0)
    if len(reliability) != B:
        raise ValueError("reliability must contain one score per sample")

    probs = F.softmax(logits, dim=1)
    weighted_demand = (reliability.unsqueeze(1) * probs).sum(dim=0)
    weighted_demand = weighted_demand / reliability.sum().clamp_min(1e-8)
    proposal = evidence_mix * weighted_demand + (1.0 - evidence_mix) * prior
    target_prop = project_box_simplex(proposal, lower, upper, total=1.0)

    scores = logits
    if rho < 1.0 - 1e-6:
        top2 = torch.topk(scores, k=min(2, K), dim=1).values
        if K > 1:
            gap = (top2[:, 0] - top2[:, 1]).clamp_min(0.05)
        else:
            gap = torch.ones(B, device=logits.device, dtype=logits.dtype)
        reject_score = top2[:, 0] + reject_margin * gap * (1.0 - 2.0 * reliability)
        aug_scores = torch.cat([scores, reject_score.unsqueeze(1)], dim=1)
        target = torch.cat(
            [rho * target_prop, torch.tensor([1.0 - rho], device=logits.device)]
        )
    else:
        aug_scores = scores
        target = target_prop

    log_kernel = aug_scores / max(epsilon, 1e-6)
    log_kernel = log_kernel - log_kernel.max(dim=1, keepdim=True).values
    log_row = torch.full((B,), -math.log(B), device=logits.device, dtype=logits.dtype)
    log_col = target.clamp_min(1e-12).log().to(dtype=logits.dtype)
    f = torch.zeros_like(log_row)
    g = torch.zeros_like(log_col)
    for _ in range(iters):
        f = log_row - torch.logsumexp(log_kernel + g.unsqueeze(0), dim=1)
        g = log_col - torch.logsumexp(log_kernel + f.unsqueeze(1), dim=0)
    plan = torch.exp(log_kernel + f.unsqueeze(1) + g.unsqueeze(0))

    # Multiplication by B converts probability mass to per-sample mass.  Rows
    # sum to one across real clusters plus reject, while mean accepted mass=rho.
    per_sample_plan = plan * B
    real_mass = per_sample_plan[:, :K]
    accept_mass = real_mass.sum(dim=1)
    if per_sample_plan.shape[1] == K + 1:
        reject_mass = per_sample_plan[:, -1]
    else:
        reject_mass = torch.zeros(B, device=logits.device, dtype=logits.dtype)
    return real_mass, accept_mass, reject_mass, target_prop


class ClusterHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, proj_dim: int, k: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, proj_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(k, proj_dim) * 0.02)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.1)))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x), dim=-1)

    def logits_from_z(self, z: torch.Tensor) -> torch.Tensor:
        protos = F.normalize(self.prototypes, dim=-1)
        tau = self.log_tau.exp().clamp(0.02, 1.0)
        return (z @ protos.t()) / tau

    def forward(self, x: torch.Tensor):
        z = self.embed(x)
        logits = self.logits_from_z(z)
        return z, logits, F.softmax(logits, dim=-1)


def prior_normalized_neighbor_agreement(
    p_student: torch.Tensor,
    p_teacher_nbr: torch.Tensor,
    nbr_w: torch.Tensor,
    pi: torch.Tensor,
    beta: float = 0.6,
) -> torch.Tensor:
    """Negative log prior-normalized label-collision probability.

    This is deliberately not called pointwise mutual information: it is a
    differentiable collision-probability surrogate whose class contribution is
    tempered by ``pi**beta``.  ``beta=0`` removes prior correction and
    ``beta=1`` applies full inverse-prior correction.
    """
    denom = pi.clamp_min(1e-6).pow(beta).view(1, 1, -1)
    agree = (p_student.unsqueeze(1) * p_teacher_nbr / denom).sum(dim=2)
    agree = agree.clamp_min(1e-8)
    per_anchor = (nbr_w * torch.log(agree)).sum(dim=1)
    return -per_anchor.mean()


# Legacy import alias.
pmi_self_distillation = prior_normalized_neighbor_agreement


def prototype_separation(head: ClusterHead) -> torch.Tensor:
    protos = F.normalize(head.prototypes, dim=-1)
    sim = protos @ protos.t()
    k = sim.size(0)
    off = sim[~torch.eye(k, dtype=torch.bool, device=sim.device)]
    return (off.clamp_min(0) ** 2).mean()


def anti_collapse_floor(p_batch_mean: torch.Tensor, floor: torch.Tensor) -> torch.Tensor:
    return F.relu(floor - p_batch_mean).sum()


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, m: float):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1 - m)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)


def gather_neighbor_probs(
    teacher_probs_all: torch.Tensor,
    nbr_idx: torch.Tensor,
    nbr_w: torch.Tensor,
    batch_ids: torch.Tensor,
    m: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = nbr_idx[batch_ids][:, :m]
    w = nbr_w[batch_ids][:, :m].clone()
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    pt = teacher_probs_all[idx]
    return pt, w


@dataclass
class BCPTConfig:
    epochs: int = 60
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 512
    proj_dim: int = 128
    dropout: float = 0.1
    k_nn: int = 20
    nbr_per_anchor: int = 10
    knn_chunk: int = 1024
    ema_momentum: float = 0.996
    beta_pmi: float = 0.6
    w_pmi: float = 1.0
    w_ot: float = 0.35
    w_sep: float = 0.5
    w_floor: float = 3.0
    floor_frac_uniform: float = 0.35
    ot_epsilon: float = 0.07
    ot_iters: int = 40
    transported_mass_start: float = 0.35
    transported_mass_end: float = 0.95
    transported_mass_warmup: float = 0.70
    evidence_mix: float = 0.50
    reject_margin: float = 1.0
    conf_tau_base: float = 0.90
    conf_tau_tail: float = 0.15
    purity_rho: float = 0.50
    selflabel_start: float = 0.45
    balance_warmup: float = 0.35
    marginal_floor_frac: float = 0.01
    marginal_eta: float = 0.10
    credal_bound_scale: float = 2.0
    credal_min_radius_frac: float = 0.02
    # Paper ablations:
    # ot_mode = "none"            -> prior-corrected neighbour objective only
    # ot_mode = "uniform"         -> conventional full-mass uniform Sinkhorn
    # ot_mode = "bounded_full"    -> credal target but no rejected mass
    # ot_mode = "bounded_partial" -> partial OT; the proposed configuration
    #                                  additionally enables anchor safeguards
    ot_mode: str = "bounded_partial"
    # pmi_mode is retained as a legacy config key: "credal" uses the estimated
    # marginal in prior-normalized neighbour agreement; "uniform" ablates it.
    pmi_mode: str = "credal"
    reliability_weighting: bool = True
    reliable_self_label: bool = True
    # Proposed robustness components.  The anchor is obtained by KMeans on the
    # same frozen features and never uses labels.  A weak, class-balanced
    # anchor loss prevents rare initialization clusters from being erased by
    # noisy self-labels.  At model selection time, the trained partition must
    # improve a label-free graph/cohesion score over that anchor; otherwise the
    # method safely returns the anchor partition instead of a degraded head.
    anchor_regularization: bool = True
    anchor_weight_start: float = 0.30
    anchor_weight_end: float = 0.05
    anchor_tail_power: float = 0.50
    anchor_corruption_frac: float = 0.0
    anchor_guard: bool = True
    anchor_fallback: bool = True
    anchor_guard_min_gain: float = 0.0
    anchor_guard_nmi_floor: float = 0.60
    anchor_blend: float = 0.10
    selection_graph_weight: float = 1.0
    selection_margin_weight: float = 0.25
    selection_stability_weight: float = 0.05
    selection_anchor_weight: float = 0.02
    feature_dropout: float = 0.05
    verbose_every: int = 10


# Backwards-compatible import name for older notebooks.  New papers and code
# should use BCPTConfig.
TangoConfig = BCPTConfig


# =============================================================================
# 8. Training one head
# =============================================================================


def train_one_head(
    Xcat_t: torch.Tensor,
    nbr_idx: torch.Tensor,
    nbr_w: torch.Tensor,
    graph_details: Optional[Dict[str, torch.Tensor]],
    init_labels: np.ndarray,
    k: int,
    cfg: BCPTConfig,
    device: torch.device,
    seed: int,
    verbose: bool = True,
) -> Tuple[ClusterHead, np.ndarray, float, Dict[str, float]]:
    set_seed(seed)
    N, D = Xcat_t.shape
    student = ClusterHead(D, cfg.hidden, cfg.proj_dim, k, cfg.dropout).to(device)

    with torch.no_grad():
        z_all = student.embed(Xcat_t)
        for c in range(k):
            mask = init_labels == c
            if mask.sum() > 0:
                student.prototypes[c].copy_(F.normalize(z_all[mask].mean(0), dim=0))

    teacher = ClusterHead(D, cfg.hidden, cfg.proj_dim, k, cfg.dropout).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)

    if graph_details is None:
        graph_details = {
            "view_idx": nbr_idx.unsqueeze(0),
            "view_w": nbr_w.unsqueeze(0),
            "agreement": torch.ones(N, device=device),
        }
    view_idx = graph_details["view_idx"]
    view_w = graph_details["view_w"]
    graph_agreement = graph_details["agreement"].clamp(0.0, 1.0)
    marg = CredalConsensusMarginal(
        k,
        device,
        floor_frac=cfg.marginal_floor_frac,
        eta=cfg.marginal_eta,
        bound_scale=cfg.credal_bound_scale,
        min_radius_frac=cfg.credal_min_radius_frac,
    )
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    m = min(cfg.nbr_per_anchor, nbr_idx.shape[1])
    running_cluster_size = torch.ones(k, device=device) / k
    uniform_pi = torch.ones(k, device=device) / k

    # Label-free initialization anchor.  Inverse-frequency weighting is based
    # only on KMeans cluster occupancy, not on semantic labels.  Local graph
    # agreement downweights uncertain boundary assignments while preserving
    # small coherent clusters.
    anchor_labels = torch.as_tensor(init_labels, dtype=torch.long, device=device)
    anchor_counts = torch.bincount(anchor_labels, minlength=k).float().clamp_min(1.0)
    anchor_class_w = (anchor_counts.mean() / anchor_counts).pow(cfg.anchor_tail_power)
    anchor_class_w = anchor_class_w / anchor_class_w.mean().clamp_min(1e-8)
    anchor_nbr = nbr_idx[:, :m]
    anchor_nbr_w = nbr_w[:, :m]
    anchor_nbr_w = anchor_nbr_w / anchor_nbr_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    anchor_local_agreement = (
        anchor_nbr_w
        * (anchor_labels[anchor_nbr] == anchor_labels.unsqueeze(1)).float()
    ).sum(dim=1)
    anchor_sample_w = anchor_class_w[anchor_labels] * (0.25 + 0.75 * anchor_local_agreement)
    anchor_sample_w = anchor_sample_w / anchor_sample_w.mean().clamp_min(1e-8)

    last_logs = {}
    for epoch in range(1, cfg.epochs + 1):
        student.train()
        with torch.no_grad():
            teacher.eval()
            teacher_probs_all = []
            for s in range(0, N, 4096):
                _, _, tp = teacher(Xcat_t[s:s + 4096])
                teacher_probs_all.append(tp)
            teacher_probs_all = torch.cat(teacher_probs_all, 0)
            credal_pi, credal_lower, credal_upper = marg.update(
                teacher_probs_all, view_idx, view_w, graph_agreement
            )

        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        epoch_pna = 0.0
        epoch_ot = 0.0
        epoch_floor = 0.0
        epoch_sep = 0.0
        epoch_anchor = 0.0
        epoch_accept = 0.0
        epoch_reject = 0.0
        batches = 0
        self_label_on = cfg.reliable_self_label and epoch >= int(cfg.selflabel_start * cfg.epochs)
        warm = min(1.0, epoch / max(1.0, cfg.balance_warmup * cfg.epochs))
        mass_progress = min(
            1.0,
            epoch / max(1.0, cfg.transported_mass_warmup * cfg.epochs),
        )
        rho = cfg.transported_mass_start + mass_progress * (
            cfg.transported_mass_end - cfg.transported_mass_start
        )
        epoch_progress = (epoch - 1) / max(1, cfg.epochs - 1)
        anchor_weight = cfg.anchor_weight_start + epoch_progress * (
            cfg.anchor_weight_end - cfg.anchor_weight_start
        )

        for s in range(0, N, cfg.batch_size):
            bids = perm[s:s + cfg.batch_size]
            xb = Xcat_t[bids]
            xb1 = F.dropout(xb, p=cfg.feature_dropout, training=True)
            xb2 = F.dropout(xb, p=cfg.feature_dropout, training=True)
            _, logit1, p1 = student(xb1)
            _, logit2, p2 = student(xb2)
            p_mean = 0.5 * (p1 + p2)
            conf = p_mean.max(dim=1).values

            pna_pi = credal_pi if cfg.pmi_mode in {"credal", "tail"} else uniform_pi
            pt_nbr, w_nbr = gather_neighbor_probs(teacher_probs_all, nbr_idx, nbr_w, bids, m)

            L_pna = 0.5 * (
                prior_normalized_neighbor_agreement(p1, pt_nbr, w_nbr, pna_pi, cfg.beta_pmi)
                + prior_normalized_neighbor_agreement(p2, pt_nbr, w_nbr, pna_pi, cfg.beta_pmi)
            )
            L_sep = prototype_separation(student)
            if cfg.pmi_mode in {"credal", "tail"}:
                collapse_floor = cfg.floor_frac_uniform * credal_lower
            else:
                collapse_floor = cfg.floor_frac_uniform * uniform_pi
            L_floor = anti_collapse_floor(p_mean.mean(dim=0), collapse_floor)
            loss = cfg.w_pmi * L_pna + cfg.w_sep * L_sep + cfg.w_floor * L_floor
            L_anchor = torch.tensor(0.0, device=device)
            if cfg.anchor_regularization and anchor_weight > 0:
                target = anchor_labels[bids]
                sample_w = anchor_sample_w[bids]
                ce1 = F.cross_entropy(logit1, target, reduction="none")
                ce2 = F.cross_entropy(logit2, target, reduction="none")
                L_anchor = (
                    0.5 * ((ce1 * sample_w).sum() + (ce2 * sample_w).sum())
                    / sample_w.sum().clamp_min(1e-8)
                )
                loss = loss + anchor_weight * L_anchor
            L_ot_value = torch.tensor(0.0, device=device)
            accept_mass = torch.ones_like(conf)
            reject_mass = torch.zeros_like(conf)

            js = 0.5 * (
                (p1 * (p1.clamp_min(1e-8).log() - p_mean.clamp_min(1e-8).log())).sum(dim=1)
                + (p2 * (p2.clamp_min(1e-8).log() - p_mean.clamp_min(1e-8).log())).sum(dim=1)
            )
            augmentation_agreement = (1.0 - js / math.log(max(k, 2))).clamp(0.0, 1.0)
            reliability = (
                conf.clamp_min(1e-6)
                * graph_agreement[bids].clamp_min(1e-6)
                * augmentation_agreement.clamp_min(1e-6)
            ).pow(1.0 / 3.0).detach()

            if cfg.ot_mode != "none":
                if cfg.ot_mode == "uniform":
                    ot_pi = uniform_pi
                    ot_lower = uniform_pi
                    ot_upper = uniform_pi
                    ot_rho = 1.0
                    ot_reliability = torch.ones_like(reliability)
                elif cfg.ot_mode == "uniform_partial":
                    ot_pi = uniform_pi
                    ot_lower = uniform_pi
                    ot_upper = uniform_pi
                    ot_rho = rho
                    ot_reliability = reliability
                elif cfg.ot_mode == "evidence_full":
                    ot_pi = uniform_pi
                    ot_lower = torch.zeros_like(uniform_pi)
                    ot_upper = torch.ones_like(uniform_pi)
                    ot_rho = 1.0
                    ot_reliability = reliability
                elif cfg.ot_mode == "bounded_full":
                    ot_pi = credal_pi
                    ot_lower = credal_lower
                    ot_upper = credal_upper
                    ot_rho = 1.0
                    ot_reliability = reliability
                elif cfg.ot_mode in {"bounded_partial", "tail"}:
                    ot_pi = credal_pi
                    ot_lower = credal_lower
                    ot_upper = credal_upper
                    ot_rho = rho
                    ot_reliability = reliability
                else:
                    raise ValueError(
                        "cfg.ot_mode must be one of: none, uniform, uniform_partial, "
                        "evidence_full, bounded_full, bounded_partial"
                    )

                if not cfg.reliability_weighting:
                    ot_reliability = torch.ones_like(ot_reliability)

                with torch.no_grad():
                    Q_mass, accept_mass, reject_mass, _ = bounded_partial_sinkhorn(
                        0.5 * (logit1 + logit2).detach(),
                        ot_pi,
                        ot_lower,
                        ot_upper,
                        reliability=ot_reliability,
                        transported_mass=ot_rho,
                        epsilon=cfg.ot_epsilon,
                        iters=cfg.ot_iters,
                        evidence_mix=cfg.evidence_mix,
                        reject_margin=cfg.reject_margin,
                    )
                logp = 0.5 * (F.log_softmax(logit1, dim=1) + F.log_softmax(logit2, dim=1))
                # Q_mass retains partial-transport mass.  Row normalization here
                # would erase rejection and reduce the method to full OT.
                L_ot_value = -(Q_mass * logp).sum(dim=1).mean() / max(ot_rho, 1e-6)
                loss = loss + warm * cfg.w_ot * L_ot_value

                if cfg.ot_mode in {"uniform_partial", "bounded_partial", "tail"} and self_label_on:
                    hard = p_mean.argmax(dim=1)
                    size_ratio = running_cluster_size / running_cluster_size.max().clamp_min(1e-8)
                    tau_c = cfg.conf_tau_base - cfg.conf_tau_tail * (1.0 - size_ratio)
                    conf_ok = conf >= tau_c[hard]
                    nbr_hard = pt_nbr.argmax(dim=2)
                    purity = (w_nbr * (nbr_hard == hard.unsqueeze(1)).float()).sum(dim=1)
                    reliable_label = (
                        conf_ok
                        & (purity >= cfg.purity_rho)
                        & (reliability >= 0.50)
                        & (accept_mass >= 0.50)
                    )
                    if reliable_label.sum() > 0:
                        loss = loss + 0.25 * (
                            F.cross_entropy(logit1[reliable_label], hard[reliable_label])
                            + F.cross_entropy(logit2[reliable_label], hard[reliable_label])
                        )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
            opt.step()
            ema_update(student, teacher, cfg.ema_momentum)

            with torch.no_grad():
                bsz = torch.bincount(p_mean.argmax(1), minlength=k).float() / p_mean.shape[0]
                running_cluster_size = 0.9 * running_cluster_size + 0.1 * bsz

            epoch_loss += float(loss.detach())
            epoch_pna += float(L_pna.detach())
            epoch_ot += float(L_ot_value.detach())
            epoch_floor += float(L_floor.detach())
            epoch_sep += float(L_sep.detach())
            epoch_anchor += float(L_anchor.detach())
            epoch_accept += float(accept_mass.mean().detach())
            epoch_reject += float(reject_mass.mean().detach())
            batches += 1

        sched.step()
        last_logs = {
            "loss": epoch_loss / max(1, batches),
            "pna": epoch_pna / max(1, batches),
            "ot": epoch_ot / max(1, batches),
            "floor": epoch_floor / max(1, batches),
            "sep": epoch_sep / max(1, batches),
            "anchor": epoch_anchor / max(1, batches),
            "anchor_weight": float(anchor_weight if cfg.anchor_regularization else 0.0),
            "pi_min": float(credal_pi.min().detach().cpu()),
            "pi_max": float(credal_pi.max().detach().cpu()),
            "bound_width": float((credal_upper - credal_lower).mean().detach().cpu()),
            "effective_n": float(marg.effective_n.detach().cpu()),
            "transported_mass": float(
                rho if cfg.ot_mode in {"uniform_partial", "bounded_partial", "tail"} else 1.0
            ),
            "accepted_mass": epoch_accept / max(1, batches),
            "rejected_mass": epoch_reject / max(1, batches),
        }
        if verbose and (epoch == 1 or epoch % cfg.verbose_every == 0 or epoch == cfg.epochs):
            print(
                f"  epoch {epoch:03d}/{cfg.epochs} "
                f"loss={last_logs['loss']:.4f} pna={last_logs['pna']:.4f} ot={last_logs['ot']:.4f} "
                f"anchor={last_logs['anchor']:.4f}@{last_logs['anchor_weight']:.3f} "
                f"pi[min={last_logs['pi_min']:.3f},max={last_logs['pi_max']:.3f}] "
                f"bounds={last_logs['bound_width']:.3f} accept={last_logs['accepted_mass']:.3f} "
                f"selflabel={'on' if self_label_on else 'off'}"
            )

    student.eval()
    with torch.no_grad():
        probs_all = []
        for s in range(0, N, 4096):
            _, _, pp = student(Xcat_t[s:s + 4096])
            probs_all.append(pp)
        probs_all = torch.cat(probs_all, 0)

        teacher.eval()
        tprobs = []
        for s in range(0, N, 4096):
            _, _, tp = teacher(Xcat_t[s:s + 4096])
            tprobs.append(tp)
        tprobs = torch.cat(tprobs, 0)

        marg.update(tprobs, view_idx, view_w, graph_agreement)
        final_pi, _, _ = marg.get()
        pna_pi = final_pi if cfg.pmi_mode in {"credal", "tail"} else uniform_pi
        sel_loss = 0.0
        for s in range(0, N, cfg.batch_size):
            bids = torch.arange(s, min(s + cfg.batch_size, N), device=device)
            pt_nbr, w_nbr = gather_neighbor_probs(tprobs, nbr_idx, nbr_w, bids, m)
            sel_loss += float(
                prior_normalized_neighbor_agreement(
                    probs_all[bids], pt_nbr, w_nbr, pna_pi, cfg.beta_pmi
                )
            )

    return student, probs_all.cpu().numpy(), float(sel_loss), last_logs


# =============================================================================
# 9. AAAI baselines and experiment drivers
# =============================================================================


@dataclass
class BaselineConfig:
    """Configuration for frozen-feature baseline adaptations.

    Important scientific note
    -------------------------
    SCAN-FS, P2OT-FS, SP2OT-FS, and PROTOCOL-FS use the published core
    mechanisms on exactly the same frozen DINOv2/BiomedCLIP features used by
    BCPT-Med. This gives a controlled feature-space comparison, but it is not
    claimed to be a byte-for-byte reproduction of each authors' full
    end-to-end training pipeline.
    """

    epochs: int = 60
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 512
    proj_dim: int = 128
    dropout: float = 0.1
    feature_dropout: float = 0.05
    verbose_every: int = 10

    # SCAN-style semantic clustering.
    scan_entropy_weight: float = 2.0
    scan_neighbors: int = 10

    # P2OT/SP2OT progressive partial OT.
    p2ot_rho_start: float = 0.20
    p2ot_rho_end: float = 0.99
    p2ot_epsilon: float = 0.10
    p2ot_gamma: float = 1.0
    p2ot_iters: int = 100
    p2ot_tol: float = 1e-6

    # SP2OT semantic regularization / MM.
    sp2ot_lambda: float = 0.50
    sp2ot_mm_iters: int = 3

    # PROTOCOL-style consensus, progressive OT, and rebalanced consistency.
    protocol_consensus_dim: int = 128
    protocol_temperature: float = 0.20
    protocol_view_weight: float = 0.50
    protocol_class_weight: float = 0.25
    protocol_logit_adjust_tau: float = 0.50
    protocol_prior_eta: float = 0.10
    protocol_conf_tau: float = 0.70


METHOD_LABELS = {
    "kmeans": "KMeans (same features)",
    "minibatch_kmeans": "MiniBatchKMeans (same features)",
    "gmm": "Gaussian mixture--diag (same features)",
    "spectral": "Spectral clustering--kNN (same features)",
    "scan": "SCAN-FS (adapted)",
    "p2ot": "P2OT-FS (adapted)",
    "sp2ot": "SP2OT-FS (adapted)",
    "protocol": "PROTOCOL-FS (adapted)",
    "bcpt_pmi": "BCPT-PNA (no OT)",
    "bcpt_uniformot": "BCPT-UniformOT (full mass)",
    "bcpt_bounded_full": "BCPT-BoundedOT (full mass)",
    "bcpt_partial_raw": "BCPT-PartialOT (without safeguards)",
    "bcpt_credal_only": "BCPT component: credal marginal only",
    "bcpt_rejection_only": "BCPT component: explicit rejection only",
    "bcpt_reliability_only": "BCPT component: reliability weighting only",
    "bcpt_anchor_retention_only": "BCPT component: anchor retention only",
    "bcpt_selection_gate_only": "BCPT component: selection gate only",
    "bcpt_fallback_only": "BCPT component: fallback only",
    "bcpt_anchor_blend_only": "BCPT component: anchor blend only",
    "bcpt_med": "BCPT-Med (proposed safeguarded partial OT)",
}

METHOD_ALIASES = {
    "kmeans": "kmeans",
    "km": "kmeans",
    "minibatch_kmeans": "minibatch_kmeans",
    "mbkmeans": "minibatch_kmeans",
    "mini_batch_kmeans": "minibatch_kmeans",
    "gmm": "gmm",
    "gaussian_mixture": "gmm",
    "gaussian-mixture": "gmm",
    "spectral": "spectral",
    "spectral_clustering": "spectral",
    "spectral-clustering": "spectral",
    "scan": "scan",
    "scan-fs": "scan",
    "p2ot": "p2ot",
    "p²ot": "p2ot",
    "ppot": "p2ot",
    "sp2ot": "sp2ot",
    "sp²ot": "sp2ot",
    "sppot": "sp2ot",
    "protocol": "protocol",
    "protocol-fs": "protocol",
    "bcpt_pmi": "bcpt_pmi",
    "bcpt-pmi": "bcpt_pmi",
    "bcpt_uniformot": "bcpt_uniformot",
    "bcpt-uniformot": "bcpt_uniformot",
    "uniformot": "bcpt_uniformot",
    "bcpt_bounded_full": "bcpt_bounded_full",
    "bcpt-bounded-full": "bcpt_bounded_full",
    "bcpt_partial_raw": "bcpt_partial_raw",
    "bcpt-partial-raw": "bcpt_partial_raw",
    "partial_raw": "bcpt_partial_raw",
    "bcpt_credal_only": "bcpt_credal_only",
    "credal_only": "bcpt_credal_only",
    "bcpt_rejection_only": "bcpt_rejection_only",
    "rejection_only": "bcpt_rejection_only",
    "bcpt_reliability_only": "bcpt_reliability_only",
    "reliability_only": "bcpt_reliability_only",
    "bcpt_anchor_retention_only": "bcpt_anchor_retention_only",
    "anchor_retention_only": "bcpt_anchor_retention_only",
    "bcpt_selection_gate_only": "bcpt_selection_gate_only",
    "selection_gate_only": "bcpt_selection_gate_only",
    "bcpt_fallback_only": "bcpt_fallback_only",
    "fallback_only": "bcpt_fallback_only",
    "bcpt_anchor_blend_only": "bcpt_anchor_blend_only",
    "anchor_blend_only": "bcpt_anchor_blend_only",
    "bcpt_med": "bcpt_med",
    "bcpt-med": "bcpt_med",
    "bcpt": "bcpt_med",
    # Legacy command aliases.  Results and paper text use the collision-free
    # BCPT-Med name.
    "tango_pmi": "bcpt_pmi",
    "tango-pmi": "bcpt_pmi",
    "tango_uniformot": "bcpt_uniformot",
    "tango-uniformot": "bcpt_uniformot",
    "tango_medpp": "bcpt_med",
    "tango-med++": "bcpt_med",
    "tango": "bcpt_med",
}

DEFAULT_ALL_METHODS = [
    "kmeans",
    "minibatch_kmeans",
    "gmm",
    "spectral",
    "scan",
    "p2ot",
    "sp2ot",
    "protocol",
    "bcpt_pmi",
    "bcpt_uniformot",
    "bcpt_bounded_full",
    "bcpt_partial_raw",
    "bcpt_med",
]

COMPONENT_ABLATIONS = [
    "bcpt_credal_only", "bcpt_rejection_only", "bcpt_reliability_only",
    "bcpt_anchor_retention_only", "bcpt_selection_gate_only",
    "bcpt_fallback_only", "bcpt_anchor_blend_only", "bcpt_med",
]

# The external comparison and internal ablation tables must not be conflated.
# Published methods below are controlled adaptations on the same frozen feature
# space; they are not presented as byte-for-byte reproductions of end-to-end
# pipelines trained with different backbones or augmentations.
CLASSICAL_BASELINES = ["kmeans", "minibatch_kmeans", "gmm", "spectral"]
PUBLISHED_FS_BASELINES = ["scan", "p2ot", "sp2ot", "protocol"]
BCPT_ABLATIONS = [
    "bcpt_pmi", "bcpt_uniformot", "bcpt_bounded_full",
    "bcpt_partial_raw",
] + COMPONENT_ABLATIONS
EXTERNAL_COMPARISON_METHODS = CLASSICAL_BASELINES + PUBLISHED_FS_BASELINES + ["bcpt_med"]


def normalize_methods(methods: Optional[List[str]]) -> List[str]:
    """Normalize CLI aliases and preserve the requested order without duplicates."""
    if methods is None:
        methods = DEFAULT_ALL_METHODS
    out: List[str] = []
    for name in methods:
        key = name.strip().lower()
        if key == "all":
            expanded = DEFAULT_ALL_METHODS
        elif key in {"components", "component_ablation", "component-ablations"}:
            expanded = COMPONENT_ABLATIONS
        else:
            if key not in METHOD_ALIASES:
                raise ValueError(
                    f"Unknown method '{name}'. Valid methods: "
                    + ", ".join(DEFAULT_ALL_METHODS)
                )
            expanded = [METHOD_ALIASES[key]]
        for item in expanded:
            if item not in out:
                out.append(item)
    return out


def progressive_rho(epoch: int, epochs: int, start: float, end: float) -> float:
    """Linear transported-mass schedule for progressive partial OT."""
    if epochs <= 1:
        return float(end)
    t = (epoch - 1) / float(epochs - 1)
    return float(start + t * (end - start))


@torch.no_grad()
def partial_ot_from_cost(
    cost: torch.Tensor,
    rho: float,
    epsilon: float = 0.10,
    gamma: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    prior: Optional[torch.Tensor] = None,
    scale_by_n: bool = True,
) -> torch.Tensor:
    """Progressive partial OT with a dummy reject column.

    This follows the core P2OT construction: a dummy column receives the
    untransported mass (1-rho), while real clusters receive rho. The real
    cluster scaling vector is regularized with gamma/(gamma+epsilon).

    Args:
        cost: [N, K] non-negative assignment cost.
        rho: transported mass in (0, 1].
        prior: optional non-uniform K-dimensional cluster prior.
        scale_by_n: return per-sample mass (average transported row mass ~= rho)
            when True; return a globally normalized transport plan when False.
    """
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2-D [N,K], got shape={tuple(cost.shape)}")
    n, k = cost.shape
    if n < 1 or k < 1:
        raise ValueError("partial OT requires N>=1 and K>=1")

    rho = float(np.clip(rho, 1e-6, 1.0 - 1e-8))
    eps = max(float(epsilon), 1e-6)
    gam = max(float(gamma), 1e-8)

    # Double precision greatly improves Sinkhorn stability for small epsilon.
    c = cost.detach().double()
    device = c.device
    dtype = c.dtype
    c = c - c.min(dim=1, keepdim=True).values
    c_ext = torch.cat([c, torch.zeros(n, 1, device=device, dtype=dtype)], dim=1)

    # Remove an irrelevant global offset before exponentiation.
    kernel = torch.exp(-c_ext / eps).clamp_min(1e-300)
    pa = torch.full((n, 1), 1.0 / n, device=device, dtype=dtype)

    if prior is None:
        p = torch.full((k,), 1.0 / k, device=device, dtype=dtype)
    else:
        p = prior.detach().to(device=device, dtype=dtype).reshape(-1)
        if p.numel() != k:
            raise ValueError(f"prior has {p.numel()} elements but K={k}")
        p = p.clamp_min(1e-12)
        p = p / p.sum().clamp_min(1e-12)

    pb_real = rho * p
    pb_dummy = torch.tensor([1.0 - rho], device=device, dtype=dtype)
    pb = torch.cat([pb_real, pb_dummy], dim=0).view(k + 1, 1)
    pb = pb / pb.sum().clamp_min(1e-12)

    b = torch.full((k + 1, 1), 1.0 / (k + 1), device=device, dtype=dtype)
    fi = gam / (gam + eps)

    for _ in range(max(1, int(max_iter))):
        a = pa / (kernel @ b).clamp_min(1e-300)
        b_new = pb / (kernel.t() @ a).clamp_min(1e-300)
        b_new[:-1] = b_new[:-1].clamp_min(1e-300).pow(fi)
        err = torch.max(torch.abs(b_new - b))
        b = b_new
        if float(err) <= tol:
            break

    plan = a * kernel * b.t()
    real = plan[:, :k]
    if scale_by_n:
        real = real * n
    return real.float()


@torch.no_grad()
def progressive_partial_ot(
    logits: torch.Tensor,
    rho: float,
    epsilon: float = 0.10,
    gamma: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    prior: Optional[torch.Tensor] = None,
    scale_by_n: bool = True,
) -> torch.Tensor:
    cost = -F.log_softmax(logits.detach(), dim=1).clamp_min(-80.0)
    return partial_ot_from_cost(
        cost,
        rho=rho,
        epsilon=epsilon,
        gamma=gamma,
        max_iter=max_iter,
        tol=tol,
        prior=prior,
        scale_by_n=scale_by_n,
    )


def partial_ot_ce(logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Cross entropy weighted by transported row mass."""
    if logits.shape != q.shape:
        raise ValueError(f"logits/q shape mismatch: {tuple(logits.shape)} vs {tuple(q.shape)}")
    mass = q.sum(dim=1)
    target = q / mass.unsqueeze(1).clamp_min(1e-12)
    per = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    return (mass * per).sum() / mass.sum().clamp_min(1e-12)


def initialize_head_from_kmeans(
    head: ClusterHead,
    Xcat_t: torch.Tensor,
    init_labels: np.ndarray,
    k: int,
):
    """Initialize normalized prototypes from KMeans assignments without labels."""
    with torch.no_grad():
        z_all = []
        for s in range(0, Xcat_t.shape[0], 4096):
            z_all.append(head.embed(Xcat_t[s:s + 4096]))
        z_all = torch.cat(z_all, dim=0)
        for c in range(k):
            idx_np = np.flatnonzero(init_labels == c)
            if idx_np.size:
                idx = torch.from_numpy(idx_np).to(Xcat_t.device)
                head.prototypes[c].copy_(F.normalize(z_all[idx].mean(0), dim=0))


@torch.no_grad()
def predict_head_probs(head: ClusterHead, X: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    head.eval()
    out = []
    for s in range(0, X.shape[0], chunk):
        _, _, p = head(X[s:s + chunk])
        out.append(p)
    return torch.cat(out, dim=0)


def _sample_graph_neighbors(
    nbr_idx: torch.Tensor,
    nbr_w: torch.Tensor,
    batch_ids: torch.Tensor,
    max_neighbors: int,
) -> torch.Tensor:
    m = min(max(1, int(max_neighbors)), nbr_idx.shape[1])
    idx = nbr_idx[batch_ids, :m]
    w = nbr_w[batch_ids, :m].clamp_min(0)
    row_sum = w.sum(dim=1, keepdim=True)
    bad = row_sum.squeeze(1) <= 0
    if bad.any():
        w = w.clone()
        w[bad] = 1.0
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    slot = torch.multinomial(w, num_samples=1).squeeze(1)
    return idx.gather(1, slot.unsqueeze(1)).squeeze(1)


def train_scan_fs(
    Xcat: np.ndarray,
    Xcat_t: torch.Tensor,
    nbr_idx: torch.Tensor,
    nbr_w: torch.Tensor,
    k: int,
    cfg: BaselineConfig,
    n_init: int,
    device: torch.device,
    seed: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """SCAN-style semantic clustering on the shared frozen feature space."""
    set_seed(seed)
    n, d = Xcat_t.shape
    init_labels, _ = spherical_kmeans_init(Xcat, k, n_init, seed)
    head = ClusterHead(d, cfg.hidden, cfg.proj_dim, k, cfg.dropout).to(device)
    initialize_head_from_kmeans(head, Xcat_t, init_labels, k)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.01)

    last = {}
    for epoch in range(1, cfg.epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        total = total_cons = total_ent = 0.0
        nb = 0
        for s in range(0, n, cfg.batch_size):
            bids = perm[s:s + cfg.batch_size]
            nids = _sample_graph_neighbors(nbr_idx, nbr_w, bids, cfg.scan_neighbors)
            xa = F.dropout(Xcat_t[bids], p=cfg.feature_dropout, training=True)
            xn = F.dropout(Xcat_t[nids], p=cfg.feature_dropout, training=True)
            _, _, pa = head(xa)
            _, _, pn = head(xn)

            sim = (pa * pn).sum(dim=1).clamp(1e-7, 1.0 - 1e-7)
            consistency = F.binary_cross_entropy(sim, torch.ones_like(sim))
            mean_p = pa.mean(dim=0).clamp_min(1e-8)
            entropy = -(mean_p * mean_p.log()).sum()
            loss = consistency - cfg.scan_entropy_weight * entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            opt.step()

            total += float(loss.detach())
            total_cons += float(consistency.detach())
            total_ent += float(entropy.detach())
            nb += 1
        sched.step()
        last = {
            "loss": total / max(1, nb),
            "consistency": total_cons / max(1, nb),
            "entropy": total_ent / max(1, nb),
        }
        if verbose and (epoch == 1 or epoch % cfg.verbose_every == 0 or epoch == cfg.epochs):
            print(
                f"  [SCAN-FS] epoch {epoch:03d}/{cfg.epochs} "
                f"loss={last['loss']:.4f} cons={last['consistency']:.4f} ent={last['entropy']:.4f}"
            )

    probs = predict_head_probs(head, Xcat_t).cpu().numpy().astype(np.float32)
    del head
    clean_memory()
    return probs, float(last.get("loss", 0.0)), last


def train_p2ot_fs(
    Xcat: np.ndarray,
    Xcat_t: torch.Tensor,
    k: int,
    cfg: BaselineConfig,
    n_init: int,
    device: torch.device,
    seed: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """P2OT-style progressive partial OT on the shared frozen feature space."""
    set_seed(seed)
    n, d = Xcat_t.shape
    init_labels, _ = spherical_kmeans_init(Xcat, k, n_init, seed)
    head = ClusterHead(d, cfg.hidden, cfg.proj_dim, k, cfg.dropout).to(device)
    initialize_head_from_kmeans(head, Xcat_t, init_labels, k)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.01)

    last = {}
    for epoch in range(1, cfg.epochs + 1):
        head.train()
        rho = progressive_rho(epoch, cfg.epochs, cfg.p2ot_rho_start, cfg.p2ot_rho_end)
        perm = torch.randperm(n, device=device)
        total = total_mass = 0.0
        nb = 0
        for s in range(0, n, cfg.batch_size):
            bids = perm[s:s + cfg.batch_size]
            xb = Xcat_t[bids]
            x1 = F.dropout(xb, p=cfg.feature_dropout, training=True)
            x2 = F.dropout(xb, p=cfg.feature_dropout, training=True)
            _, logit1, _ = head(x1)
            _, logit2, _ = head(x2)
            with torch.no_grad():
                q1 = progressive_partial_ot(
                    logit1, rho, cfg.p2ot_epsilon, cfg.p2ot_gamma,
                    cfg.p2ot_iters, cfg.p2ot_tol,
                )
                q2 = progressive_partial_ot(
                    logit2, rho, cfg.p2ot_epsilon, cfg.p2ot_gamma,
                    cfg.p2ot_iters, cfg.p2ot_tol,
                )
            loss = 0.5 * (partial_ot_ce(logit1, q2) + partial_ot_ce(logit2, q1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            opt.step()

            total += float(loss.detach())
            total_mass += 0.5 * float(q1.sum(1).mean() + q2.sum(1).mean())
            nb += 1
        sched.step()
        last = {
            "loss": total / max(1, nb),
            "rho": rho,
            "transported_mass": total_mass / max(1, nb),
        }
        if verbose and (epoch == 1 or epoch % cfg.verbose_every == 0 or epoch == cfg.epochs):
            print(
                f"  [P2OT-FS] epoch {epoch:03d}/{cfg.epochs} "
                f"loss={last['loss']:.4f} rho={rho:.3f} mass={last['transported_mass']:.3f}"
            )

    probs = predict_head_probs(head, Xcat_t).cpu().numpy().astype(np.float32)
    del head
    clean_memory()
    return probs, float(last.get("loss", 0.0)), last


@torch.no_grad()
def build_sparse_semantic_matrix(nbr_idx: torch.Tensor, nbr_w: torch.Tensor) -> torch.Tensor:
    """Build symmetric D^-1/2 A D^-1/2 sparse semantic graph for SP2OT-FS."""
    n, m = nbr_idx.shape
    device = nbr_idx.device
    src = torch.arange(n, device=device).view(-1, 1).expand(n, m).reshape(-1)
    dst = nbr_idx.reshape(-1)
    val = nbr_w.reshape(-1).float()
    mask = (val > 0) & (src != dst)
    src, dst, val = src[mask], dst[mask], val[mask]

    # Explicitly symmetrize before normalized adjacency construction.
    ii = torch.cat([src, dst], dim=0)
    jj = torch.cat([dst, src], dim=0)
    vv = torch.cat([val, val], dim=0)
    A = torch.sparse_coo_tensor(torch.stack([ii, jj]), vv, (n, n), device=device).coalesce()
    indices = A.indices()
    values = A.values()
    deg = torch.zeros(n, device=device, dtype=values.dtype)
    deg.scatter_add_(0, indices[0], values)
    inv_sqrt = deg.clamp_min(1e-12).rsqrt()
    values = values * inv_sqrt[indices[0]] * inv_sqrt[indices[1]]
    return torch.sparse_coo_tensor(indices, values, (n, n), device=device).coalesce()


@torch.no_grad()
def sp2ot_pseudolabels(
    logits: torch.Tensor,
    semantic_matrix: torch.Tensor,
    rho: float,
    cfg: BaselineConfig,
) -> torch.Tensor:
    """MM semantic-regularized partial OT pseudo-labels for SP2OT-FS."""
    base_cost = -F.log_softmax(logits.detach(), dim=1).clamp_min(-80.0)
    q = partial_ot_from_cost(
        base_cost,
        rho=rho,
        epsilon=cfg.p2ot_epsilon,
        gamma=cfg.p2ot_gamma,
        max_iter=cfg.p2ot_iters,
        tol=cfg.p2ot_tol,
        scale_by_n=False,
    )
    # For symmetric S, (S + S^T)Q = 2SQ.
    for _ in range(max(1, cfg.sp2ot_mm_iters)):
        sq = torch.sparse.mm(semantic_matrix, q)
        mm_cost = base_cost - 2.0 * cfg.sp2ot_lambda * sq
        q = partial_ot_from_cost(
            mm_cost,
            rho=rho,
            epsilon=cfg.p2ot_epsilon,
            gamma=cfg.p2ot_gamma,
            max_iter=cfg.p2ot_iters,
            tol=cfg.p2ot_tol,
            scale_by_n=False,
        )
    return q * logits.shape[0]


def train_sp2ot_fs(
    Xcat: np.ndarray,
    Xcat_t: torch.Tensor,
    semantic_matrix: torch.Tensor,
    k: int,
    cfg: BaselineConfig,
    n_init: int,
    device: torch.device,
    seed: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """SP2OT-style semantic MM + progressive partial OT on shared features."""
    set_seed(seed)
    n, d = Xcat_t.shape
    init_labels, _ = spherical_kmeans_init(Xcat, k, n_init, seed)
    head = ClusterHead(d, cfg.hidden, cfg.proj_dim, k, cfg.dropout).to(device)
    initialize_head_from_kmeans(head, Xcat_t, init_labels, k)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.01)

    last = {}
    for epoch in range(1, cfg.epochs + 1):
        rho = progressive_rho(epoch, cfg.epochs, cfg.p2ot_rho_start, cfg.p2ot_rho_end)
        head.eval()
        with torch.no_grad():
            logits_all = []
            for s in range(0, n, 4096):
                _, lg, _ = head(Xcat_t[s:s + 4096])
                logits_all.append(lg)
            logits_all = torch.cat(logits_all, dim=0)
            q_all = sp2ot_pseudolabels(logits_all, semantic_matrix, rho, cfg)

        head.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        nb = 0
        for s in range(0, n, cfg.batch_size):
            bids = perm[s:s + cfg.batch_size]
            xb = F.dropout(Xcat_t[bids], p=cfg.feature_dropout, training=True)
            _, logits, _ = head(xb)
            loss = partial_ot_ce(logits, q_all[bids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            opt.step()
            total += float(loss.detach())
            nb += 1
        sched.step()
        last = {
            "loss": total / max(1, nb),
            "rho": rho,
            "transported_mass": float(q_all.sum(1).mean().detach()),
        }
        if verbose and (epoch == 1 or epoch % cfg.verbose_every == 0 or epoch == cfg.epochs):
            print(
                f"  [SP2OT-FS] epoch {epoch:03d}/{cfg.epochs} "
                f"loss={last['loss']:.4f} rho={rho:.3f} mass={last['transported_mass']:.3f}"
            )

    probs = predict_head_probs(head, Xcat_t).cpu().numpy().astype(np.float32)
    del head
    clean_memory()
    return probs, float(last.get("loss", 0.0)), last


class ProtocolConsensusNet(nn.Module):
    """Compact consensus projector for a controlled PROTOCOL-FS adaptation."""

    def __init__(
        self,
        view_dims: List[int],
        hidden: int,
        consensus_dim: int,
        k: int,
        dropout: float,
    ):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, consensus_dim),
            )
            for d in view_dims
        ])
        self.prototypes = nn.Parameter(torch.randn(k, consensus_dim) * 0.02)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.1)))

    def project_views(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(views) != len(self.projectors):
            raise ValueError(f"Expected {len(self.projectors)} views, got {len(views)}")
        return [F.normalize(p(x), dim=-1) for p, x in zip(self.projectors, views)]

    def logits_from_z(self, z: torch.Tensor) -> torch.Tensor:
        protos = F.normalize(self.prototypes, dim=-1)
        tau = self.log_tau.exp().clamp(0.02, 1.0)
        return (z @ protos.t()) / tau

    def forward(self, views: List[torch.Tensor]):
        zs = self.project_views(views)
        zc = F.normalize(torch.stack(zs, dim=0).mean(dim=0), dim=-1)
        logits = self.logits_from_z(zc)
        return zs, zc, logits, F.softmax(logits, dim=-1)


def symmetric_infonce(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    if a.shape[0] <= 1:
        return torch.zeros((), device=a.device, dtype=a.dtype)
    sim = (a @ b.t()) / max(float(temperature), 1e-6)
    target = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.t(), target))


@torch.no_grad()
def predict_protocol_probs(
    model: ProtocolConsensusNet,
    views_t: List[torch.Tensor],
    chunk: int = 4096,
) -> torch.Tensor:
    model.eval()
    out = []
    n = views_t[0].shape[0]
    for s in range(0, n, chunk):
        batch_views = [v[s:s + chunk] for v in views_t]
        _, _, _, p = model(batch_views)
        out.append(p)
    return torch.cat(out, dim=0)


def train_protocol_fs(
    views_np: List[np.ndarray],
    views_t: List[torch.Tensor],
    Xcat: np.ndarray,
    k: int,
    cfg: BaselineConfig,
    n_init: int,
    device: torch.device,
    seed: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """PROTOCOL-style multi-view consensus adaptation on shared frozen features.

    The implementation uses the paper's central ingredients in controlled form:
    multi-view consensus projection, progressive partial OT, a dynamic class
    prior, logit adjustment, feature-level consensus contrast, and class-level
    consistency. It intentionally does not claim exact reproduction of the
    authors' custom transformer/end-to-end pipeline.
    """
    if len(views_t) < 2:
        raise ValueError("PROTOCOL-FS requires at least two feature views/encoders")
    set_seed(seed)
    n = views_t[0].shape[0]
    if any(v.shape[0] != n for v in views_t):
        raise ValueError("All PROTOCOL-FS views must contain the same number of samples")

    model = ProtocolConsensusNet(
        [v.shape[1] for v in views_t],
        hidden=cfg.hidden,
        consensus_dim=cfg.protocol_consensus_dim,
        k=k,
        dropout=cfg.dropout,
    ).to(device)

    # Label-free KMeans initialization in the same concatenated frozen space.
    init_labels, _ = spherical_kmeans_init(Xcat, k, n_init, seed)
    with torch.no_grad():
        z_all = []
        for s in range(0, n, 4096):
            _, zc, _, _ = model([v[s:s + 4096] for v in views_t])
            z_all.append(zc)
        z_all = torch.cat(z_all, dim=0)
        for c in range(k):
            idx_np = np.flatnonzero(init_labels == c)
            if idx_np.size:
                idx = torch.from_numpy(idx_np).to(device)
                model.prototypes[c].copy_(F.normalize(z_all[idx].mean(0), dim=0))

    prior = torch.ones(k, device=device) / k
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.01)

    last = {}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        rho = progressive_rho(epoch, cfg.epochs, cfg.p2ot_rho_start, cfg.p2ot_rho_end)
        perm = torch.randperm(n, device=device)
        total = total_ot = total_feat = total_cls = 0.0
        nb = 0
        for s in range(0, n, cfg.batch_size):
            bids = perm[s:s + cfg.batch_size]
            batch_views = [
                F.dropout(v[bids], p=cfg.feature_dropout, training=True)
                for v in views_t
            ]
            zs, zc, logits, probs = model(batch_views)

            with torch.no_grad():
                q = progressive_partial_ot(
                    logits,
                    rho=rho,
                    epsilon=cfg.p2ot_epsilon,
                    gamma=cfg.p2ot_gamma,
                    max_iter=cfg.p2ot_iters,
                    tol=cfg.p2ot_tol,
                    prior=prior,
                )

            adjusted_logits = logits + cfg.protocol_logit_adjust_tau * torch.log(prior.clamp_min(1e-8)).view(1, -1)
            l_ot = partial_ot_ce(adjusted_logits, q)

            l_feat = torch.stack([
                symmetric_infonce(zv, zc, cfg.protocol_temperature) for zv in zs
            ]).mean()

            p_cons = probs.detach().clamp_min(1e-8)
            cls_terms = []
            for zv in zs:
                pv = F.softmax(model.logits_from_z(zv), dim=1).clamp_min(1e-8)
                cls_terms.append(F.kl_div(pv.log(), p_cons, reduction="batchmean"))
            l_cls = torch.stack(cls_terms).mean()

            loss = l_ot + cfg.protocol_view_weight * l_feat + cfg.protocol_class_weight * l_cls
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            with torch.no_grad():
                conf = probs.max(dim=1).values
                mask = conf >= cfg.protocol_conf_tau
                use = probs[mask] if int(mask.sum()) >= k else probs
                hist = use.mean(dim=0).clamp_min(1e-5)
                hist = hist / hist.sum()
                prior = (1.0 - cfg.protocol_prior_eta) * prior + cfg.protocol_prior_eta * hist
                prior = prior.clamp_min(1e-5)
                prior = prior / prior.sum()

            total += float(loss.detach())
            total_ot += float(l_ot.detach())
            total_feat += float(l_feat.detach())
            total_cls += float(l_cls.detach())
            nb += 1
        sched.step()
        last = {
            "loss": total / max(1, nb),
            "ot": total_ot / max(1, nb),
            "feature": total_feat / max(1, nb),
            "class": total_cls / max(1, nb),
            "rho": rho,
            "prior_min": float(prior.min().detach()),
            "prior_max": float(prior.max().detach()),
        }
        if verbose and (epoch == 1 or epoch % cfg.verbose_every == 0 or epoch == cfg.epochs):
            print(
                f"  [PROTOCOL-FS] epoch {epoch:03d}/{cfg.epochs} "
                f"loss={last['loss']:.4f} ot={last['ot']:.4f} "
                f"feat={last['feature']:.4f} cls={last['class']:.4f} rho={rho:.3f}"
            )

    probs = predict_protocol_probs(model, views_t).cpu().numpy().astype(np.float32)
    del model
    clean_memory()
    return probs, float(last.get("loss", 0.0)), last


def prepare_shared(
    views_np: List[np.ndarray],
    y: np.ndarray,
    k: int,
    cfg: BCPTConfig,
    device: torch.device,
    mutual: Optional[bool] = None,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    Xcat = l2norm(np.concatenate(views_np, axis=1))
    n = Xcat.shape[0]
    if any(v.shape[0] != n for v in views_np):
        raise ValueError("All feature views must contain exactly the same number of samples")
    if len(y) != n:
        raise ValueError(f"Feature/label length mismatch: N={n}, labels={len(y)}")
    print(f"[data] N={n} views={[v.shape[1] for v in views_np]} concat_dim={Xcat.shape[1]} K={k}")
    print("[protocol] labels are withheld from training and used only after head selection")
    if mutual is None:
        mutual = len(views_np) > 1
    print(f"[graph] building consensus graph: k_nn={cfg.k_nn}, mutual={mutual}, chunk={cfg.knn_chunk}")
    views_t = [torch.from_numpy(v.astype(np.float32)).to(device) for v in views_np]
    Xcat_t = torch.from_numpy(Xcat.astype(np.float32)).to(device)
    nbr_idx, nbr_w, graph_details = build_consensus_graph(
        views_t,
        cfg.k_nn,
        mutual=mutual,
        knn_chunk=cfg.knn_chunk,
        return_details=True,
    )
    return Xcat, Xcat_t, nbr_idx, nbr_w, graph_details


def kmeans_multiseed(
    Xcat: np.ndarray,
    y: np.ndarray,
    k: int,
    seeds: List[int],
) -> Tuple[List[Dict[str, float]], Dict[str, Tuple[float, float]]]:
    rows = []
    for seed in seeds:
        km = KMeans(n_clusters=k, n_init=30, random_state=seed, max_iter=500, algorithm="lloyd").fit(Xcat)
        m = compute_metrics(y, km.labels_.astype(int), k)
        m["seed"] = seed
        rows.append(m)
        print(f" [KMeans seed={seed}] ACC={m['ACC']:.4f} NMI={m['NMI']:.4f} ARI={m['ARI']:.4f} BalACC={m['BalancedACC']:.4f}")
    return rows, summarize_rows(rows)


def minibatch_kmeans_multiseed(
    Xcat: np.ndarray,
    y: np.ndarray,
    k: int,
    seeds: List[int],
    batch_size: int = 1024,
) -> Tuple[List[Dict[str, float]], Dict[str, Tuple[float, float]]]:
    from sklearn.cluster import MiniBatchKMeans

    rows = []
    bs = min(max(k * 4, 32), max(32, int(batch_size)), len(Xcat))
    for seed in seeds:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=bs,
            n_init=10,
            max_iter=500,
            reassignment_ratio=0.01,
        ).fit(Xcat)
        pred = km.labels_.astype(int)
        m = compute_metrics(y, pred, k)
        m["seed"] = seed
        rows.append(m)
        print(f" [MiniBatchKMeans seed={seed}] ACC={m['ACC']:.4f} NMI={m['NMI']:.4f} ARI={m['ARI']:.4f}")
    return rows, summarize_rows(rows)


def gaussian_mixture_multiseed(
    Xcat: np.ndarray,
    y: np.ndarray,
    k: int,
    seeds: List[int],
) -> Tuple[List[Dict[str, float]], Dict[str, Tuple[float, float]]]:
    """Diagonal-covariance GMM on the shared frozen feature representation.

    A diagonal covariance is intentional: a full covariance matrix in the
    concatenated foundation-model space is both statistically ill-conditioned
    and unnecessarily expensive.  No labels are used by the estimator.
    """
    from sklearn.mixture import GaussianMixture

    rows = []
    for seed in seeds:
        model = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=1e-5,
            max_iter=200,
            n_init=3,
            init_params="kmeans",
            random_state=seed,
        )
        pred = model.fit_predict(Xcat).astype(int)
        m = compute_metrics(y, pred, k)
        m["seed"] = seed
        rows.append(m)
        print(
            f" [GMM-diag seed={seed}] ACC={m['ACC']:.4f} "
            f"NMI={m['NMI']:.4f} ARI={m['ARI']:.4f} BalACC={m['BalancedACC']:.4f}"
        )
    return rows, summarize_rows(rows)


def spectral_clustering_multiseed(
    Xcat: np.ndarray,
    y: np.ndarray,
    k: int,
    seeds: List[int],
    n_neighbors: int = 20,
) -> Tuple[List[Dict[str, float]], Dict[str, Tuple[float, float]]]:
    """Sparse kNN spectral clustering on the same normalized representation."""
    from sklearn.cluster import SpectralClustering

    rows = []
    nn = min(max(k + 1, int(n_neighbors)), max(1, len(Xcat) - 1))
    for seed in seeds:
        model = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=nn,
            assign_labels="kmeans",
            n_init=20,
            random_state=seed,
            eigen_solver="arpack",
        )
        pred = model.fit_predict(Xcat).astype(int)
        m = compute_metrics(y, pred, k)
        m["seed"] = seed
        rows.append(m)
        print(
            f" [Spectral-kNN seed={seed}] ACC={m['ACC']:.4f} "
            f"NMI={m['NMI']:.4f} ARI={m['ARI']:.4f} BalACC={m['BalancedACC']:.4f}"
        )
    return rows, summarize_rows(rows)


def assign_candidate_stability(candidates: List[Dict[str, Any]]) -> None:
    """Attach permutation-invariant cross-head stability to each candidate."""
    if len(candidates) == 1:
        candidates[0]["stability"] = 1.0
        return
    predictions = [c["probs"].argmax(axis=1) for c in candidates]
    for i, c in enumerate(candidates):
        agreement = [
            normalized_mutual_info_score(predictions[i], predictions[j])
            for j in range(len(candidates))
            if j != i
        ]
        c["stability"] = float(np.mean(agreement))


def select_best_head(candidates: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    """Generic label-free stability selector used by comparison baselines."""
    if not candidates:
        raise ValueError("No candidate heads were produced")
    assign_candidate_stability(candidates)
    max_active = max(c["active"] for c in candidates)
    survivors = [c for c in candidates if c["active"] == max_active]
    survivors.sort(key=lambda c: (-c["stability"], c["loss"]))
    return survivors[0]


def label_free_partition_quality(
    Xcat: np.ndarray,
    pred: np.ndarray,
    nbr_idx: np.ndarray,
    nbr_w: np.ndarray,
    k: int,
) -> Dict[str, float]:
    """Graph consistency and centroid margin without semantic labels.

    The score intentionally avoids a uniform-entropy reward, because the target
    medical data can be strongly imbalanced.  Empty clusters receive a hard
    penalty, while coherent small clusters are not penalized merely for size.
    """
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    if len(pred) != len(Xcat):
        raise ValueError("partition length does not match feature count")
    counts = np.bincount(pred, minlength=k)
    active = int((counts > 0).sum())

    w = np.asarray(nbr_w, dtype=np.float64)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    same = pred[np.asarray(nbr_idx, dtype=np.int64)] == pred[:, None]
    graph = float((w * same).sum(axis=1).mean())

    centers = np.zeros((k, Xcat.shape[1]), dtype=np.float32)
    for c in range(k):
        if counts[c] > 0:
            centers[c] = Xcat[pred == c].mean(axis=0)
    centers = l2norm(centers)
    sims = np.asarray(Xcat, dtype=np.float32) @ centers.T
    assigned = sims[np.arange(len(pred)), pred]
    masked = sims.copy()
    masked[np.arange(len(pred)), pred] = -np.inf
    second = masked.max(axis=1) if k > 1 else np.zeros(len(pred), dtype=np.float32)
    margin = float(np.mean(assigned - second))

    empty_penalty = float(k - active) / max(k, 1)
    return {
        "quality": graph + 0.25 * margin - empty_penalty,
        "graph": graph,
        "margin": margin,
        "active_exact": float(active),
    }


def align_anchor_probs(
    candidate_probs: np.ndarray,
    anchor_labels: np.ndarray,
    k: int,
    blend: float,
) -> np.ndarray:
    """Align an unsupervised anchor to candidate IDs and softly blend it."""
    blend = float(np.clip(blend, 0.0, 1.0))
    if blend <= 0:
        return candidate_probs
    pred = candidate_probs.argmax(axis=1)
    anchor = np.asarray(anchor_labels, dtype=np.int64)
    contingency = np.zeros((k, k), dtype=np.int64)
    for p, a in zip(pred, anchor):
        contingency[int(p), int(a)] += 1
    row, col = linear_sum_assignment(-contingency)
    anchor_to_candidate = {int(a): int(p) for p, a in zip(row, col)}
    mapped = np.asarray([anchor_to_candidate.get(int(a), int(a) % k) for a in anchor])
    anchor_probs = np.zeros_like(candidate_probs, dtype=np.float32)
    anchor_probs[np.arange(len(mapped)), mapped] = 1.0
    out = (1.0 - blend) * candidate_probs.astype(np.float32) + blend * anchor_probs
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-8)


def select_guarded_bcpt_head(
    candidates: List[Dict[str, Any]],
    Xcat: np.ndarray,
    nbr_idx: torch.Tensor,
    nbr_w: torch.Tensor,
    anchor_labels: np.ndarray,
    k: int,
    cfg: BCPTConfig,
) -> Dict[str, Any]:
    """Select a trained head only when it passes an unsupervised anchor guard."""
    if not candidates:
        raise ValueError("No BCPT candidates were produced")
    assign_candidate_stability(candidates)
    idx_np = nbr_idx.detach().cpu().numpy()
    w_np = nbr_w.detach().cpu().numpy()
    anchor_q = label_free_partition_quality(Xcat, anchor_labels, idx_np, w_np, k)

    accepted = []
    pass_flags = []
    for c in candidates:
        pred = c["probs"].argmax(axis=1)
        q = label_free_partition_quality(Xcat, pred, idx_np, w_np, k)
        anchor_nmi = float(normalized_mutual_info_score(anchor_labels, pred))
        c.update(q)
        c["anchor_nmi"] = anchor_nmi
        c["selection_score"] = (
            cfg.selection_graph_weight * q["graph"]
            + cfg.selection_margin_weight * q["margin"]
            + cfg.selection_stability_weight * c["stability"]
            + cfg.selection_anchor_weight * anchor_nmi
        )
        passed = bool(
            int(q["active_exact"]) == k
            and q["quality"] >= anchor_q["quality"] + cfg.anchor_guard_min_gain
            and anchor_nmi >= cfg.anchor_guard_nmi_floor
        )
        c["gate_passed"] = float(passed)
        pass_flags.append(float(passed))
        if passed:
            accepted.append(c)

    # With the selection gate disabled, all trained candidates remain eligible.
    pool = accepted if cfg.anchor_guard else list(candidates)
    if not cfg.anchor_guard and not cfg.anchor_fallback:
        pool = [select_best_head(candidates, k)]
    # Fallback-only ablation: evaluate the normally selected candidate against
    # the same label-free contract, but do not filter/rank the candidate family.
    if not cfg.anchor_guard and cfg.anchor_fallback:
        generic = select_best_head(candidates, k)
        pool = [generic] if generic.get("gate_passed", 0.0) >= 0.5 else []

    if not pool and cfg.anchor_fallback:
        anchor_probs = np.zeros((len(anchor_labels), k), dtype=np.float32)
        anchor_probs[np.arange(len(anchor_labels)), anchor_labels] = 1.0
        fallback = _candidate_from_probs(
            anchor_probs,
            loss=float("inf"),
            logs={"anchor_fallback": 1.0},
            head=-1,
            k=k,
        )
        fallback.update(anchor_q)
        fallback.update({
            "stability": 1.0,
            "anchor_nmi": 1.0,
            "selection_score": anchor_q["quality"],
            "anchor_fallback": True,
            "anchor_quality": anchor_q["quality"],
            "gate_pass_rate": float(np.mean(pass_flags)) if pass_flags else 0.0,
        })
        return fallback

    # Gate-only ablation must still return a trained head when every candidate
    # fails; this makes the effect of fallback separately measurable.
    if not pool:
        pool = list(candidates)
    pool.sort(key=lambda c: (-c["selection_score"], c["loss"]))
    best = pool[0]
    blended = align_anchor_probs(best["probs"], anchor_labels, k, cfg.anchor_blend)
    result = _candidate_from_probs(blended, best["loss"], best["logs"], best["head"], k)
    result.update({key: best[key] for key in (
        "stability", "quality", "graph", "margin", "active_exact",
        "anchor_nmi", "selection_score",
    )})
    result["anchor_fallback"] = False
    result["anchor_quality"] = anchor_q["quality"]
    result["gate_pass_rate"] = float(np.mean(pass_flags)) if pass_flags else 0.0
    return result


def _candidate_from_probs(probs: np.ndarray, loss: float, logs: Dict[str, float], head: int, k: int) -> Dict[str, Any]:
    if probs.ndim != 2 or probs.shape[1] != k:
        raise ValueError(f"Expected probabilities [N,{k}], got {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise FloatingPointError("Non-finite probabilities detected")
    pred = probs.argmax(axis=1)
    fr = np.bincount(pred, minlength=k).astype(np.float64) / max(len(pred), 1)
    active = int((fr > 0.0).sum())
    p = np.clip(fr, 1e-12, 1.0)
    ent = float(-(p * np.log(p)).sum())
    return {"active": active, "ent": ent, "loss": float(loss), "probs": probs, "logs": logs, "head": head}


def _prediction_bundle_path(out_dir: str, label: str, seed: int) -> str:
    safe_label = (
        label.replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
    )
    return os.path.join(out_dir, "predictions", f"{safe_label}_seed{seed}.npz")


def _save_prediction_bundle(out_dir: Optional[str], label: str, seed: int, pred: np.ndarray, probs: np.ndarray, y: np.ndarray):
    if not out_dir:
        return
    pred_dir = os.path.join(out_dir, "predictions")
    ensure_dir(pred_dir)
    np.savez_compressed(
        _prediction_bundle_path(out_dir, label, seed),
        pred=pred.astype(int),
        probs=probs.astype(np.float32),
        y=y.astype(int),
    )


def recover_rows_from_prediction_bundles(
    out_dir: str,
    label: str,
    seeds: List[int],
    k: int,
) -> Optional[List[Dict[str, float]]]:
    """Recover evaluation rows written before an interrupted suite checkpoint.

    Prediction bundles contain labels only for evaluation and are written after
    each completed seed.  Resource and BCPT-internal diagnostics cannot be
    reconstructed, so they remain NaN and are explicitly marked as recovered.
    """
    paths = [_prediction_bundle_path(out_dir, label, int(seed)) for seed in seeds]
    if not all(os.path.isfile(path) for path in paths):
        return None
    rows: List[Dict[str, float]] = []
    for seed, path in zip(seeds, paths):
        with np.load(path, allow_pickle=False) as bundle:
            pred = np.asarray(bundle["pred"], dtype=int).reshape(-1)
            target = np.asarray(bundle["y"], dtype=int).reshape(-1)
        if len(pred) != len(target) or len(pred) == 0:
            return None
        row = compute_metrics(target, pred, k)
        row.update({
            "seed": int(seed),
            "RuntimeSec": float("nan"),
            "PeakCudaMiB": float("nan"),
            "PeakProcessMiB": float("nan"),
            "RecoveredFromPrediction": 1.0,
        })
        rows.append(row)
    return rows


def train_eval_probs_method(
    trainer,
    y: np.ndarray,
    k: int,
    seeds: List[int],
    n_heads: int,
    label: str,
    out_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, Tuple[float, float]]]:
    """Evaluate any label-free trainer that returns (probs, objective, logs)."""
    rows = []
    for seed in seeds:
        resource_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        resource_mark = start_resource_measurement(resource_device)
        print(f"\n=== [{label}] SEED {seed} ===")
        candidates = []
        for h in range(max(1, n_heads)):
            run_seed = seed * 1000 + h
            print(f" [head {h + 1}/{max(1, n_heads)} | internal_seed={run_seed}]")
            probs, score, logs = trainer(run_seed)
            candidates.append(_candidate_from_probs(probs, score, logs, h, k))
        best = select_best_head(candidates, k)
        pred = best["probs"].argmax(axis=1)
        m = compute_metrics(y, pred, k)
        m.update(finish_resource_measurement(resource_mark, resource_device))
        m["seed"] = seed
        rows.append(m)
        print(
            f" [select] head={best['head']} active={best['active']}/{k} "
            f"stability={best.get('stability', 1.0):.3f} objective={best['loss']:.4f}"
        )
        print(f" [{label} eval] " + " | ".join(f"{kk}={vv:.4f}" for kk, vv in m.items() if kk != "seed"))
        _save_prediction_bundle(out_dir, label, seed, pred, best["probs"], y)
    return rows, summarize_rows(rows)


def train_eval_method(
    Xcat: np.ndarray,
    Xcat_t: torch.Tensor,
    nbr_idx: torch.Tensor,
    nbr_w: torch.Tensor,
    graph_details: Dict[str, torch.Tensor],
    y: np.ndarray,
    k: int,
    cfg: BCPTConfig,
    seeds: List[int],
    n_heads: int,
    n_init: int,
    device: torch.device,
    label: str,
    out_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, Tuple[float, float]]]:
    rows = []
    for seed in seeds:
        resource_mark = start_resource_measurement(device)
        print(f"\n=== [{label}] SEED {seed} ===")
        candidates = []
        anchor_labels = None
        needs_anchor = (
            cfg.anchor_regularization or cfg.anchor_guard or cfg.anchor_fallback
            or cfg.anchor_blend > 0.0
        )
        if needs_anchor:
            anchor_labels, _ = spherical_kmeans_init(Xcat, k, max(30, n_init), seed)
            if cfg.anchor_corruption_frac > 0:
                rng = np.random.default_rng(seed + 9049)
                n_bad = int(round(len(anchor_labels) * cfg.anchor_corruption_frac))
                bad = rng.choice(len(anchor_labels), size=min(n_bad, len(anchor_labels)), replace=False)
                offsets = rng.integers(1, max(k, 2), size=len(bad))
                anchor_labels[bad] = (anchor_labels[bad] + offsets) % k
                print(
                    f" [stress] corrupted {len(bad)}/{len(anchor_labels)} anchor assignments "
                    f"({cfg.anchor_corruption_frac:.1%})"
                )
            print(
                f" [anchor] label-free spherical KMeans, n_init={max(30, n_init)}, "
                f"active={np.unique(anchor_labels).size}/{k}"
            )
        for h in range(max(1, n_heads)):
            run_seed = seed * 1000 + h
            if cfg.anchor_regularization:
                assert anchor_labels is not None
                init_labels = anchor_labels.copy()
            else:
                init_labels, _ = spherical_kmeans_init(Xcat, k, n_init, run_seed)
            print(f" [head {h + 1}/{max(1, n_heads)} | internal_seed={run_seed}]")
            _, probs, sel_loss, logs = train_one_head(
                Xcat_t, nbr_idx, nbr_w, graph_details, init_labels, k, cfg, device,
                seed=run_seed,
                verbose=True,
            )
            candidates.append(_candidate_from_probs(probs, sel_loss, logs, h, k))

        selection_components = cfg.anchor_guard or cfg.anchor_fallback or cfg.anchor_blend > 0.0
        if selection_components:
            assert anchor_labels is not None
            best = select_guarded_bcpt_head(
                candidates, Xcat, nbr_idx, nbr_w, anchor_labels, k, cfg
            )
            for c in candidates:
                passed = int(
                    int(c.get("active_exact", 0)) == k
                    and c.get("quality", -float("inf"))
                    >= best.get("anchor_quality", float("inf")) + cfg.anchor_guard_min_gain
                    and c.get("anchor_nmi", 0.0) >= cfg.anchor_guard_nmi_floor
                )
                print(
                    f"  [candidate {c['head']}] quality={c.get('quality', float('nan')):.4f} "
                    f"graph={c.get('graph', float('nan')):.4f} "
                    f"margin={c.get('margin', float('nan')):.4f} "
                    f"anchor_nmi={c.get('anchor_nmi', float('nan')):.3f} "
                    f"stability={c.get('stability', float('nan')):.3f} pass={passed}"
                )
        else:
            best = select_best_head(candidates, k)
        best_pred = best["probs"].argmax(axis=1)
        m = compute_metrics(y, best_pred, k)
        logs = best.get("logs", {}) or {}
        m.update({
            "FallbackFrequency": float(bool(best.get("anchor_fallback", False))),
            "GatePassFrequency": float(best.get("gate_pass_rate", float("nan"))),
            "AcceptedMass": float(logs.get("accepted_mass", float("nan"))),
            "RejectedMass": float(logs.get("rejected_mass", float("nan"))),
            "CredalIntervalWidth": float(logs.get("bound_width", float("nan"))),
            "EffectiveSampleSize": float(logs.get("effective_n", float("nan"))),
        })
        m.update(finish_resource_measurement(resource_mark, device))
        m["seed"] = seed
        rows.append(m)
        if selection_components:
            print(
                f" [select] head={best['head']} active={best['active']}/{k} "
                f"fallback={int(best.get('anchor_fallback', False))} "
                f"quality={best.get('quality', float('nan')):.4f} "
                f"anchor_quality={best.get('anchor_quality', float('nan')):.4f} "
                f"anchor_nmi={best.get('anchor_nmi', float('nan')):.3f} "
                f"stability={best.get('stability', 1.0):.3f}"
            )
        else:
            print(
                f" [select] head={best['head']} active={best['active']}/{k} "
                f"stability={best.get('stability', 1.0):.3f} loss={best['loss']:.4f}"
            )
        print(f" [{label} eval] " + " | ".join(f"{kk}={vv:.4f}" for kk, vv in m.items() if kk != "seed"))
        _save_prediction_bundle(out_dir, label, seed, best_pred, best["probs"], y)
    return rows, summarize_rows(rows)


PAPER_METRICS = [
    "ACC", "NMI", "ARI", "MacroF1", "BalancedACC", "RareRecall",
    "WorstRecall", "HeadRecall", "MediumRecall", "TailRecall",
    "ActiveClusters", "CollapseFlag",
]
DIAGNOSTIC_METRICS = [
    "FallbackFrequency", "GatePassFrequency", "AcceptedMass", "RejectedMass",
    "CredalIntervalWidth", "EffectiveSampleSize", "RuntimeSec", "PeakCudaMiB",
    "PeakProcessMiB",
]
HIGHER_BETTER = {
    "ACC", "NMI", "ARI", "MacroF1", "BalancedACC", "RareRecall",
    "WorstRecall", "HeadRecall", "MediumRecall", "TailRecall", "ActiveClusters",
}


def best_per_metric(summaries: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict[str, str]:
    best = {}
    for met in PAPER_METRICS:
        vals = {name: s[met][0] for name, s in summaries.items() if met in s and np.isfinite(s[met][0])}
        if not vals:
            continue
        best[met] = (max if met in HIGHER_BETTER else min)(vals, key=vals.get)
    return best


def write_summary_csv(summaries: Dict[str, Dict[str, Tuple[float, float]]], path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["method"]
        for m in PAPER_METRICS:
            header += [f"{m}_mean", f"{m}_std"]
        w.writerow(header)
        for method, s in summaries.items():
            row = [method]
            for m in PAPER_METRICS:
                mean, std = s.get(m, (float("nan"), float("nan")))
                row += [f"{mean:.6f}", f"{std:.6f}"]
            w.writerow(row)


def write_diagnostic_summary_csv(
    summaries: Dict[str, Dict[str, Tuple[float, float]]], path: str
):
    """Write safeguard, transport, runtime, and memory diagnostics."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["method"]
        for metric in DIAGNOSTIC_METRICS:
            header += [f"{metric}_mean", f"{metric}_std"]
        w.writerow(header)
        for method, summary in summaries.items():
            row = [method]
            for metric in DIAGNOSTIC_METRICS:
                mean, std = summary.get(metric, (float("nan"), float("nan")))
                row += [f"{mean:.6f}", f"{std:.6f}"]
            w.writerow(row)


def write_per_class_recall_csv(
    all_rows: Dict[str, List[Dict[str, float]]], path: str
) -> None:
    """Write one row per method, seed, and semantic class (evaluation only)."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "seed", "class_id", "recall"])
        w.writeheader()
        for method, rows in all_rows.items():
            for row in rows:
                for key, value in row.items():
                    if key.startswith("RecallClass"):
                        w.writerow({
                            "method": method,
                            "seed": row.get("seed", -1),
                            "class_id": key.replace("RecallClass", ""),
                            "recall": value,
                        })


def write_seed_rows_csv(all_rows: Dict[str, List[Dict[str, float]]], path: str):
    ensure_dir(os.path.dirname(path) or ".")
    extra = sorted({
        key for rows in all_rows.values() for row in rows for key in row
        if key not in {"method", "seed"} and key not in PAPER_METRICS
    })
    fieldnames = ["method", "seed"] + PAPER_METRICS + extra
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for method, rows in all_rows.items():
            for r in rows:
                out = {"method": method, "seed": r.get("seed", -1)}
                for m in fieldnames[2:]:
                    out[m] = r.get(m, float("nan"))
                w.writerow(out)


def read_seed_rows_csv(path: str) -> Dict[str, List[Dict[str, float]]]:
    """Load previously completed seed rows for an explicit resume request."""
    if not os.path.isfile(path):
        return {}
    out: Dict[str, List[Dict[str, float]]] = {}
    with open(path, "r", newline="") as f:
        for raw in csv.DictReader(f):
            method = raw.get("method", "")
            if not method:
                continue
            row: Dict[str, float] = {"seed": int(float(raw.get("seed", -1)))}
            for metric in raw:
                if metric in {"method", "seed"}:
                    continue
                try:
                    row[metric] = float(raw.get(metric, "nan"))
                except (TypeError, ValueError):
                    row[metric] = float("nan")
            out.setdefault(method, []).append(row)
    return out


def completed_rows_for_seeds(
    existing: Dict[str, List[Dict[str, float]]],
    method: str,
    seeds: List[int],
) -> Optional[List[Dict[str, float]]]:
    """Return ordered cached rows only when every requested seed is present."""
    by_seed = {int(r["seed"]): r for r in existing.get(method, [])}
    if not all(int(s) in by_seed for s in seeds):
        return None
    return [by_seed[int(s)] for s in seeds]


def write_pairwise_significance(
    all_rows: Dict[str, List[Dict[str, float]]],
    proposed_label: str,
    path: str,
):
    """Paired two-sided Wilcoxon tests with per-metric Holm correction."""
    ensure_dir(os.path.dirname(path) or ".")
    metrics = [
        "ACC", "NMI", "ARI", "MacroF1", "BalancedACC",
        "WorstRecall", "TailRecall",
    ]
    prop = {int(r["seed"]): r for r in all_rows.get(proposed_label, [])}
    records = []
    if prop:
        for metric in metrics:
            metric_records = []
            for method, rows in all_rows.items():
                if method == proposed_label:
                    continue
                base = {int(r["seed"]): r for r in rows}
                shared = sorted(set(prop) & set(base))
                x = np.asarray([prop[s].get(metric, np.nan) for s in shared], dtype=float)
                y = np.asarray([base[s].get(metric, np.nan) for s in shared], dtype=float)
                keep = np.isfinite(x) & np.isfinite(y)
                x, y = x[keep], y[keep]
                if len(x) < 2:
                    continue
                delta = x - y
                if np.allclose(delta, 0.0):
                    statistic, pvalue = 0.0, 1.0
                else:
                    test = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
                    statistic, pvalue = float(test.statistic), float(test.pvalue)
                metric_records.append({
                    "proposed": proposed_label,
                    "baseline": method,
                    "metric": metric,
                    "n_pairs": len(x),
                    "mean_delta": float(delta.mean()),
                    "statistic": statistic,
                    "p_value": pvalue,
                })

            ordered = sorted(metric_records, key=lambda r: r["p_value"])
            running = 0.0
            mtests = len(ordered)
            for rank, rec in enumerate(ordered):
                adjusted = min(1.0, (mtests - rank) * rec["p_value"])
                running = max(running, adjusted)
                rec["p_holm"] = running
            records.extend(metric_records)

    fields = [
        "proposed", "baseline", "metric", "n_pairs", "mean_delta",
        "statistic", "p_value", "p_holm",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_latex_table(
    summaries: Dict[str, Dict[str, Tuple[float, float]]],
    dataset: str,
    encoders: str,
    path: str,
    caption: Optional[str] = None,
    table_label: Optional[str] = None,
    groups: Optional[List[Tuple[str, List[str]]]] = None,
):
    ensure_dir(os.path.dirname(path) or ".")
    best = best_per_metric(summaries)
    cols = ["ACC", "NMI", "ARI", "BalancedACC", "RareRecall", "WorstRecall", "ActiveClusters", "CollapseFlag"]
    headers = {
        "ACC": "ACC", "NMI": "NMI", "ARI": "ARI", "BalancedACC": "Bal. Acc",
        "RareRecall": "RareRec.", "WorstRecall": "WorstRec.", "ActiveClusters": "Act. K", "CollapseFlag": "Collapse",
    }

    def fmt(method: str, met: str):
        mean, std = summaries[method].get(met, (float("nan"), 0.0))
        if not np.isfinite(mean):
            return "--"
        if met == "ActiveClusters":
            body = f"{mean:.1f}"
            return f"\\textbf{{{body}}}" if best.get(met) == method else body
        body = f"{mean:.4f}{{\\scriptsize$\\pm${std:.3f}}}"
        if best.get(met) == method:
            body = f"\\textbf{{{mean:.4f}}}{{\\scriptsize$\\pm${std:.3f}}}"
        return body

    if caption is None:
        caption = (
            f"Label-free clustering on {dataset} using frozen features ({encoders}). "
            "Labels are used only for Hungarian-matched evaluation. Mean $\\pm$ std "
            "over seeds. Methods marked FS are controlled feature-space adaptations."
        )
    if table_label is None:
        table_label = f"tab:{dataset.lower()}_bcpt_med"

    lines = [
        "% Requires \\usepackage{booktabs} and \\usepackage{graphicx}",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{table_label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\toprule",
        "Method & " + " & ".join(headers[c] for c in cols) + " \\\\",
        "\\midrule",
    ]
    if groups:
        emitted = set()
        for group_name, group_methods in groups:
            present = [m for m in group_methods if m in summaries]
            if not present:
                continue
            if emitted:
                lines.append("\\addlinespace")
            lines.append(
                f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{group_name}}}}} \\\\"
            )
            for method in present:
                display = f"\\textbf{{{method}}}" if method == METHOD_LABELS["bcpt_med"] else method
                lines.append(display + " & " + " & ".join(fmt(method, c) for c in cols) + " \\\\")
                emitted.add(method)
        # Preserve any requested result that was not assigned to a group.
        for method in summaries.keys():
            if method not in emitted:
                lines.append(method + " & " + " & ".join(fmt(method, c) for c in cols) + " \\\\")
    else:
        for method in summaries.keys():
            lines.append(method + " & " + " & ".join(fmt(method, c) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def select_available_summaries(
    summaries: Dict[str, Dict[str, Tuple[float, float]]],
    method_keys: List[str],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Return an ordered label-keyed subset containing only methods that ran."""
    return {
        METHOD_LABELS[key]: summaries[METHOD_LABELS[key]]
        for key in method_keys
        if METHOD_LABELS[key] in summaries
    }


def print_console_table(summaries: Dict[str, Dict[str, Tuple[float, float]]]):
    if not summaries:
        print("[warn] no method summaries were produced")
        return
    methods = list(summaries.keys())
    colw = max(len(m) for m in methods) + 2
    width = 18
    header = "Method".ljust(colw) + "".join(f"{m:>{width}}" for m in PAPER_METRICS)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for method in methods:
        line = method.ljust(colw)
        for m in PAPER_METRICS:
            mean, std = summaries[method].get(m, (float("nan"), 0.0))
            if not np.isfinite(mean):
                cell = "--"
            elif m == "ActiveClusters":
                cell = f"{mean:.1f}"
            else:
                cell = f"{mean:.4f}±{std:.3f}"
            line += cell.rjust(width)
        print(line)
    print("=" * len(header))
    print("Higher is better except CollapseFlag; lower CollapseFlag is better.")


def make_method_configs(base_cfg: BCPTConfig) -> Dict[str, BCPTConfig]:
    return {
        "bcpt_pmi": replace(
            base_cfg, ot_mode="none", pmi_mode="credal", reliable_self_label=False,
            anchor_regularization=False, anchor_guard=False, anchor_fallback=False,
            anchor_blend=0.0,
        ),
        "bcpt_uniformot": replace(
            base_cfg, ot_mode="uniform", pmi_mode="uniform", reliable_self_label=False,
            anchor_regularization=False, anchor_guard=False, anchor_fallback=False,
            anchor_blend=0.0,
        ),
        "bcpt_bounded_full": replace(
            base_cfg, ot_mode="bounded_full", pmi_mode="credal", reliable_self_label=False,
            anchor_regularization=False, anchor_guard=False, anchor_fallback=False,
            anchor_blend=0.0,
        ),
        "bcpt_partial_raw": replace(
            base_cfg, ot_mode="bounded_partial", pmi_mode="credal", reliable_self_label=True,
            anchor_regularization=False, anchor_guard=False, anchor_fallback=False,
            anchor_blend=0.0,
        ),
        "bcpt_credal_only": replace(
            base_cfg, ot_mode="none", pmi_mode="credal", reliability_weighting=False,
            reliable_self_label=False, anchor_regularization=False,
            anchor_guard=False, anchor_fallback=False, anchor_blend=0.0,
        ),
        "bcpt_rejection_only": replace(
            base_cfg, ot_mode="uniform_partial", pmi_mode="uniform",
            reliability_weighting=False, reliable_self_label=False,
            anchor_regularization=False, anchor_guard=False,
            anchor_fallback=False, anchor_blend=0.0,
        ),
        "bcpt_reliability_only": replace(
            base_cfg, ot_mode="evidence_full", pmi_mode="uniform",
            reliability_weighting=True, reliable_self_label=False,
            anchor_regularization=False, anchor_guard=False,
            anchor_fallback=False, anchor_blend=0.0,
        ),
        "bcpt_anchor_retention_only": replace(
            base_cfg, ot_mode="none", pmi_mode="uniform",
            reliability_weighting=False, reliable_self_label=False,
            anchor_regularization=True, anchor_guard=False,
            anchor_fallback=False, anchor_blend=0.0,
        ),
        "bcpt_selection_gate_only": replace(
            base_cfg, ot_mode="none", pmi_mode="uniform",
            reliability_weighting=False, reliable_self_label=False,
            anchor_regularization=False, anchor_guard=True,
            anchor_fallback=False, anchor_blend=0.0,
        ),
        "bcpt_fallback_only": replace(
            base_cfg, ot_mode="none", pmi_mode="uniform",
            reliability_weighting=False, reliable_self_label=False,
            anchor_regularization=False, anchor_guard=False,
            anchor_fallback=True, anchor_blend=0.0,
        ),
        "bcpt_anchor_blend_only": replace(
            base_cfg, ot_mode="none", pmi_mode="uniform",
            reliability_weighting=False, reliable_self_label=False,
            anchor_regularization=False, anchor_guard=False,
            anchor_fallback=False, anchor_blend=0.10,
        ),
        "bcpt_med": replace(
            base_cfg, ot_mode="bounded_partial", pmi_mode="credal", reliable_self_label=True,
            reliability_weighting=True, anchor_regularization=True,
            anchor_guard=True, anchor_fallback=True,
        ),
    }


def run_one_dataset(
    dataset: str,
    encoders: List[str],
    split: str = "test",
    img_size: int = 224,
    feature_batch: int = 64,
    cfg: Optional[BCPTConfig] = None,
    baseline_cfg: Optional[BaselineConfig] = None,
    methods: Optional[List[str]] = None,
    seeds: List[int] = [0, 1, 2, 3, 4],
    n_heads: int = 5,
    baseline_heads: int = 5,
    n_init: int = 10,
    device: Optional[torch.device] = None,
    cache_dir: str = "./feat_cache",
    out_root: str = "./bcpt_med_outputs",
    mutual: Optional[bool] = None,
    amp: bool = True,
    imbalance_ratio: Optional[float] = None,
    imbalance_seed: int = 0,
    stress_test: Optional[str] = None,
    stress_strength: float = 0.35,
    reduced_sample_fraction: float = 0.50,
    wrong_k_delta: int = 1,
    reuse_existing: bool = False,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    if device is None:
        device = get_device()
    if cfg is None:
        cfg = BCPTConfig()
    if baseline_cfg is None:
        baseline_cfg = BaselineConfig(epochs=cfg.epochs, batch_size=cfg.batch_size)
    methods = normalize_methods(methods)
    if mutual is None:
        mutual = len(encoders) > 1

    enc_str = "+".join(encoders)
    dataset_tag = dataset
    if imbalance_ratio is not None:
        dataset_tag += f"_ir{imbalance_ratio:g}_split{imbalance_seed}"
    if stress_test not in {None, "none", "clean"}:
        dataset_tag += f"_stress-{stress_test}"
    out_dir = os.path.join(out_root, dataset_tag, enc_str.replace("/", "_"))
    ensure_dir(out_dir)
    print("\n" + "#" * 100)
    print(f"DATASET={dataset} | ENCODERS={enc_str} | SPLIT={split} | DEVICE={device} | MUTUAL={mutual}")
    print("METHODS=" + ", ".join(METHOD_LABELS[m] for m in methods))
    print("#" * 100)

    views, y, k = load_medmnist_views(dataset, split, encoders, img_size, feature_batch, device, cache_dir, amp=amp)
    benchmark_targets = None
    if imbalance_ratio is not None:
        lt_idx, benchmark_targets = make_long_tailed_indices(
            y, k, imbalance_ratio=imbalance_ratio, seed=imbalance_seed
        )
        views = [v[lt_idx] for v in views]
        y = y[lt_idx]
        print(
            f"[benchmark] controlled long tail IR={imbalance_ratio:g} seed={imbalance_seed} "
            f"N={len(y)} targets={benchmark_targets.tolist()}"
        )
    views, y, k, stress_meta = apply_feature_stress(
        views, y, k, stress_test, stress_strength, reduced_sample_fraction,
        wrong_k_delta, imbalance_seed,
    )
    if stress_meta.get("stress_test") != "clean":
        print(f"[stress] {json.dumps(stress_meta, sort_keys=True)}")
    if stress_test == "weak_anchor":
        cfg = replace(cfg, anchor_corruption_frac=max(0.50, stress_strength))
    Xcat, Xcat_t, nbr_idx, nbr_w, graph_details = prepare_shared(
        views, y, k, cfg, device, mutual=mutual
    )
    if stress_test == "corrupted_graph":
        nbr_idx, nbr_w, graph_details = corrupt_graph_tensors(
            nbr_idx, nbr_w, graph_details, stress_strength, imbalance_seed
        )
    views_t = [torch.from_numpy(v.astype(np.float32)).to(device) for v in views]

    summaries: Dict[str, Dict[str, Tuple[float, float]]] = {}
    all_rows: Dict[str, List[Dict[str, float]]] = {}
    seed_results_path = os.path.join(out_dir, "all_seed_results.csv")
    existing_rows = read_seed_rows_csv(seed_results_path) if reuse_existing else {}
    if reuse_existing:
        print(
            f"[resume] loaded {sum(len(v) for v in existing_rows.values())} seed rows "
            f"from {seed_results_path}"
        )
    semantic_matrix = None
    sp2ot_cached = completed_rows_for_seeds(
        existing_rows, METHOD_LABELS["sp2ot"], seeds
    )
    if "sp2ot" in methods and sp2ot_cached is None:
        print("[SP2OT-FS] building symmetric sparse semantic matrix")
        semantic_matrix = build_sparse_semantic_matrix(nbr_idx, nbr_w)

    bcpt_cfgs = make_method_configs(cfg)

    for method in methods:
        label = METHOD_LABELS[method]
        print(f"\n>>> Method: {label}")

        cached = completed_rows_for_seeds(existing_rows, label, seeds)
        if cached is not None:
            rows = cached
            summary = summarize_rows(rows)
            summaries[label] = summary
            all_rows[label] = rows
            print(f" [resume] reused {len(rows)} completed seeds; training skipped")
            continue

        if reuse_existing:
            recovered = recover_rows_from_prediction_bundles(
                out_dir, label, seeds, k
            )
            if recovered is not None:
                rows = recovered
                summary = summarize_rows(rows)
                summaries[label] = summary
                all_rows[label] = rows
                checkpoint_rows = dict(existing_rows)
                checkpoint_rows.update(all_rows)
                write_seed_rows_csv(checkpoint_rows, seed_results_path)
                print(
                    f" [resume] recovered {len(rows)} completed seeds from prediction "
                    "bundles; runtime/memory and method-internal diagnostics are unavailable"
                )
                continue

        method_resource_mark = start_resource_measurement(device)

        if method == "kmeans":
            rows, summary = kmeans_multiseed(Xcat, y, k, seeds)

        elif method == "minibatch_kmeans":
            rows, summary = minibatch_kmeans_multiseed(Xcat, y, k, seeds, baseline_cfg.batch_size)

        elif method == "gmm":
            rows, summary = gaussian_mixture_multiseed(Xcat, y, k, seeds)

        elif method == "spectral":
            rows, summary = spectral_clustering_multiseed(Xcat, y, k, seeds)

        elif method == "scan":
            rows, summary = train_eval_probs_method(
                trainer=lambda run_seed: train_scan_fs(
                    Xcat, Xcat_t, nbr_idx, nbr_w, k, baseline_cfg, n_init, device, run_seed, True,
                ),
                y=y, k=k, seeds=seeds, n_heads=baseline_heads, label=label, out_dir=out_dir,
            )

        elif method == "p2ot":
            rows, summary = train_eval_probs_method(
                trainer=lambda run_seed: train_p2ot_fs(
                    Xcat, Xcat_t, k, baseline_cfg, n_init, device, run_seed, True,
                ),
                y=y, k=k, seeds=seeds, n_heads=baseline_heads, label=label, out_dir=out_dir,
            )

        elif method == "sp2ot":
            assert semantic_matrix is not None
            rows, summary = train_eval_probs_method(
                trainer=lambda run_seed: train_sp2ot_fs(
                    Xcat, Xcat_t, semantic_matrix, k, baseline_cfg, n_init, device, run_seed, True,
                ),
                y=y, k=k, seeds=seeds, n_heads=baseline_heads, label=label, out_dir=out_dir,
            )

        elif method == "protocol":
            if len(views) < 2:
                print("[skip] PROTOCOL-FS requires at least two encoders/views; this single-view run is skipped.")
                continue
            rows, summary = train_eval_probs_method(
                trainer=lambda run_seed: train_protocol_fs(
                    views, views_t, Xcat, k, baseline_cfg, n_init, device, run_seed, True,
                ),
                y=y, k=k, seeds=seeds, n_heads=baseline_heads, label=label, out_dir=out_dir,
            )

        elif method in bcpt_cfgs:
            rows, summary = train_eval_method(
                Xcat, Xcat_t, nbr_idx, nbr_w, graph_details, y, k, bcpt_cfgs[method],
                seeds=seeds,
                n_heads=n_heads,
                n_init=n_init,
                device=device,
                label=label,
                out_dir=out_dir,
            )
        else:
            raise RuntimeError(f"Unhandled method: {method}")

        method_resources = finish_resource_measurement(method_resource_mark, device)
        for row in rows:
            row.setdefault("RuntimeSec", method_resources["RuntimeSec"] / max(len(rows), 1))
            row.setdefault("PeakCudaMiB", method_resources["PeakCudaMiB"])
            row.setdefault("PeakProcessMiB", method_resources["PeakProcessMiB"])

        # Recompute after attaching resource fields.
        summary = summarize_rows(rows)
        summaries[label] = summary
        all_rows[label] = rows

        # Commit each completed method immediately.  Large AAAI suites can run
        # for many hours; if a later ablation fails, --reuse-existing should
        # retain every method that already finished instead of waiting for the
        # entire dataset suite to complete.  Keep not-yet-visited cached rows
        # as well so a second interruption cannot discard prior work.
        checkpoint_rows = dict(existing_rows)
        checkpoint_rows.update(all_rows)
        write_seed_rows_csv(checkpoint_rows, seed_results_path)
        print(
            f" [checkpoint] saved {sum(len(v) for v in checkpoint_rows.values())} "
            f"seed rows to {seed_results_path}"
        )
        clean_memory()

    print_console_table(summaries)
    write_summary_csv(summaries, os.path.join(out_dir, "summary.csv"))
    write_diagnostic_summary_csv(
        summaries, os.path.join(out_dir, "diagnostic_summary.csv")
    )
    write_seed_rows_csv(all_rows, seed_results_path)
    write_per_class_recall_csv(
        all_rows, os.path.join(out_dir, "per_class_recall.csv")
    )
    write_pairwise_significance(
        all_rows,
        METHOD_LABELS["bcpt_med"],
        os.path.join(out_dir, "paired_significance.csv"),
    )

    # Save a full diagnostic table, a clean external comparison, and a
    # separate ablation table.  The latter two are the paper-facing outputs.
    write_latex_table(summaries, dataset, enc_str, os.path.join(out_dir, "table.tex"))

    comparison = select_available_summaries(summaries, EXTERNAL_COMPARISON_METHODS)
    write_summary_csv(comparison, os.path.join(out_dir, "external_comparison.csv"))
    comparison_groups = [
        ("Classical frozen-feature baselines", [METHOD_LABELS[m] for m in CLASSICAL_BASELINES]),
        ("Published-method feature-space adaptations", [METHOD_LABELS[m] for m in PUBLISHED_FS_BASELINES]),
        ("Proposed", [METHOD_LABELS["bcpt_med"]]),
    ]
    comparison_latex = write_latex_table(
        comparison,
        dataset,
        enc_str,
        os.path.join(out_dir, "external_comparison_table.tex"),
        caption=(
            f"Controlled label-free comparison on {dataset} with shared frozen features "
            f"({enc_str}). Labels are used only for Hungarian-matched evaluation. "
            "Values are mean $\\pm$ standard deviation over identical seeds. FS denotes "
            "a feature-space adaptation of the published method, not an official "
            "end-to-end reproduction."
        ),
        table_label=f"tab:{dataset.lower()}_external_comparison",
        groups=comparison_groups,
    )

    ablations = select_available_summaries(summaries, BCPT_ABLATIONS)
    write_summary_csv(ablations, os.path.join(out_dir, "ablation_summary.csv"))
    write_latex_table(
        ablations,
        dataset,
        enc_str,
        os.path.join(out_dir, "ablation_table.tex"),
        caption=(
            f"BCPT-Med ablation study on {dataset} with {enc_str} features. "
            "Values are mean $\\pm$ standard deviation over identical seeds."
        ),
        table_label=f"tab:{dataset.lower()}_bcpt_ablation",
        groups=[("BCPT components", [METHOD_LABELS[m] for m in BCPT_ABLATIONS])],
    )

    print("\n[External comparison LaTeX table]\n" + comparison_latex)
    print(f"[saved] {out_dir}")

    eval_counts = np.bincount(y, minlength=k)
    positive_counts = eval_counts[eval_counts > 0]
    meta = {
        "dataset": dataset,
        "encoders": encoders,
        "split": split,
        "img_size": img_size,
        "seeds": seeds,
        "n_heads": n_heads,
        "baseline_heads": baseline_heads,
        "n_init": n_init,
        "mutual": mutual,
        "controlled_imbalance_ratio": imbalance_ratio,
        "controlled_imbalance_seed": imbalance_seed if imbalance_ratio is not None else None,
        "controlled_imbalance_targets": (
            benchmark_targets.tolist() if benchmark_targets is not None else None
        ),
        "stress": stress_meta,
        "labels_used_for_benchmark_resampling": bool(imbalance_ratio is not None),
        "known_k_protocol": True,
        "labels_used_for_training_or_head_selection": False,
        "evaluation_class_counts": eval_counts.tolist(),
        "evaluation_imbalance_ratio": (
            float(positive_counts.max() / positive_counts.min()) if len(positive_counts) else float("nan")
        ),
        "equal_neural_head_budget": bool(n_heads == baseline_heads),
        "reuse_existing_requested": bool(reuse_existing),
        "methods": methods,
        "method_labels": {m: METHOD_LABELS[m] for m in methods},
        "cfg": cfg.__dict__,
        "baseline_cfg": baseline_cfg.__dict__,
        "proposed_method": {
            "name": "BCPT-Med",
            "transport": "reject-augmented partial OT with retained per-sample accepted mass",
            "target_marginal": "view-disagreement confidence interval plus bounded-simplex projection",
            "sample_reliability": "cross-foundation neighbour agreement x predictive confidence x augmentation agreement",
            "anchor_retention": "class-balanced graph-weighted KMeans pseudo-label regularization",
            "selection": (
                "label-free graph consistency plus centroid margin and cross-head stability; "
                "fallback to the initialization anchor when trained heads do not improve it"
            ),
        },
        "comparison_note": (
            "SCAN-FS, P2OT-FS, SP2OT-FS, and PROTOCOL-FS are controlled "
            "feature-space adaptations of published core mechanisms using the same frozen "
            "features as BCPT-Med; they are not claimed as exact full-pipeline reproductions."
        ),
        "diagnostic_definitions": {
            "FallbackFrequency": "fraction of outer seeds returning the KMeans anchor",
            "GatePassFrequency": "fraction of trained candidate heads passing every gate",
            "AcceptedMass": "mean final-epoch real-cluster transport row mass",
            "RejectedMass": "mean final-epoch reject-column row mass",
            "CredalIntervalWidth": "mean final-epoch upper-minus-lower marginal width",
            "EffectiveSampleSize": "final reliability-weighted marginal effective sample size",
            "RuntimeSec": "wall-clock seconds per outer seed (or method total divided by seeds)",
            "PeakCudaMiB": "peak allocated CUDA memory during the measured run",
            "PeakProcessMiB": "process maximum resident set size",
        },
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    del views, views_t, Xcat_t, nbr_idx, nbr_w, graph_details
    if semantic_matrix is not None:
        del semantic_matrix
    clean_memory()
    return summaries


def run_suite(
    datasets: List[str],
    encoders: List[str],
    split: str = "test",
    img_size: int = 224,
    feature_batch: int = 64,
    cfg: Optional[BCPTConfig] = None,
    baseline_cfg: Optional[BaselineConfig] = None,
    methods: Optional[List[str]] = None,
    seeds: List[int] = [0, 1, 2, 3, 4],
    n_heads: int = 5,
    baseline_heads: int = 5,
    n_init: int = 10,
    device: Optional[torch.device] = None,
    cache_dir: str = "./feat_cache",
    out_root: str = "./bcpt_med_outputs",
    mutual: Optional[bool] = None,
    amp: bool = True,
    imbalance_ratio: Optional[float] = None,
    imbalance_seed: int = 0,
    stress_test: Optional[str] = None,
    stress_strength: float = 0.35,
    reduced_sample_fraction: float = 0.50,
    wrong_k_delta: int = 1,
    reuse_existing: bool = False,
):
    if device is None:
        device = get_device()
    if cfg is None:
        cfg = BCPTConfig()
    if baseline_cfg is None:
        baseline_cfg = BaselineConfig(epochs=cfg.epochs, batch_size=cfg.batch_size)
    methods = normalize_methods(methods)
    ensure_dir(out_root)

    global_rows = []
    for dataset in datasets:
        summaries = run_one_dataset(
            dataset=dataset,
            encoders=encoders,
            split=split,
            img_size=img_size,
            feature_batch=feature_batch,
            cfg=cfg,
            baseline_cfg=baseline_cfg,
            methods=methods,
            seeds=seeds,
            n_heads=n_heads,
            baseline_heads=baseline_heads,
            n_init=n_init,
            device=device,
            cache_dir=cache_dir,
            out_root=out_root,
            mutual=mutual,
            amp=amp,
            imbalance_ratio=imbalance_ratio,
            imbalance_seed=imbalance_seed,
            stress_test=stress_test,
            stress_strength=stress_strength,
            reduced_sample_fraction=reduced_sample_fraction,
            wrong_k_delta=wrong_k_delta,
            reuse_existing=reuse_existing,
        )
        for method, s in summaries.items():
            row = {
                "dataset": dataset,
                "method": method,
                "encoders": "+".join(encoders),
                "imbalance_ratio": imbalance_ratio,
                "imbalance_seed": imbalance_seed if imbalance_ratio is not None else None,
                "stress_test": stress_test or "clean",
            }
            for m in PAPER_METRICS:
                mean, std = s.get(m, (float("nan"), float("nan")))
                row[f"{m}_mean"] = mean
                row[f"{m}_std"] = std
            global_rows.append(row)

    global_name = "global_summary.csv"
    if imbalance_ratio is not None:
        global_name = f"global_summary_ir{imbalance_ratio:g}_split{imbalance_seed}.csv"
    if stress_test not in {None, "none", "clean"}:
        global_name = global_name.replace(".csv", f"_stress-{stress_test}.csv")
    global_path = os.path.join(out_root, global_name)
    with open(global_path, "w", newline="") as f:
        fieldnames = [
            "dataset", "method", "encoders", "imbalance_ratio", "imbalance_seed",
            "stress_test"
        ] + [
            x for m in PAPER_METRICS for x in (f"{m}_mean", f"{m}_std")
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in global_rows:
            w.writerow(r)
    print(f"[saved global] {global_path}")
    return global_rows


def zip_outputs(out_root: str = "./bcpt_med_outputs", zip_name: str = "bcpt_med_outputs"):
    base = shutil.make_archive(zip_name, "zip", out_root)
    print(f"[zip] {base}")
    return base


NATIVE_BASELINE_SPECS = {
    "p2ot": {
        "repo": "https://github.com/rhfeiyang/PPOT.git",
        "directory": "PPOT",
        "supported": ["cifar100", "imagenet-r", "inature100", "inature500", "inature1000"],
    },
    "sp2ot": {
        "repo": "https://github.com/rhfeiyang/SPPOT.git",
        "directory": "SPPOT",
        "supported": ["cifar100", "imagenet-r", "inature100", "inature500", "inature1000"],
    },
    "protocol": {
        "repo": "https://github.com/Scarlett125/PROTOCOL.git",
        "directory": "PROTOCOL",
        "supported": ["official_multiview_data"],
    },
}


def setup_native_baselines(native_root: Path) -> Dict[str, Any]:
    """Clone official repositories once and write a compatibility manifest."""
    native_root.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {}
    for name, spec in NATIVE_BASELINE_SPECS.items():
        target = native_root / spec["directory"]
        status = "present"
        if not target.is_dir():
            print(f"[native setup] cloning {name}: {spec['repo']} -> {target}")
            proc = subprocess.run(
                ["git", "clone", "--depth", "1", spec["repo"], str(target)],
                text=True, capture_output=True,
            )
            status = "cloned" if proc.returncode == 0 else "clone_failed"
            if proc.returncode != 0:
                print(f"[native setup] {name} failed: {proc.stderr.strip()}")
        manifest[name] = {
            **spec,
            "path": str(target),
            "status": status,
            "medmnist_directly_supported": False,
            "comparison_scope": (
                "Official/native pipeline on its published datasets only; do not merge "
                "with the controlled MedMNIST frozen-feature table."
            ),
        }
    manifest_path = native_root / "native_baseline_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[native setup] manifest: {manifest_path}")
    return manifest


def native_command(
    method: str,
    dataset: str,
    imbalance_ratio: float,
    output_dir: str,
    python_executable: str,
) -> List[str]:
    """Build commands from the official repository READMEs."""
    method = method.lower()
    ds = dataset.lower()
    if method not in NATIVE_BASELINE_SPECS:
        raise ValueError(f"Unknown native method: {method}")
    if ds not in NATIVE_BASELINE_SPECS[method]["supported"]:
        raise ValueError(
            f"Official {method} does not directly support dataset '{dataset}'. "
            f"Supported: {NATIVE_BASELINE_SPECS[method]['supported']}"
        )
    if method == "protocol":
        return [python_executable, "train.py"]

    if ds == "cifar100":
        db, classes = "cifar_im", 100
    elif ds == "imagenet-r":
        db, classes = "imagenet-r_im", 200
    else:
        db = "iNature_im"
        classes = int(ds.replace("inature", ""))
    command = [
        python_executable, "train.py", "--train_db_name", db,
        "--val_db_name", db, "--num_classes", str(classes),
        "--output_dir", output_dir,
    ]
    if method == "p2ot":
        if ds == "cifar100":
            command += ["--imbalance_ratio", str(1.0 / float(imbalance_ratio)), "--num_heads", "2"]
        else:
            command += ["--num_heads", "1"]
    else:
        command += ["--mm_factor", "1000", "--topk_similarity", "20", "--bank_use"]
        if ds == "cifar100":
            command += ["--batch_size", "512"]
    return command


def run_native_baseline_adapters(
    methods: List[str],
    native_root: Path,
    dataset: str,
    imbalance_ratio: float,
    out_root: str,
    execute: bool = False,
    python_executable: str = sys.executable,
) -> List[Dict[str, Any]]:
    """Generate or execute official commands and record explicit status rows."""
    ensure_dir(out_root)
    records: List[Dict[str, Any]] = []
    for method in methods:
        spec = NATIVE_BASELINE_SPECS[method]
        repo = native_root / spec["directory"]
        record: Dict[str, Any] = {
            "method": method, "dataset": dataset,
            "imbalance_ratio": imbalance_ratio, "repo": spec["repo"],
            "repository_path": str(repo), "executed": False,
        }
        try:
            command = native_command(
                method, dataset, imbalance_ratio,
                str(Path(out_root).resolve() / f"native_{method}_{dataset}"),
                python_executable,
            )
            record["command"] = shlex.join(command)
            if not repo.is_dir():
                record["status"] = "repository_missing_run_setup_first"
            elif not execute:
                record["status"] = "command_generated_not_executed"
            else:
                log_path = Path(out_root) / f"native_{method}_{dataset}.log"
                with log_path.open("w", encoding="utf-8") as log:
                    proc = subprocess.run(command, cwd=str(repo), stdout=log, stderr=subprocess.STDOUT)
                record.update({
                    "executed": True,
                    "return_code": int(proc.returncode),
                    "status": "completed" if proc.returncode == 0 else "failed",
                    "log": str(log_path),
                })
        except Exception as exc:
            record.update({"status": "unsupported", "reason": str(exc)})
        records.append(record)
        print(f"[native] {method}: {record['status']}")
    with open(os.path.join(out_root, "native_status.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return records


def run_aaai_final_suite(
    datasets: List[str],
    encoders: List[str],
    cfg: BCPTConfig,
    baseline_cfg: BaselineConfig,
    seeds: List[int],
    n_heads: int,
    baseline_heads: int,
    n_init: int,
    device: torch.device,
    cache_dir: str,
    out_root: str,
    feature_batch: int,
    img_size: int,
    reuse_existing: bool,
    imbalance_ratios: List[float],
    stress_tests: List[str],
    stress_strength: float,
    reduced_sample_fraction: float,
    wrong_k_delta: int,
) -> None:
    """Run clean, component, imbalance, and safeguard evidence as separate suites."""
    clean_methods = list(dict.fromkeys(
        CLASSICAL_BASELINES + PUBLISHED_FS_BASELINES + COMPONENT_ABLATIONS
    ))
    run_suite(
        datasets=datasets, encoders=encoders, methods=clean_methods, cfg=cfg,
        baseline_cfg=baseline_cfg, seeds=seeds, n_heads=n_heads,
        baseline_heads=baseline_heads, n_init=n_init, device=device,
        cache_dir=cache_dir, out_root=os.path.join(out_root, "clean_and_components"),
        feature_batch=feature_batch, img_size=img_size, reuse_existing=reuse_existing,
    )
    imbalance_methods = ["p2ot", "sp2ot", "protocol", "bcpt_med"]
    for ratio in imbalance_ratios:
        run_suite(
            datasets=datasets, encoders=encoders, methods=imbalance_methods, cfg=cfg,
            baseline_cfg=baseline_cfg, seeds=seeds, n_heads=n_heads,
            baseline_heads=baseline_heads, n_init=n_init, device=device,
            cache_dir=cache_dir, out_root=os.path.join(out_root, "controlled_imbalance"),
            feature_batch=feature_batch, img_size=img_size,
            imbalance_ratio=ratio, imbalance_seed=0, reuse_existing=reuse_existing,
        )
    for stress in stress_tests:
        run_suite(
            datasets=datasets, encoders=encoders,
            methods=["kmeans", "bcpt_partial_raw", "bcpt_med"], cfg=cfg,
            baseline_cfg=baseline_cfg, seeds=seeds, n_heads=n_heads,
            baseline_heads=baseline_heads, n_init=n_init, device=device,
            cache_dir=cache_dir, out_root=os.path.join(out_root, "safeguard_stress"),
            feature_batch=feature_batch, img_size=img_size,
            stress_test=stress, stress_strength=stress_strength,
            reduced_sample_fraction=reduced_sample_fraction,
            wrong_k_delta=wrong_k_delta, reuse_existing=reuse_existing,
        )


def run_algorithmic_self_test():
    """Check bounded-simplex feasibility and partial-transport mass conservation."""
    device = torch.device("cpu")
    set_seed(7)
    B, K = 37, 4
    logits = torch.randn(B, K, device=device)
    prior = torch.tensor([0.48, 0.27, 0.17, 0.08], device=device)
    lower = torch.tensor([0.38, 0.18, 0.10, 0.03], device=device)
    upper = torch.tensor([0.58, 0.36, 0.25, 0.16], device=device)
    reliability = torch.linspace(0.05, 0.98, B, device=device)
    rho = 0.73

    projected = project_box_simplex(prior, lower, upper)
    assert abs(float(projected.sum()) - 1.0) < 1e-5
    assert bool(torch.all(projected >= lower - 1e-6))
    assert bool(torch.all(projected <= upper + 1e-6))

    # Regression: an exact uniform constraint represented in float32 can sum
    # slightly above one.  This is the configuration used by the explicit-
    # rejection-only ablation and must remain feasible.
    uniform11 = torch.ones(11, device=device) / 11
    projected11 = project_box_simplex(uniform11, uniform11, uniform11)
    assert abs(float(projected11.sum()) - 1.0) < 1e-6
    assert bool(torch.allclose(projected11, uniform11, atol=2e-7, rtol=2e-7))

    q, accept, reject, target = bounded_partial_sinkhorn(
        logits,
        prior,
        lower,
        upper,
        reliability,
        transported_mass=rho,
        epsilon=0.10,
        iters=200,
    )
    assert q.shape == (B, K)
    assert bool(torch.allclose(accept + reject, torch.ones(B), atol=2e-4, rtol=2e-4))
    assert abs(float(accept.mean()) - rho) < 2e-4
    assert abs(float(reject.mean()) - (1.0 - rho)) < 2e-4
    assert bool(torch.allclose(q.mean(dim=0), rho * target, atol=2e-4, rtol=2e-4))
    assert bool(torch.all(target >= lower - 1e-6))
    assert bool(torch.all(target <= upper + 1e-6))
    print("[self-test] bounded simplex: PASS")
    print("[self-test] float32 fixed-uniform marginal regression: PASS")
    print("[self-test] partial-transport row/column mass conservation: PASS")


# =============================================================================
# 10. Ready-to-run notebook / paper configurations
# =============================================================================


QUICK_TEST = {
    "datasets": ["pathmnist"],
    "encoders": ["dinov2_vitb14"],
    "methods": ["kmeans", "scan", "p2ot", "bcpt_med"],
    "seeds": [0],
    "n_heads": 1,
    "baseline_heads": 1,
    "n_init": 3,
    "cfg": BCPTConfig(epochs=5, batch_size=512, k_nn=20, nbr_per_anchor=10, knn_chunk=1024),
    "baseline_cfg": BaselineConfig(epochs=5, batch_size=512, p2ot_iters=50),
}

PAPER_RUN_SINGLE_BACKBONE = {
    "datasets": ["pathmnist", "organcmnist", "organamnist", "tissuemnist"],
    "encoders": ["dinov2_vitb14"],
    "methods": [
        "kmeans", "minibatch_kmeans", "scan", "p2ot", "sp2ot",
        "bcpt_pmi", "bcpt_uniformot", "bcpt_bounded_full",
        "bcpt_partial_raw", "bcpt_med",
    ],
    "seeds": [0, 1, 2, 3, 4],
    "n_heads": 5,
    "baseline_heads": 5,
    "n_init": 10,
    "cfg": BCPTConfig(epochs=60, batch_size=512, k_nn=20, nbr_per_anchor=10, knn_chunk=1024),
    "baseline_cfg": BaselineConfig(epochs=60, batch_size=512),
}

PAPER_RUN_CROSS_FOUNDATION = {
    "datasets": ["pathmnist", "bloodmnist", "organcmnist", "organamnist", "tissuemnist"],
    "encoders": ["dinov2_vitb14", "biomedclip"],
    "methods": DEFAULT_ALL_METHODS,
    "seeds": [0, 1, 2, 3, 4],
    "n_heads": 5,
    "baseline_heads": 5,
    "n_init": 10,
    "cfg": BCPTConfig(epochs=60, batch_size=512, k_nn=20, nbr_per_anchor=10, knn_chunk=1024),
    "baseline_cfg": BaselineConfig(epochs=60, batch_size=512),
}


def run_quick_test(methods: Optional[List[str]] = None):
    configure_offline_paths(verbose=True)
    device = get_device()
    print("[device]", device)
    return run_suite(
        datasets=QUICK_TEST["datasets"],
        encoders=QUICK_TEST["encoders"],
        methods=methods or QUICK_TEST["methods"],
        cfg=QUICK_TEST["cfg"],
        baseline_cfg=QUICK_TEST["baseline_cfg"],
        seeds=QUICK_TEST["seeds"],
        n_heads=QUICK_TEST["n_heads"],
        baseline_heads=QUICK_TEST["baseline_heads"],
        n_init=QUICK_TEST["n_init"],
        device=device,
        cache_dir="./feat_cache",
        out_root="./bcpt_med_outputs_quick",
        mutual=None,
    )


def run_paper_single_backbone(
    methods: Optional[List[str]] = None,
    baseline_epochs: Optional[int] = None,
):
    configure_offline_paths(verbose=True)
    device = get_device()
    print("[device]", device)
    bcfg = PAPER_RUN_SINGLE_BACKBONE["baseline_cfg"]
    if baseline_epochs is not None:
        bcfg = replace(bcfg, epochs=baseline_epochs)
    return run_suite(
        datasets=PAPER_RUN_SINGLE_BACKBONE["datasets"],
        encoders=PAPER_RUN_SINGLE_BACKBONE["encoders"],
        methods=methods or PAPER_RUN_SINGLE_BACKBONE["methods"],
        cfg=PAPER_RUN_SINGLE_BACKBONE["cfg"],
        baseline_cfg=bcfg,
        seeds=PAPER_RUN_SINGLE_BACKBONE["seeds"],
        n_heads=PAPER_RUN_SINGLE_BACKBONE["n_heads"],
        baseline_heads=PAPER_RUN_SINGLE_BACKBONE["baseline_heads"],
        n_init=PAPER_RUN_SINGLE_BACKBONE["n_init"],
        device=device,
        cache_dir="./feat_cache",
        out_root="./bcpt_med_outputs_single",
        mutual=None,
    )


def run_paper_cross_foundation(
    methods: Optional[List[str]] = None,
    baseline_epochs: Optional[int] = None,
):
    configure_offline_paths(verbose=True)
    device = get_device()
    print("[device]", device)
    bcfg = PAPER_RUN_CROSS_FOUNDATION["baseline_cfg"]
    if baseline_epochs is not None:
        bcfg = replace(bcfg, epochs=baseline_epochs)
    return run_suite(
        datasets=PAPER_RUN_CROSS_FOUNDATION["datasets"],
        encoders=PAPER_RUN_CROSS_FOUNDATION["encoders"],
        methods=methods or PAPER_RUN_CROSS_FOUNDATION["methods"],
        cfg=PAPER_RUN_CROSS_FOUNDATION["cfg"],
        baseline_cfg=bcfg,
        seeds=PAPER_RUN_CROSS_FOUNDATION["seeds"],
        n_heads=PAPER_RUN_CROSS_FOUNDATION["n_heads"],
        baseline_heads=PAPER_RUN_CROSS_FOUNDATION["baseline_heads"],
        n_init=PAPER_RUN_CROSS_FOUNDATION["n_init"],
        device=device,
        cache_dir="./feat_cache",
        out_root="./bcpt_med_outputs_cross",
        mutual=None,
    )


# =============================================================================
# 11. CLI entry point
# =============================================================================


def parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="BCPT-Med offline MedMNIST clustering with AAAI comparison baselines"
    )
    p.add_argument("--install", action="store_true", help="install Python dependencies and exit")
    p.add_argument(
        "--self-test",
        action="store_true",
        help="verify bounded-simplex feasibility and partial-transport mass conservation",
    )
    p.add_argument("--quick", action="store_true", help="run a short sanity experiment")
    p.add_argument("--cross", action="store_true", help="run the cross-foundation paper configuration")
    p.add_argument(
        "--aaai-final-suite", action="store_true",
        help="run clean comparison, component ablations, IR 10/50/100, and safeguard stresses",
    )
    p.add_argument("--list-methods", action="store_true", help="print supported method names and exit")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--encoders", nargs="+", default=None)
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="subset of methods; use 'all' or names from --list-methods",
    )
    p.add_argument("--epochs", type=int, default=60, help="BCPT-Med epochs")
    p.add_argument("--baseline-epochs", type=int, default=None, help="external-baseline epochs; default: --epochs")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-heads", type=int, default=5, help="candidate heads per BCPT-Med seed")
    p.add_argument(
        "--baseline-heads",
        type=int,
        default=None,
        help="candidate heads per neural baseline seed; default: same as --n-heads",
    )
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--feature-batch", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument(
        "--imbalance-ratio",
        type=float,
        default=None,
        help="optional controlled exponential long-tail ratio (e.g. 10, 50, 100)",
    )
    p.add_argument(
        "--imbalance-seed",
        type=int,
        default=0,
        help="seed controlling which semantic classes become head/tail classes",
    )
    p.add_argument("--imbalance-ratios", nargs="+", type=float, default=[10.0, 50.0, 100.0])
    p.add_argument("--stress-test", choices=["clean"] + STRESS_TESTS, default="clean")
    p.add_argument("--stress-tests", nargs="+", choices=STRESS_TESTS, default=STRESS_TESTS)
    p.add_argument("--stress-strength", type=float, default=0.35)
    p.add_argument("--reduced-sample-fraction", type=float, default=0.50)
    p.add_argument("--wrong-k-delta", type=int, default=1)
    p.add_argument("--out-root", default="./bcpt_med_outputs")
    p.add_argument("--cache-dir", default="./feat_cache")
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "reuse complete method/seed rows from an existing all_seed_results.csv "
            "under --out-root; methods with missing seeds are rerun"
        ),
    )
    p.add_argument(
        "--offline-pack-dir",
        default=None,
        help=(
            "path to offline_pack; default is ./offline_pack relative to this script "
            "or BCPT_OFFLINE_PACK environment variable (legacy TANGO_OFFLINE_PACK is accepted)"
        ),
    )
    p.add_argument(
        "--no-auto-download", action="store_true",
        help="require MedMNIST files to exist instead of downloading missing files",
    )
    p.add_argument("--setup-native-baselines", action="store_true")
    p.add_argument(
        "--native-baselines", nargs="+", choices=list(NATIVE_BASELINE_SPECS), default=None,
        help="generate or execute official P2OT/SP2OT/PROTOCOL repository commands",
    )
    p.add_argument("--native-dataset", default="cifar100")
    p.add_argument("--native-imbalance-ratio", type=float, default=100.0)
    p.add_argument("--execute-native", action="store_true")
    p.add_argument("--native-python", default=sys.executable)
    p.add_argument("--cpu", action="store_true")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print("[info] ignoring unknown args:", unknown)
    return args


def main(argv=None):
    global AUTO_DOWNLOAD_MEDMNIST
    args = parse_args(argv)

    if args.self_test:
        run_algorithmic_self_test()
        return

    if args.list_methods:
        print("Supported methods:")
        for key in list(dict.fromkeys(DEFAULT_ALL_METHODS + COMPONENT_ABLATIONS)):
            print(f"  {key:18s} -> {METHOD_LABELS[key]}")
        return

    configure_offline_paths(args.offline_pack_dir, verbose=True)
    AUTO_DOWNLOAD_MEDMNIST = not args.no_auto_download

    if args.install:
        install_colab_dependencies(include_biomedclip=True)
        return

    if args.quick:
        run_quick_test(methods=args.methods)
        return

    native_root = OFFLINE_PACK_DIR / "native_baselines"
    if args.setup_native_baselines:
        setup_native_baselines(native_root)
    if args.native_baselines:
        run_native_baseline_adapters(
            args.native_baselines, native_root, args.native_dataset,
            args.native_imbalance_ratio, os.path.join(args.out_root, "native_runs"),
            execute=args.execute_native, python_executable=args.native_python,
        )
        if not args.aaai_final_suite and args.datasets is None and not args.cross:
            return

    baseline_epochs = args.baseline_epochs if args.baseline_epochs is not None else args.epochs
    baseline_heads = args.baseline_heads if args.baseline_heads is not None else args.n_heads

    if args.cross and args.datasets is None:
        run_paper_cross_foundation(methods=args.methods, baseline_epochs=baseline_epochs)
        return

    device = get_device(force_cpu=args.cpu)
    cfg = BCPTConfig(epochs=args.epochs, batch_size=args.batch_size)
    bcfg = BaselineConfig(epochs=baseline_epochs, batch_size=args.batch_size)

    if args.aaai_final_suite:
        run_aaai_final_suite(
            datasets=args.datasets or ["pathmnist", "bloodmnist", "organamnist"],
            encoders=args.encoders or ["dinov2_vitb14", "biomedclip"],
            cfg=cfg, baseline_cfg=bcfg, seeds=args.seeds,
            n_heads=args.n_heads, baseline_heads=baseline_heads,
            n_init=args.n_init, device=device, cache_dir=args.cache_dir,
            out_root=args.out_root, feature_batch=args.feature_batch,
            img_size=args.img_size, reuse_existing=args.reuse_existing,
            imbalance_ratios=args.imbalance_ratios, stress_tests=args.stress_tests,
            stress_strength=args.stress_strength,
            reduced_sample_fraction=args.reduced_sample_fraction,
            wrong_k_delta=args.wrong_k_delta,
        )
        return

    if args.datasets is not None:
        datasets = args.datasets
        encoders = args.encoders or ["dinov2_vitb14"]
        run_suite(
            datasets=datasets,
            encoders=encoders,
            methods=args.methods or DEFAULT_ALL_METHODS,
            cfg=cfg,
            baseline_cfg=bcfg,
            seeds=args.seeds,
            n_heads=args.n_heads,
            baseline_heads=baseline_heads,
            n_init=args.n_init,
            device=device,
            cache_dir=args.cache_dir,
            out_root=args.out_root,
            feature_batch=args.feature_batch,
            img_size=args.img_size,
            mutual=None,
            imbalance_ratio=args.imbalance_ratio,
            imbalance_seed=args.imbalance_seed,
            stress_test=args.stress_test,
            stress_strength=args.stress_strength,
            reduced_sample_fraction=args.reduced_sample_fraction,
            wrong_k_delta=args.wrong_k_delta,
            reuse_existing=args.reuse_existing,
        )
    else:
        run_paper_single_backbone(methods=args.methods, baseline_epochs=baseline_epochs)


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in {"ZMQInteractiveShell", "Shell"}
    except Exception:
        return False


if __name__ == "__main__" and not _running_in_notebook():
    main()
