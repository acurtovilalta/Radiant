"""Post-hoc diagnostics of the joint X-ray/MRI embedding space (modality mixing, anatomy preservation)."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radiant.data import MRIDataConfig, XRayDataConfig, build_mri_dataloader, build_xray_dataloader
from radiant.models import MultiModalEncoderSystem, MultiModalEncoderSystemConfig
from radiant.training import trainer as trainer_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the joint X-ray/MRI embedding space from a trained checkpoint.")
    parser.add_argument("--config", required=True, help="Path to trainer YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--output-json", required=True, help="Path to output metrics JSON.")
    parser.add_argument("--max-samples-per-modality", type=int, default=1000, help="Max val samples per modality.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, eps, None)
    return array / norms


def _clone_xray_val_config(config: XRayDataConfig) -> XRayDataConfig:
    payload = asdict(config)
    payload["split"] = "val"
    payload["shuffle"] = False
    payload["drop_last"] = False
    payload["num_workers"] = int(max(1, payload.get("num_workers", 0)))
    return XRayDataConfig(**payload)


def _clone_mri_val_config(config: MRIDataConfig) -> MRIDataConfig:
    payload = asdict(config)
    payload["split"] = "val"
    payload["shuffle"] = False
    payload["drop_last"] = False
    payload["num_workers"] = int(max(1, payload.get("num_workers", 0)))
    return MRIDataConfig(**payload)


def _load_model_system(config_path: str, checkpoint_path: str, device: torch.device) -> tuple[Any, MultiModalEncoderSystem]:
    payload = trainer_module.load_yaml(config_path)
    config = trainer_module._build_config(payload)  # Reuse trainer config parsing.
    model_system = MultiModalEncoderSystem(
        MultiModalEncoderSystemConfig(
            xray=config.xray_model,
            mri=config.mri_model,
            xray_device=str(device),
            mri_device=str(device),
            output_device=str(device),
        )
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_system.xray_encoder.load_state_dict(checkpoint["xray_encoder"])
    model_system.mri_encoder.load_state_dict(checkpoint["mri_encoder"])
    model_system.eval()
    return config, model_system.to(device)


@torch.no_grad()
def collect_embeddings(
    model_system: MultiModalEncoderSystem,
    loader: Any,
    modality: str,
    device: torch.device,
    max_samples: int,
) -> tuple[np.ndarray, list[str]]:
    if modality not in {"xray", "mri"}:
        raise ValueError(f"Unsupported modality '{modality}'.")
    embeddings: list[np.ndarray] = []
    families: list[str] = []
    collected = 0
    for batch in loader:
        if collected >= max_samples:
            break
        views = batch["global_views"][0].to(device, non_blocking=True)
        if modality == "xray":
            output = model_system(xray=views, move_to_output_device=True)["xray"]
        else:
            output = model_system(
                mri=views,
                move_to_output_device=True,
                compute_projection_bridge=False,
            )["mri"]
        emb = output.embedding.detach().cpu().to(torch.float32).numpy()
        remaining = max_samples - collected
        take = min(remaining, emb.shape[0])
        embeddings.append(emb[:take])
        families.extend(str(family) for family in batch["anatomy_family"][:take])
        collected += take
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.concatenate(embeddings, axis=0), families


def _train_test_split_with_optional_stratify(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    classes, counts = np.unique(labels, return_counts=True)
    use_stratify = (
        classes.size >= 2
        and np.min(counts) >= 2
        and int(np.floor(0.3 * labels.shape[0])) >= classes.size
        and int(np.floor(0.7 * labels.shape[0])) >= classes.size
    )
    stratify = labels if use_stratify else None
    return train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=seed,
        stratify=stratify,
    )


def linear_probe_balanced_accuracy(features: np.ndarray, labels: np.ndarray, seed: int) -> float | None:
    if features.shape[0] < 4:
        return None
    if np.unique(labels).size < 2:
        return None
    x_train, x_test, y_train, y_test = _train_test_split_with_optional_stratify(features, labels, seed)
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return None
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    return float(balanced_accuracy_score(y_test, predictions))


def within_anatomy_modality_silhouette(
    xray_embeddings: np.ndarray,
    xray_families: list[str],
    mri_embeddings: np.ndarray,
    mri_families: list[str],
) -> float | None:
    scores: list[float] = []
    all_families = sorted(set(xray_families).intersection(mri_families))
    for family in all_families:
        x_idx = [index for index, value in enumerate(xray_families) if value == family]
        m_idx = [index for index, value in enumerate(mri_families) if value == family]
        if len(x_idx) == 0 or len(m_idx) == 0:
            continue
        if len(x_idx) + len(m_idx) < 3:
            continue
        features = np.concatenate([xray_embeddings[x_idx], mri_embeddings[m_idx]], axis=0)
        modality_labels = np.asarray([0] * len(x_idx) + [1] * len(m_idx), dtype=np.int64)
        # silhouette_score requires at least 2 labels and n_samples > n_labels.
        if np.unique(modality_labels).size < 2 or features.shape[0] <= 2:
            continue
        scores.append(float(silhouette_score(features, modality_labels, metric="cosine")))
    if not scores:
        return None
    return float(np.mean(scores))


def _retrieval_topk_accuracy(
    query_embeddings: np.ndarray,
    query_families: list[str],
    ref_embeddings: np.ndarray,
    ref_families: list[str],
    k: int,
) -> float | None:
    if query_embeddings.shape[0] == 0 or ref_embeddings.shape[0] == 0:
        return None
    k_eff = min(int(k), int(ref_embeddings.shape[0]))
    similarities = query_embeddings @ ref_embeddings.T
    correct = 0
    for index in range(similarities.shape[0]):
        row = similarities[index]
        topk_idx = np.argpartition(-row, kth=k_eff - 1)[:k_eff]
        query_family = query_families[index]
        if any(ref_families[j] == query_family for j in topk_idx):
            correct += 1
    return float(correct / similarities.shape[0])


def _entropy(probabilities: torch.Tensor, dim: int) -> torch.Tensor:
    probs = probabilities.clamp_min(1e-12)
    return -(probs * torch.log(probs)).sum(dim=dim)


@torch.no_grad()
def compute_projection_bridge_diagnostics(
    model_system: MultiModalEncoderSystem,
    xray_loader: Any,
    mri_loader: Any,
    device: torch.device,
    max_samples_per_modality: int,
) -> dict[str, float | None]:
    """Entropy of the projection bridge's depth attention (how focused each of the K projections is)."""
    depth_entropies: list[float] = []
    depth_entropies_norm: list[float] = []

    xray_seen = 0
    mri_seen = 0
    bridge_outputs_seen = 0

    for x_batch, m_batch in zip(xray_loader, mri_loader):
        if xray_seen >= max_samples_per_modality or mri_seen >= max_samples_per_modality:
            break

        x_views = x_batch["global_views"][0].to(device, non_blocking=True)
        m_views = m_batch["global_views"][0].to(device, non_blocking=True)

        x_out = model_system(xray=x_views, move_to_output_device=True)["xray"]
        m_out = model_system(
            mri=m_views,
            move_to_output_device=True,
            compute_projection_bridge=True,
        )["mri"]

        xray_seen += int(x_out.embedding.shape[0])
        mri_seen += int(m_out.embedding.shape[0])

        mri_proj_embs = m_out.mri_proj_embs
        mri_depth_attn = m_out.mri_depth_attn
        if mri_proj_embs is None or mri_depth_attn is None:
            continue

        bridge_outputs_seen += int(mri_proj_embs.shape[0])

        # Depth attention entropy over D for each [B, K].
        depth_prob = mri_depth_attn
        depth_entropy = _entropy(depth_prob, dim=-1)  # [B, K]
        depth_entropies.extend(depth_entropy.flatten().detach().cpu().tolist())
        depth_dim = int(depth_prob.shape[-1])
        if depth_dim > 1:
            depth_entropies_norm.extend((depth_entropy / math.log(depth_dim)).flatten().detach().cpu().tolist())
        else:
            depth_entropies_norm.extend([0.0] * int(depth_entropy.numel()))

    if bridge_outputs_seen == 0:
        return {
            "projection_depth_attention_entropy_mean": None,
            "projection_depth_attention_entropy_normalized_mean": None,
        }

    depth_mean = float(np.mean(depth_entropies)) if depth_entropies else None
    depth_norm_mean = float(np.mean(depth_entropies_norm)) if depth_entropies_norm else None

    return {
        "projection_depth_attention_entropy_mean": depth_mean,
        "projection_depth_attention_entropy_normalized_mean": depth_norm_mean,
    }


def build_counts(
    xray_families: list[str],
    mri_families: list[str],
) -> dict[str, Any]:
    families = sorted(set(xray_families).union(mri_families))
    per_family: dict[str, dict[str, int]] = {}
    for family in families:
        x_count = sum(1 for value in xray_families if value == family)
        m_count = sum(1 for value in mri_families if value == family)
        per_family[family] = {
            "xray": int(x_count),
            "mri": int(m_count),
            "total": int(x_count + m_count),
        }
    return {
        "xray_embeddings": int(len(xray_families)),
        "mri_embeddings": int(len(mri_families)),
        "anatomy_families": int(len(families)),
        "per_family_counts": per_family,
    }


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config, model_system = _load_model_system(args.config, args.checkpoint, device=device)

    xray_val_loader = build_xray_dataloader(_clone_xray_val_config(config.xray_data))
    mri_val_loader = build_mri_dataloader(_clone_mri_val_config(config.mri_data))

    xray_embeddings, xray_families = collect_embeddings(
        model_system=model_system,
        loader=xray_val_loader,
        modality="xray",
        device=device,
        max_samples=int(args.max_samples_per_modality),
    )
    mri_embeddings, mri_families = collect_embeddings(
        model_system=model_system,
        loader=mri_val_loader,
        modality="mri",
        device=device,
        max_samples=int(args.max_samples_per_modality),
    )

    if xray_embeddings.shape[0] == 0 or mri_embeddings.shape[0] == 0:
        raise RuntimeError(
            "Failed to collect validation embeddings for one or both modalities. "
            "Check manifests/splits and max-samples-per-modality."
        )

    xray_embeddings = l2_normalize(xray_embeddings.astype(np.float32))
    mri_embeddings = l2_normalize(mri_embeddings.astype(np.float32))

    all_embeddings = np.concatenate([xray_embeddings, mri_embeddings], axis=0)
    modality_labels = np.asarray([0] * xray_embeddings.shape[0] + [1] * mri_embeddings.shape[0], dtype=np.int64)
    anatomy_labels = np.asarray(xray_families + mri_families)

    modality_probe = linear_probe_balanced_accuracy(all_embeddings, modality_labels, seed=int(args.seed))
    anatomy_probe = linear_probe_balanced_accuracy(all_embeddings, anatomy_labels, seed=int(args.seed))
    silhouette = within_anatomy_modality_silhouette(
        xray_embeddings=xray_embeddings,
        xray_families=xray_families,
        mri_embeddings=mri_embeddings,
        mri_families=mri_families,
    )

    top1_x2m = _retrieval_topk_accuracy(
        query_embeddings=xray_embeddings,
        query_families=xray_families,
        ref_embeddings=mri_embeddings,
        ref_families=mri_families,
        k=1,
    )
    top1_m2x = _retrieval_topk_accuracy(
        query_embeddings=mri_embeddings,
        query_families=mri_families,
        ref_embeddings=xray_embeddings,
        ref_families=xray_families,
        k=1,
    )
    top5_x2m = _retrieval_topk_accuracy(
        query_embeddings=xray_embeddings,
        query_families=xray_families,
        ref_embeddings=mri_embeddings,
        ref_families=mri_families,
        k=5,
    )
    top5_m2x = _retrieval_topk_accuracy(
        query_embeddings=mri_embeddings,
        query_families=mri_families,
        ref_embeddings=xray_embeddings,
        ref_families=xray_families,
        k=5,
    )
    projection_diag = compute_projection_bridge_diagnostics(
        model_system=model_system,
        xray_loader=xray_val_loader,
        mri_loader=mri_val_loader,
        device=device,
        max_samples_per_modality=int(args.max_samples_per_modality),
    )

    metrics: dict[str, Any] = {
        "modality_linear_probe_balanced_acc": modality_probe,
        "anatomy_linear_probe_balanced_acc": anatomy_probe,
        "within_anatomy_modality_silhouette": silhouette,
        "cross_modal_anatomy_retrieval_top1": {
            "xray_to_mri": top1_x2m,
            "mri_to_xray": top1_m2x,
        },
        "cross_modal_anatomy_retrieval_top5": {
            "xray_to_mri": top5_x2m,
            "mri_to_xray": top5_m2x,
        },
        "embedding_counts": build_counts(xray_families=xray_families, mri_families=mri_families),
        **projection_diag,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
