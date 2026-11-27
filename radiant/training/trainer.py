"""Staged trainer for RADIANT: modality-specific warm-up -> joint X-ray + MRI training."""

from __future__ import annotations

import importlib
import itertools
import datetime
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from radiant.data.anatomy import UNKNOWN_ANATOMY_LABEL
from radiant.data.datasets import (
    MRIDataConfig,
    SharedFamilyBatchSampler,
    XRayDataConfig,
    build_mri_dataloader,
    build_xray_dataloader,
)
from radiant.losses import (
    CentroidAlignmentConfig,
    SwAVAssigner,
    SwAVAssignmentConfig,
    SwAVLossConfig,
    XRayViewHead,
    centroid_alignment_loss,
    projection_attention_regularization,
    projection_mil_contrastive_loss,
    swav_loss,
    xray_family_mil_loss,
)
from radiant.models import (
    AnatomyPrototypeBanks,
    EncoderOutput,
    MRIEncoderConfig,
    MultiModalEncoderSystem,
    MultiModalEncoderSystemConfig,
    PrototypeBankConfig,
    XRayEncoderConfig,
)
from radiant.models.dinov3_backbone import get_embedding_and_norm_modules, get_transformer_blocks
from radiant.models.xray_encoder import get_resnet_stage_modules
from radiant.losses.xray_view import encode_categorical_labels, normalize_metadata_label

def _import_wandb() -> Any:
    """Import the real wandb package without letting the repo's run-artifact directory shadow it."""
    repo_root = Path(__file__).resolve().parents[2]
    original_sys_path = list(sys.path)
    try:
        sys.path = [
            entry
            for entry in original_sys_path
            if Path(entry or ".").resolve() != repo_root
        ]
        module = importlib.import_module("wandb")
    except Exception:  # pragma: no cover - optional dependency
        return None
    finally:
        sys.path = original_sys_path
    return module if hasattr(module, "init") else None


wandb = _import_wandb()


BRIDGE_GRADIENT_AUDIT_KEYS: tuple[str, ...] = (
    "grad_norm/xray_encoder_total",
    "grad_norm/mri_encoder_total",
    "grad_norm/mri_projection_bridge",
    "grad_norm/mri_projection_head",
    "grad_norm/mri_layer4",
    "grad_norm/xray_layer4",
    "grad_norm/prototype_banks",
    "grad_norm/xray_view_head",
)


MRI_BACKBONE_STAGE_PREFIXES: dict[str, str] = {
    "conv1": "conv1.",
    "bn1": "bn1.",
    "stem": "stem.",  # 2D slice encoder of control C4
    "layer1": "layer1.",
    "layer2": "layer2.",
    "layer3": "layer3.",
    "layer4": "layer4.",
}


@dataclass(slots=True)
class StageScheduleConfig:
    """Training-stage schedule for staged multimodal optimization."""

    warm_start_epochs: int = 20
    joint_epochs: int = 46
    cross_modal_ramp_epochs: int = 10
    enable_xray_vit_gradual_unfreeze: bool = False
    xray_vit_unfreeze_start_epoch: int = 0
    xray_vit_unfreeze_epochs_per_stage: int = 1
    xray_vit_unfreeze_blocks_per_stage: int = 1
    xray_vit_freeze_patch_embed_until_full: bool = True
    enable_xray_cnn_gradual_unfreeze: bool = False
    xray_cnn_unfreeze_start_epoch: int = 0
    xray_cnn_unfreeze_epochs_per_stage: int = 1
    xray_cnn_unfreeze_stages_per_step: int = 1


@dataclass(slots=True)
class OptimizerConfig:
    """Optimizer and scheduler settings."""

    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip_norm: float | None = 1.0
    min_lr_ratio: float = 0.1
    warmup_epochs: int = 1


@dataclass(slots=True)
class CheckpointConfig:
    """Checkpoint and resume configuration.

    `final.pt` is written once, after the last epoch, and is the checkpoint used for
    downstream evaluation. `latest.pt` (every `save_every_epochs` and at the end) is for resuming.
    """

    output_dir: str = "checkpoints"
    save_every_epochs: int = 1
    keep_last_only: bool = False
    resume_path: str | None = None


@dataclass(slots=True)
class WandBConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = False
    project: str = "radiant"
    entity: str | None = None
    run_name: str | None = None
    mode: str | None = None
    tags: list[str] = field(default_factory=list)
    group: str | None = None


@dataclass(slots=True)
class DistributedConfig:
    """Distributed data-parallel training configuration."""

    enabled: bool = False
    backend: str = "nccl"
    timeout_minutes: int = 120


@dataclass(slots=True)
class RuntimeConfig:
    """Low-level runtime configuration for CUDA and cuDNN behavior."""

    cudnn_enabled: bool = True
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = False
    allow_cudnn_fallback_on_init_error: bool = True


@dataclass(slots=True)
class AblationConfig:
    """Top-level methodology ablations exposed as simple config flags."""

    # "anatomy_specific": one prototype bank per anatomy family; "global": a single bank shared by
    # all families (ablation A5, no anatomy conditioning).
    prototype_scope: str = "anatomy_specific"
    # False: X-ray-only pretraining (RADIANT-X, C1); the joint stage is skipped.
    enable_mri: bool = True


@dataclass(slots=True)
class TrainerConfig:
    """Full trainer configuration."""

    seed: int = 13
    output_dir: str = "outputs"
    use_amp: bool = True
    log_every_steps: int = 20
    log_bridge_gradient_audit: bool = False
    bridge_gradient_audit_every_steps: int = 100
    val_every_epochs: int = 1
    xray_data: XRayDataConfig = field(default_factory=lambda: XRayDataConfig(manifest_path=""))
    mri_data: MRIDataConfig = field(default_factory=lambda: MRIDataConfig(manifest_path=""))
    xray_model: XRayEncoderConfig = field(default_factory=XRayEncoderConfig)
    mri_model: MRIEncoderConfig = field(default_factory=MRIEncoderConfig)
    prototype_banks: PrototypeBankConfig = field(default_factory=lambda: PrototypeBankConfig(embedding_dim=256))
    xray_assignment: SwAVAssignmentConfig = field(default_factory=SwAVAssignmentConfig)
    mri_assignment: SwAVAssignmentConfig = field(default_factory=lambda: SwAVAssignmentConfig(use_feature_queue=True, queue_size=256))
    xray_swav_loss: SwAVLossConfig = field(default_factory=SwAVLossConfig)
    mri_swav_loss: SwAVLossConfig = field(default_factory=SwAVLossConfig)
    centroid_alignment: CentroidAlignmentConfig = field(default_factory=CentroidAlignmentConfig)
    schedule: StageScheduleConfig = field(default_factory=StageScheduleConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    # X-ray view/plane prediction from the X-ray embedding (warm-up stage only).
    lambda_xray_view: float = 0.0
    # Cross-modal prototype-centroid alignment (joint stage, ramped in over cross_modal_ramp_epochs).
    lambda_centroid_alignment: float = 0.0
    lambda_projection_mil_contrastive: float = 0.0
    lambda_projection_attention_diversity: float = 0.0
    projection_mil_temperature: float = 0.07
    # Control C1 (X-ray-only runs): the projection MIL objective with same-family X-rays in place of
    # MRI projection bags -- see Trainer._xray_family_mil_loss. 0.0 (default) disables it.
    lambda_xray_family_mil: float = 0.0
    # Weight on the MRI loss in the backward pass (warm-up MRI steps and joint steps). Control C3 uses
    # 0.25 so the shared encoder keeps learning X-rays.
    lambda_mri_loss: float = 1.0
    ablations: AblationConfig = field(default_factory=AblationConfig)
    devices: dict[str, str | None] = field(
        default_factory=lambda: {"xray": None, "mri": None, "output": None}
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dataclass_from_dict(config_type: type[Any], payload: dict[str, Any]) -> Any:
    field_names = {field.name for field in config_type.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {key: value for key, value in payload.items() if key in field_names}
    return config_type(**filtered)


def _build_config(payload: dict[str, Any]) -> TrainerConfig:
    return TrainerConfig(
        seed=int(payload.get("seed", 13)),
        output_dir=str(payload.get("output_dir", "outputs")),
        use_amp=bool(payload.get("use_amp", True)),
        log_every_steps=int(payload.get("log_every_steps", 20)),
        log_bridge_gradient_audit=bool(payload.get("log_bridge_gradient_audit", False)),
        bridge_gradient_audit_every_steps=int(payload.get("bridge_gradient_audit_every_steps", 100)),
        val_every_epochs=int(payload.get("val_every_epochs", 1)),
        xray_data=_dataclass_from_dict(XRayDataConfig, payload.get("xray_data", {})),
        mri_data=_dataclass_from_dict(MRIDataConfig, payload.get("mri_data", {})),
        xray_model=_dataclass_from_dict(XRayEncoderConfig, payload.get("xray_model", {})),
        mri_model=_dataclass_from_dict(MRIEncoderConfig, payload.get("mri_model", {})),
        prototype_banks=_dataclass_from_dict(PrototypeBankConfig, payload.get("prototype_banks", {})),
        xray_assignment=_dataclass_from_dict(SwAVAssignmentConfig, payload.get("xray_assignment", {})),
        mri_assignment=_dataclass_from_dict(SwAVAssignmentConfig, payload.get("mri_assignment", {})),
        xray_swav_loss=_dataclass_from_dict(SwAVLossConfig, payload.get("xray_swav_loss", {})),
        mri_swav_loss=_dataclass_from_dict(SwAVLossConfig, payload.get("mri_swav_loss", {})),
        centroid_alignment=_dataclass_from_dict(CentroidAlignmentConfig, payload.get("centroid_alignment", {})),
        schedule=_dataclass_from_dict(StageScheduleConfig, payload.get("schedule", {})),
        optimizer=_dataclass_from_dict(OptimizerConfig, payload.get("optimizer", {})),
        checkpoint=_dataclass_from_dict(CheckpointConfig, payload.get("checkpoint", {})),
        wandb=_dataclass_from_dict(WandBConfig, payload.get("wandb", {})),
        distributed=_dataclass_from_dict(DistributedConfig, payload.get("distributed", {})),
        runtime=_dataclass_from_dict(RuntimeConfig, payload.get("runtime", {})),
        lambda_xray_view=float(payload.get("lambda_xray_view", 0.0) or 0.0),
        lambda_centroid_alignment=float(payload.get("lambda_centroid_alignment", 0.0) or 0.0),
        lambda_projection_mil_contrastive=float(payload.get("lambda_projection_mil_contrastive", 0.0) or 0.0),
        lambda_projection_attention_diversity=float(
            payload.get("lambda_projection_attention_diversity", 0.0) or 0.0
        ),
        projection_mil_temperature=float(payload.get("projection_mil_temperature", 0.07) or 0.07),
        lambda_xray_family_mil=float(payload.get("lambda_xray_family_mil", 0.0) or 0.0),
        lambda_mri_loss=float(payload.get("lambda_mri_loss", 1.0)),
        ablations=_dataclass_from_dict(AblationConfig, payload.get("ablations", {})),
        devices=dict(payload.get("devices", {"xray": None, "mri": None, "output": None})),
    )


def set_seed(seed: int) -> None:
    """Set seeds for reproducible training behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def cosine_with_warmup(epoch: int, total_epochs: int, base_lr: float, warmup_epochs: int, min_lr: float) -> float:
    """Cosine schedule with linear warmup at epoch granularity."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_lr + (base_lr - min_lr) * cosine


def build_trainer_from_yaml(path: str | Path) -> "StageTrainer":
    """Construct a trainer directly from a YAML configuration file."""
    return StageTrainer(_build_config(load_yaml(path)))


def train_from_config(path: str | Path) -> dict[str, Any]:
    """Convenience entry point for YAML-driven training."""
    trainer = build_trainer_from_yaml(path)
    return trainer.train()


class WandBLogger:
    """Thin WandB wrapper with graceful fallback when disabled."""

    def __init__(self, config: WandBConfig, full_config: TrainerConfig, enabled: bool = True) -> None:
        self.enabled = bool(enabled and config.enabled and wandb is not None)
        self.run = None
        if not self.enabled:
            return
        self.run = wandb.init(
            project=config.project,
            entity=config.entity,
            name=config.run_name,
            mode=config.mode,
            tags=config.tags,
            group=config.group,
            config=asdict(full_config),
        )

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(payload, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def _cycle(loader: DataLoader[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Create an endless iterator over a dataloader."""
    while True:
        for batch in loader:
            yield batch


def _batch_to_cpu_device_type(batch_device: torch.device) -> str:
    return "cuda" if batch_device.type == "cuda" else "cpu"


def _rank_print_label() -> str:
    if "RANK" not in os.environ:
        return "[single]"
    return f"[rank={os.environ.get('RANK')} local_rank={os.environ.get('LOCAL_RANK')}]"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file atomically to avoid partial or interleaved outputs."""
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class StageTrainer:
    """Two-stage trainer: modality-specific warm-up, then joint X-ray + MRI training."""

    def __init__(self, config: TrainerConfig) -> None:
        self.config = self._apply_ablations(config)
        self.rank, self.world_size, self.local_rank, self.is_distributed = self._setup_distributed()
        set_seed(config.seed + self.rank)
        self._configure_runtime()
        self._validate_embedding_dimension_compatibility()
        self._cudnn_fallback_applied = False
        self._projection_joint_path_context = False

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        if self.is_distributed:
            local_device = f"cuda:{self.local_rank}"
            xray_device = local_device
            mri_device = local_device
            output_device = local_device
        else:
            xray_device = self.config.devices.get("xray")
            mri_device = self.config.devices.get("mri")
            output_device = self.config.devices.get("output")
        self.model_system = MultiModalEncoderSystem(
            MultiModalEncoderSystemConfig(
                xray=self.config.xray_model,
                mri=self.config.mri_model,
                xray_device=xray_device,
                mri_device=mri_device,
                output_device=output_device or xray_device or mri_device or "cpu",
            )
        )
        self.xray_encoder_module = self.model_system.xray_encoder
        self.mri_encoder_module = self.model_system.mri_encoder
        self.loss_device = self.model_system.output_device
        self._load_pretrained_mri_backbone()

        self.xray_train_sampler = self._build_sampler(self.config.xray_data, train=True)
        self.mri_train_sampler = self._build_sampler(self.config.mri_data, train=True)
        self.xray_train_loader = build_xray_dataloader(self.config.xray_data, sampler=self.xray_train_sampler)
        self.mri_train_loader = build_mri_dataloader(self.config.mri_data, sampler=self.mri_train_sampler)
        self.xray_joint_train_sampler = None
        self.mri_joint_train_sampler = None
        self.xray_joint_train_loader = None
        self.mri_joint_train_loader = None
        self._build_joint_family_train_loaders()
        self.total_training_steps_estimate = self._estimate_total_training_steps()
        self._sync_anatomy_families_from_training_data()

        prototype_config = self.config.prototype_banks
        self.prototype_banks = AnatomyPrototypeBanks(prototype_config).to(self.loss_device)
        self.prototype_banks_module = self.prototype_banks

        if self.is_distributed:
            # dinov3_vitb16 always needs this (unused pooler/head outputs regardless
            # of gradual unfreeze). enable_xray_cnn_gradual_unfreeze needs it too:
            # _apply_xray_cnn_gradual_unfreeze toggles resnet stage .requires_grad_()
            # every epoch (all stages frozen through warm_start, then progressively
            # unfrozen), and DDP's default assumes a fixed set of grad-producing
            # parameters across iterations. No static_graph either: the trainable set
            # genuinely changes epoch to epoch.
            xray_find_unused_parameters = self.config.xray_model.backbone == "dinov3_vitb16" or bool(
                self.config.schedule.enable_xray_cnn_gradual_unfreeze
            )
            self.model_system.xray_encoder = DistributedDataParallel(
                self.model_system.xray_encoder,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=xray_find_unused_parameters,
                broadcast_buffers=False,
            )
            # Bridge/head parameters are unused during warm-up and become active
            # during joint training, so the unused set is dynamic across stages.
            mri_find_unused_parameters = bool(self.config.mri_model.use_mri_projection_bridge)
            # encoder_type "shared_xray" (C3) has a parameter-free MRI module: nothing to wrap.
            if any(True for _ in self.model_system.mri_encoder.parameters()):
                self.model_system.mri_encoder = DistributedDataParallel(
                    self.model_system.mri_encoder,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=mri_find_unused_parameters,
                    broadcast_buffers=False,
                )
            # The shared prototype banks are invoked multiple times per iteration
            # (assignment crops, prediction crops, and both modalities during joint
            # training). Wrapping them with DDP can trigger reducer errors like
            # "Expected to mark a variable ready only once" when the same family bank
            # participates in multiple autograd paths in one step. We therefore keep
            # prototype banks as a plain module and synchronize gradients manually in
            # _backward_step, mirroring the handling used for the X-ray view head below.

        self.xray_assigner = SwAVAssigner(self.prototype_banks, self.config.xray_assignment).to(self.loss_device)
        self.mri_assigner = SwAVAssigner(self.prototype_banks, self.config.mri_assignment).to(self.loss_device)

        self.xray_val_loader = None
        self.mri_val_loader = None
        self.xray_val_sampler = None
        self.mri_val_sampler = None
        if self.config.xray_data.split != "val":
            xray_val_cfg = XRayDataConfig(**{**asdict(self.config.xray_data), "split": "val", "shuffle": False})
            self.xray_val_sampler = self._build_sampler(xray_val_cfg, train=False)
            self.xray_val_loader = build_xray_dataloader(xray_val_cfg, sampler=self.xray_val_sampler)
        if self.config.mri_data.split != "val":
            mri_val_cfg = MRIDataConfig(**{**asdict(self.config.mri_data), "split": "val", "shuffle": False})
            self.mri_val_sampler = self._build_sampler(mri_val_cfg, train=False)
            self.mri_val_loader = build_mri_dataloader(mri_val_cfg, sampler=self.mri_val_sampler)
        self.xray_view_label_to_id = self._build_metadata_label_to_id(self.xray_train_loader, field_name="view_label")
        # Plain module (not DDP-wrapped): it is unused in joint-stage steps; its gradients are
        # synchronized manually in _backward_step.
        self.xray_view_head = XRayViewHead(
            embedding_dim=int(self.config.xray_model.embedding_dim),
            num_classes=len(self.xray_view_label_to_id),
        ).to(self.loss_device)
        self._assert_joint_regularizer_shapes()

        self.optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in itertools.chain(
                    self.model_system.parameters(),
                    self.prototype_banks.parameters(),
                    self.xray_view_head.parameters(),
                )
                if parameter.requires_grad
            ],
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=self.config.optimizer.betas,
            eps=self.config.optimizer.eps,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=bool(self.config.use_amp and self.loss_device.type == "cuda"),
        )
        self.wandb = WandBLogger(
            self.config.wandb,
            self.config,
            enabled=self.rank == 0,
        )

        self.state = {
            "epoch": 0,
            "global_step": 0,
        }
        self._last_gradient_audit: dict[str, float] = {}
        self._last_gradient_audit_step: int | None = None
        if self.config.checkpoint.resume_path:
            self.load_checkpoint(self.config.checkpoint.resume_path)

    def _setup_distributed(self) -> tuple[int, int, int, bool]:
        torchrun_env_present = any(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
        if not self.config.distributed.enabled:
            if torchrun_env_present:
                raise RuntimeError(
                    "Detected torchrun environment variables, but distributed.enabled is false in the trainer "
                    "config. Set distributed.enabled: true for multi-GPU data-parallel training."
                )
            return 0, 1, 0, False
        required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
        if not all(name in os.environ for name in required):
            raise RuntimeError(
                "distributed.enabled is true, but torchrun environment variables are missing. Launch with "
                "`torchrun --nproc_per_node=... train.py --config ...`."
            )
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if not torch.cuda.is_available():
            raise RuntimeError(
                "distributed.enabled is true, but CUDA is unavailable in this environment. Check the active "
                "PyTorch build and NVIDIA driver compatibility."
            )
        visible_device_count = torch.cuda.device_count()
        if local_rank >= visible_device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is out of range for the current CUDA visibility "
                f"({visible_device_count} visible device(s))."
            )
        torch.cuda.set_device(local_rank)
        try:
            torch.cuda.init()
            _ = torch.cuda.get_device_name(local_rank)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize CUDA on local rank {local_rank}. "
                "Try a different pair of GPUs or verify the active CUDA environment."
            ) from exc
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.distributed.backend,
                init_method="env://",
                timeout=datetime.timedelta(minutes=int(self.config.distributed.timeout_minutes)),
            )
        return rank, world_size, local_rank, True

    def _cleanup_distributed(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def _distributed_barrier(self) -> None:
        """Synchronize all ranks when distributed training is active."""
        if self.is_distributed and dist.is_available() and dist.is_initialized():
            if self.loss_device.type == "cuda":
                dist.barrier(device_ids=[self.local_rank])
            else:
                dist.barrier()

    def _configure_runtime(self) -> None:
        """Apply low-level backend runtime flags."""
        torch.backends.cudnn.enabled = self.config.runtime.cudnn_enabled
        torch.backends.cudnn.benchmark = (
            self.config.runtime.cudnn_enabled and self.config.runtime.cudnn_benchmark
        )
        torch.backends.cudnn.deterministic = self.config.runtime.cudnn_deterministic

    @staticmethod
    def _is_cudnn_initialization_error(error: RuntimeError) -> bool:
        message = str(error)
        return "CUDNN_STATUS_NOT_INITIALIZED" in message

    def _disable_cudnn_fallback(self) -> None:
        """Disable cuDNN after an initialization failure and continue with native CUDA kernels."""
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        self._cudnn_fallback_applied = True

    def _run_with_cudnn_fallback(self, step_fn: Any, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Retry a step once with cuDNN disabled if cuDNN initialization fails."""
        try:
            return step_fn(*args, **kwargs)
        except RuntimeError as error:
            if (
                not self.config.runtime.allow_cudnn_fallback_on_init_error
                or self._cudnn_fallback_applied
                or not self._is_cudnn_initialization_error(error)
            ):
                raise
            self._disable_cudnn_fallback()
            return step_fn(*args, **kwargs)

    def _build_sampler(
        self,
        data_config: XRayDataConfig | MRIDataConfig,
        train: bool,
    ) -> Sampler[int] | None:
        if not self.is_distributed:
            return None
        if isinstance(data_config, XRayDataConfig):
            dataset = build_xray_dataloader(data_config).dataset
        else:
            dataset = build_mri_dataloader(data_config).dataset
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=train,
            drop_last=data_config.drop_last if train else False,
        )

    def _estimate_total_training_steps(self) -> int:
        """Estimate total optimizer updates across the full staged schedule."""
        warm_start_steps = self.config.schedule.warm_start_epochs * (
            len(self.xray_train_loader) + len(self.mri_train_loader)
        )
        joint_loader_x = self.xray_joint_train_loader or self.xray_train_loader
        joint_loader_m = self.mri_joint_train_loader or self.mri_train_loader
        joint_steps = self.config.schedule.joint_epochs * max(len(joint_loader_x), len(joint_loader_m))
        return warm_start_steps + joint_steps

    def _build_joint_family_train_loaders(self) -> None:
        """Build joint-stage loaders that share the same anatomy-family subset schedule."""
        if self.config.schedule.joint_epochs > 0 and not (
            self.config.xray_data.shared_family_batching and self.config.mri_data.shared_family_batching
        ):
            warnings.warn(
                "Joint stage is enabled, but shared_family_batching is disabled for one or both modalities. "
                "The cross-modal losses compare X-rays and MRIs of the same anatomy family, so they need "
                "anatomy-aligned joint batches.",
                RuntimeWarning,
                stacklevel=2,
            )
        use_shared_family_batching = bool(
            self.config.xray_data.shared_family_batching or self.config.mri_data.shared_family_batching
        )
        if not use_shared_family_batching:
            return
        if not (self.config.xray_data.shared_family_batching and self.config.mri_data.shared_family_batching):
            raise ValueError(
                "shared_family_batching must be enabled for both xray_data and mri_data to synchronize "
                "anatomy families across modalities."
            )
        if self.config.xray_data.families_per_batch != self.config.mri_data.families_per_batch:
            raise ValueError(
                "xray_data.families_per_batch and mri_data.families_per_batch must match when "
                "shared_family_batching is enabled."
            )

        xray_dataset = self.xray_train_loader.dataset
        mri_dataset = self.mri_train_loader.dataset
        shared_families = sorted(set(xray_dataset.anatomy_families) & set(mri_dataset.anatomy_families))
        if not shared_families:
            raise RuntimeError(
                "shared_family_batching is enabled, but X-ray and MRI train splits do not share any anatomy families."
            )

        num_joint_batches = max(len(self.xray_train_loader), len(self.mri_train_loader))
        shared_seed = int(self.config.seed)
        self.xray_joint_train_sampler = SharedFamilyBatchSampler(
            anatomy_families=xray_dataset.anatomy_families,
            batch_size=self.config.xray_data.batch_size,
            num_batches=num_joint_batches,
            families_per_batch=self.config.xray_data.families_per_batch,
            shared_families=shared_families,
            shuffle=self.config.xray_data.shuffle,
            seed=shared_seed,
            rank=self.rank if self.is_distributed else 0,
            world_size=self.world_size if self.is_distributed else 1,
        )
        self.mri_joint_train_sampler = SharedFamilyBatchSampler(
            anatomy_families=mri_dataset.anatomy_families,
            batch_size=self.config.mri_data.batch_size,
            num_batches=num_joint_batches,
            families_per_batch=self.config.xray_data.families_per_batch,
            shared_families=shared_families,
            shuffle=self.config.mri_data.shuffle,
            seed=shared_seed,
            rank=self.rank if self.is_distributed else 0,
            world_size=self.world_size if self.is_distributed else 1,
        )
        self.xray_joint_train_loader = build_xray_dataloader(
            self.config.xray_data,
            batch_sampler=self.xray_joint_train_sampler,
        )
        self.mri_joint_train_loader = build_mri_dataloader(
            self.config.mri_data,
            batch_sampler=self.mri_joint_train_sampler,
        )

    def _collect_training_anatomy_families(self) -> tuple[str, ...]:
        """Collect canonical anatomy labels observed in the current train loaders."""
        labels = tuple(
            sorted(set(self.xray_train_loader.dataset.anatomy_families) | set(self.mri_train_loader.dataset.anatomy_families))
        )
        if labels:
            return labels
        return (UNKNOWN_ANATOMY_LABEL,)

    def _sync_anatomy_families_from_training_data(self) -> None:
        """Create one prototype bank per anatomy family seen in the training data (the assigners' queues follow the banks)."""
        if self.config.ablations.prototype_scope == "global":
            families = tuple(dict.fromkeys(self.config.prototype_banks.families))
            if not families:
                families = (UNKNOWN_ANATOMY_LABEL,)
        else:
            families = self._collect_training_anatomy_families()
        self.config.prototype_banks.families = families
        if isinstance(self.config.prototype_banks.prototypes_per_family, dict):
            configured_counts = self.config.prototype_banks.prototypes_per_family
            default_count = int(next(iter(configured_counts.values()))) if configured_counts else 64
            self.config.prototype_banks.prototypes_per_family = {
                family: int(configured_counts.get(family, default_count))
                for family in families
            }

    @staticmethod
    def _apply_ablations(config: TrainerConfig) -> TrainerConfig:
        ablations = config.ablations

        if ablations.prototype_scope not in {"anatomy_specific", "global"}:
            raise ValueError("ablations.prototype_scope must be 'anatomy_specific' or 'global'.")

        if not ablations.enable_mri:
            # X-ray-only pretraining: there is no MRI batch to pair with, so the joint stage and
            # every cross-modal loss are switched off here, whatever the config says.
            config.schedule.joint_epochs = 0
            config.lambda_centroid_alignment = 0.0
            config.schedule.cross_modal_ramp_epochs = 0

        if ablations.prototype_scope == "global":
            prototype_count = (
                config.prototype_banks.prototypes_per_family
                if isinstance(config.prototype_banks.prototypes_per_family, int)
                else next(iter(config.prototype_banks.prototypes_per_family.values()))
            )
            config.prototype_banks.families = (UNKNOWN_ANATOMY_LABEL,)
            config.prototype_banks.prototypes_per_family = int(prototype_count)

        return config

    @staticmethod
    def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
        """Create a differentiable zero scalar on the same device/dtype as a reference tensor."""
        return reference.sum() * 0.0

    @staticmethod
    def _ensure_2d_embedding(name: str, embedding: torch.Tensor) -> None:
        """Validate embedding tensor shape for downstream representation losses."""
        if embedding.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, dim], got {tuple(embedding.shape)}.")

    @staticmethod
    def _build_metadata_label_to_id(
        loader: DataLoader[dict[str, Any]] | None,
        field_name: str,
    ) -> dict[str, int]:
        """Build a deterministic vocabulary from non-empty metadata labels in a loader dataset."""
        if loader is None:
            return {}
        dataset = getattr(loader, "dataset", None)
        records = getattr(dataset, "records", None)
        if records is None:
            return {}
        labels: set[str] = set()
        for record in records:
            label = normalize_metadata_label(record.get(field_name, ""))
            if label:
                labels.add(label)
        return {label: index for index, label in enumerate(sorted(labels))}

    def _encode_metadata_labels(self, labels: list[Any], mapping: dict[str, int]) -> torch.Tensor:
        """Encode metadata labels (X-ray views) as integer ids, -1 for unknown."""
        return encode_categorical_labels(labels=labels, mapping=mapping, device=self.loss_device)

    def _assert_joint_regularizer_shapes(self) -> None:
        """Validate regularizer-related configuration choices before training starts."""
        for field_name in (
            "lambda_xray_view",
            "lambda_projection_attention_diversity",
        ):
            value = float(getattr(self.config, field_name))
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative, got {value}.")
        if float(self.config.lambda_projection_mil_contrastive) < 0.0:
            raise ValueError(
                "lambda_projection_mil_contrastive must be non-negative, "
                f"got {self.config.lambda_projection_mil_contrastive}."
            )
        if float(self.config.projection_mil_temperature) <= 0.0:
            raise ValueError(
                "projection_mil_temperature must be > 0.0, "
                f"got {self.config.projection_mil_temperature}."
            )
        if float(self.config.xray_swav_loss.temperature) <= 0.0:
            raise ValueError(
                "xray_swav_loss.temperature must be > 0.0, "
                f"got {self.config.xray_swav_loss.temperature}."
            )

    def _validate_embedding_dimension_compatibility(self) -> None:
        """X-ray, MRI, MRI-projection and prototype embeddings must all live in one space."""
        dims = {
            "xray_model.embedding_dim": int(self.config.xray_model.embedding_dim),
            "mri_model.embedding_dim": int(self.config.mri_model.embedding_dim),
            "prototype_banks.embedding_dim": int(self.config.prototype_banks.embedding_dim),
        }
        if self.config.mri_model.use_mri_projection_bridge:
            dims["mri_model.mri_projection_embedding_dim"] = int(self.config.mri_model.mri_projection_embedding_dim)
        if len(set(dims.values())) != 1:
            raise ValueError(f"All embedding dimensions must be equal, got {dims}.")

    def _iter_stage(self, epoch_index: int) -> tuple[str, float]:
        schedule = self.config.schedule
        if epoch_index < schedule.warm_start_epochs:
            return "warm_start", 0.0
        joint_offset = epoch_index - schedule.warm_start_epochs
        if joint_offset < schedule.joint_epochs:
            if schedule.cross_modal_ramp_epochs <= 1:
                weight = 1.0
            else:
                # First joint epoch: zero; final ramp epoch: full weight.
                weight = min(1.0, float(joint_offset) / float(schedule.cross_modal_ramp_epochs - 1))
            return "joint", weight
        return "done", 0.0

    def _set_lr(self, epoch_index: int) -> float:
        total_epochs = self.config.schedule.warm_start_epochs + self.config.schedule.joint_epochs
        lr = cosine_with_warmup(
            epoch=epoch_index,
            total_epochs=total_epochs,
            base_lr=self.config.optimizer.lr,
            warmup_epochs=self.config.optimizer.warmup_epochs,
            min_lr=self.config.optimizer.lr * self.config.optimizer.min_lr_ratio,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    @staticmethod
    def _split_encoder_output(output: EncoderOutput, batch_size: int, parts: int) -> list[EncoderOutput]:
        """Split a batched encoder output back into per-view chunks."""
        split_kwargs = {
            "feature_map": list(output.feature_map.split(batch_size, dim=0)),
            "pooled_feature": list(output.pooled_feature.split(batch_size, dim=0)),
            "embedding": list(output.embedding.split(batch_size, dim=0)),
            "projection": list(output.projection.split(batch_size, dim=0)),
        }
        intermediate_splits = {
            name: list(feature.split(batch_size, dim=0))
            for name, feature in output.intermediate_feature_maps.items()
        }
        mri_proj_feats_splits = (
            list(output.mri_proj_feats.split(batch_size, dim=0))
            if output.mri_proj_feats is not None
            else None
        )
        mri_proj_embs_splits = (
            list(output.mri_proj_embs.split(batch_size, dim=0))
            if output.mri_proj_embs is not None
            else None
        )
        mri_depth_attn_splits = (
            list(output.mri_depth_attn.split(batch_size, dim=0))
            if output.mri_depth_attn is not None
            else None
        )
        outputs: list[EncoderOutput] = []
        for index in range(parts):
            outputs.append(
                EncoderOutput(
                    feature_map=split_kwargs["feature_map"][index],
                    pooled_feature=split_kwargs["pooled_feature"][index],
                    embedding=split_kwargs["embedding"][index],
                    intermediate_feature_maps={
                        name: tensors[index] for name, tensors in intermediate_splits.items()
                    },
                    projection=split_kwargs["projection"][index],
                    mri_proj_feats=(
                        mri_proj_feats_splits[index]
                        if mri_proj_feats_splits is not None
                        else None
                    ),
                    mri_proj_embs=(
                        mri_proj_embs_splits[index]
                        if mri_proj_embs_splits is not None
                        else None
                    ),
                    mri_depth_attn=(
                        mri_depth_attn_splits[index]
                        if mri_depth_attn_splits is not None
                        else None
                    ),
                )
            )
        return outputs

    def _stack_views_on_device(self, views: list[torch.Tensor], indices: list[int], device: str) -> torch.Tensor:
        """Move each (pinned) view to device first, then concatenate on-device.

        `torch.cat` on CPU tensors allocates its output through the ordinary
        pageable allocator even when every input is pinned, which silently
        downgrades the subsequent `.to(device, non_blocking=True)` inside the
        encoder to a blocking copy. Concatenating after the H2D transfer keeps
        every transfer genuinely async. A single-tensor group is returned
        directly instead of paying for a no-op `cat`.
        """
        if len(indices) == 1:
            return views[indices[0]].to(device, non_blocking=True)
        return torch.cat([views[index].to(device, non_blocking=True) for index in indices], dim=0)

    def _encode_xray_views(self, batch: dict[str, Any]) -> list[EncoderOutput]:
        views = batch["global_views"] + batch["local_views"]
        grouped_indices: dict[tuple[int, ...], list[int]] = {}
        for index, crop in enumerate(views):
            grouped_indices.setdefault(tuple(crop.shape[1:]), []).append(index)

        outputs: list[EncoderOutput | None] = [None] * len(views)
        for indices in grouped_indices.values():
            stacked = self._stack_views_on_device(views, indices, self.model_system.xray_device)
            encoded = self.model_system(xray=stacked, move_to_output_device=True)["xray"]
            split_outputs = self._split_encoder_output(
                encoded,
                batch_size=views[indices[0]].shape[0],
                parts=len(indices),
            )
            for output_index, split_output in zip(indices, split_outputs):
                outputs[output_index] = split_output
        return [output for output in outputs if output is not None]

    def _encode_mri_views(
        self,
        batch: dict[str, Any],
        compute_projection_bridge: bool = True,
    ) -> list[EncoderOutput]:
        views = batch["global_views"] + batch["local_views"]
        # Crop 0 is kept separate when the bridge is active: its projection
        # head's BatchNorm must see only this global crop, not other views.
        grouped_indices: dict[tuple[tuple[int, ...], bool], list[int]] = {}
        for index, crop in enumerate(views):
            key = (tuple(crop.shape[1:]), compute_projection_bridge and index == 0)
            grouped_indices.setdefault(key, []).append(index)

        outputs: list[EncoderOutput | None] = [None] * len(views)
        # The final DDP forward must include the bridge when it is active so
        # unused-parameter discovery sees the complete joint-training graph.
        for (_, group_computes_bridge), indices in sorted(
            grouped_indices.items(), key=lambda item: item[0][1]
        ):
            stacked = self._stack_views_on_device(views, indices, self.model_system.mri_device)
            encoded = self.model_system(
                mri=stacked,
                move_to_output_device=True,
                compute_projection_bridge=group_computes_bridge,
            )["mri"]
            split_outputs = self._split_encoder_output(
                encoded,
                batch_size=views[indices[0]].shape[0],
                parts=len(indices),
            )
            for output_index, split_output in zip(indices, split_outputs):
                outputs[output_index] = split_output
        return [output for output in outputs if output is not None]

    def _centroid_alignment_loss(
        self,
        xray_outputs: dict[str, Any],
        mri_outputs: dict[str, Any],
        ramp_weight: float,
    ) -> torch.Tensor:
        """Weighted, ramped centroid alignment between the X-ray and MRI crop-0 embeddings."""
        weight = float(self.config.lambda_centroid_alignment)
        if weight <= 0.0:
            return xray_outputs["xray_projection"].new_tensor(0.0)
        loss = centroid_alignment_loss(
            xray_embeddings=xray_outputs["xray_projection"],
            xray_assignments=xray_outputs["xray_assignment"],
            mri_embeddings=mri_outputs["mri_projection"],
            mri_assignments=mri_outputs["mri_assignment"],
            config=self.config.centroid_alignment,
        )
        return ramp_weight * weight * loss

    def _projection_mil_loss(
        self,
        xray_batch: dict[str, Any],
        mri_batch: dict[str, Any],
        xray_outputs: dict[str, Any],
        mri_outputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Projection-level MIL contrastive loss between X-rays and MRI projection bags."""
        zero = self._zero_loss(xray_outputs["xray_loss"])
        zero_stat = zero.detach()
        zero_stats = {
            "valid_xray_anchors": zero_stat,
            "positive_fraction": zero_stat,
        }
        if not self._projection_joint_path_context:
            return zero, zero_stats
        if float(self.config.lambda_projection_mil_contrastive) <= 0.0:
            return zero, zero_stats
        if not self.config.mri_model.use_mri_projection_bridge:
            return zero, zero_stats
        z_m_proj = mri_outputs.get("mri_proj_embs")
        if z_m_proj is None:
            return zero, zero_stats
        return projection_mil_contrastive_loss(
            z_x=xray_outputs["xray_projection"],
            z_m_proj=z_m_proj,
            xray_families=list(xray_batch.get("anatomy_family", [])),
            mri_families=list(mri_batch.get("anatomy_family", [])),
            temperature=float(self.config.projection_mil_temperature),
        )

    def _xray_view_loss(self, embedding: torch.Tensor, view_labels: list[Any]) -> torch.Tensor:
        """X-ray view/plane prediction from the X-ray embedding."""
        if self.config.lambda_xray_view <= 0.0 or len(view_labels) != int(embedding.shape[0]):
            return self._zero_loss(embedding)
        self._ensure_2d_embedding("xray_embedding", embedding)
        return self.xray_view_head(embedding, self._encode_metadata_labels(view_labels, self.xray_view_label_to_id))

    def _xray_family_mil_loss(self, xray_embedding: torch.Tensor, anatomy_family: list[str]) -> torch.Tensor:
        """Control C1: projection MIL with same-family X-rays in place of MRI projection bags."""
        if self.config.lambda_xray_family_mil <= 0.0:
            return self._zero_loss(xray_embedding)
        return xray_family_mil_loss(
            z_x=xray_embedding,
            xray_families=[str(family) for family in anatomy_family],
            temperature=float(self.config.projection_mil_temperature),
        )

    @staticmethod
    def _assignment_entropy(assignments: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Compute mean entropy over valid prototype assignments."""
        max_prototypes = assignments.shape[1]
        mask = (
            torch.arange(max_prototypes, device=assignments.device)
            .unsqueeze(0)
            .expand(assignments.shape[0], -1)
            < counts.unsqueeze(1)
        )
        probs = (assignments * mask.to(assignments.dtype)).clamp_min(1e-12)
        entropy = -(probs * probs.log()).sum(dim=-1)
        return entropy.mean()

    @staticmethod
    def _prototype_usage(assignments: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Estimate normalized prototype usage across a batch."""
        max_prototypes = assignments.shape[1]
        mask = (
            torch.arange(max_prototypes, device=assignments.device)
            .unsqueeze(0)
            .expand(assignments.shape[0], -1)
            < counts.unsqueeze(1)
        )
        masked = assignments * mask.to(assignments.dtype)
        usage = masked.sum(dim=0)
        usage = usage / usage.sum().clamp_min(1e-12)
        active = (usage > 1e-6).sum().to(assignments.dtype)
        denom = torch.tensor(float(max(1, max_prototypes)), device=assignments.device, dtype=assignments.dtype)
        return active / denom

    @staticmethod
    def _prototype_occupancy(assignments: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Compute normalized prototype occupancy across a batch."""
        max_prototypes = assignments.shape[1]
        mask = (
            torch.arange(max_prototypes, device=assignments.device)
            .unsqueeze(0)
            .expand(assignments.shape[0], -1)
            < counts.unsqueeze(1)
        )
        masked = assignments * mask.to(assignments.dtype)
        occupancy = masked.sum(dim=0)
        return occupancy / occupancy.sum().clamp_min(1e-12)

    @staticmethod
    def _collapsed_prototype_alert(
        occupancy: torch.Tensor,
        usage: torch.Tensor,
        usage_threshold: float = 0.2,
        dominance_threshold: float = 0.85,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return binary collapse alerts based on low usage or dominant occupancy."""
        low_usage = (usage < usage_threshold).to(occupancy.dtype)
        dominant_proto = (occupancy.max() > dominance_threshold).to(occupancy.dtype)
        return low_usage, dominant_proto

    @staticmethod
    def _embedding_norm(embedding: torch.Tensor) -> torch.Tensor:
        """Compute the mean L2 norm of an embedding batch."""
        return embedding.norm(dim=-1).mean()

    def _log_iteration(self, payload: dict[str, Any]) -> None:
        """Log a single iteration payload to WandB."""
        self._log({"trainer/log_scope": "iteration"} | payload)

    def _log_epoch_summary(self, payload: dict[str, Any]) -> None:
        """Log epoch-aggregated metrics to WandB."""
        self._log({"trainer/log_scope": "epoch"} | payload)

    def _build_minimal_step_log(
        self,
        *,
        stage: str,
        lr: float,
        total_loss: float,
        xray_loss: float | None = None,
        mri_loss: float | None = None,
        projection_mil_loss: float | None = None,
        projection_mil_positive_fraction: float | None = None,
        centroid_loss: float | None = None,
        xray_assignment_entropy: float | None = None,
        mri_assignment_entropy: float | None = None,
    ) -> dict[str, float | str]:
        payload: dict[str, float | str] = {
            "train/stage": stage,
            "train/lr": float(lr),
            "train/total_loss": float(total_loss),
        }
        if xray_loss is not None:
            payload["train/xray_loss"] = float(xray_loss)
        if mri_loss is not None:
            payload["train/mri_loss"] = float(mri_loss)
        if self.config.lambda_centroid_alignment > 0.0 and centroid_loss is not None:
            payload["train/centroid_loss"] = float(centroid_loss)
        if xray_assignment_entropy is not None:
            payload["train/xray_assignment_entropy"] = float(xray_assignment_entropy)
        if mri_assignment_entropy is not None:
            payload["train/mri_assignment_entropy"] = float(mri_assignment_entropy)
        if projection_mil_loss is not None:
            payload["train/projection_mil_loss"] = float(projection_mil_loss)
        if projection_mil_positive_fraction is not None:
            payload["train/projection_mil_positive_fraction"] = float(projection_mil_positive_fraction)
        return payload

    def _build_epoch_log(self, record: dict[str, Any]) -> dict[str, Any]:
        keys: list[str] = [
            "epoch",
            "train/total_loss",
            "train/xray_loss",
            "train/xray_swav_loss",
            "train/mri_loss",
            "train/xray_assignment_entropy",
            "train/mri_assignment_entropy",
            "train/lr",
            "val/total_loss",
            "val/xray_swav_loss",
        ]
        if self.config.lambda_xray_view > 0.0:
            keys.extend(["train/xray_view_loss", "val/xray_view_loss"])
        if self.config.lambda_centroid_alignment > 0.0:
            keys.extend(["train/centroid_loss", "val/centroid_loss"])
        if float(self.config.lambda_projection_mil_contrastive) > 0.0:
            keys.extend(
                [
                    "train/projection_mil_loss",
                    "train/projection_mil_valid_xray_anchors",
                    "train/projection_mil_positive_fraction",
                    "val/projection_mil_loss",
                    "val/projection_mil_valid_xray_anchors",
                    "val/projection_mil_positive_fraction",
                ]
            )
        if self.config.log_bridge_gradient_audit:
            keys.extend(BRIDGE_GRADIENT_AUDIT_KEYS)
        return {k: record[k] for k in keys if k in record}

    def _print_epoch_gradient_audit(self, epoch: int, record: dict[str, Any]) -> None:
        if not self.config.log_bridge_gradient_audit:
            return
        audit_parts = [
            f"{metric_name}={float(record[metric_name]):.3e}"
            for metric_name in BRIDGE_GRADIENT_AUDIT_KEYS
            if metric_name in record
        ]
        if not audit_parts:
            return
        print(
            f"{_rank_print_label()} epoch={epoch + 1} gradient_audit " + " ".join(audit_parts),
            flush=True,
        )

    @staticmethod
    def _all_reduce_mean_scalar(value: float, device: torch.device) -> float:
        """Average a scalar across ranks when distributed training is active."""
        if not dist.is_available() or not dist.is_initialized():
            return value
        tensor = torch.tensor([value], device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
        return float(tensor.item())

    def _run_xray_step(
        self,
        batch: dict[str, Any],
        update_queue: bool = False,
        train: bool = True,
        view_loss_active: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self._run_with_cudnn_fallback(
            self._run_xray_step_impl,
            batch,
            update_queue=update_queue,
            train=train,
            view_loss_active=view_loss_active,
        )

    def _run_xray_step_impl(
        self,
        batch: dict[str, Any],
        update_queue: bool = False,
        train: bool = True,
        view_loss_active: bool = False,
    ) -> dict[str, torch.Tensor]:
        """X-ray SwAV loss (+ view prediction when `view_loss_active`, i.e. warm-up only)."""
        device_type = _batch_to_cpu_device_type(self.loss_device)
        with torch.amp.autocast(device_type=device_type, enabled=self.scaler.is_enabled()):
            encoded_views = self._encode_xray_views(batch)
            if not encoded_views:
                raise RuntimeError("X-ray step received no encoded views.")
            projections = [output.embedding for output in encoded_views]
            output = swav_loss(
                projections=projections,
                anatomy_family=batch["anatomy_family"],
                assigner=self.xray_assigner,
                config=self.config.xray_swav_loss,
                update_queue=update_queue,
                advance_assigner_step=train,
            )
            embedding = encoded_views[0].embedding
            self._ensure_2d_embedding("xray_embedding", embedding)
            if view_loss_active:
                view_loss = self._xray_view_loss(embedding, view_labels=batch.get("view_label", []))
            else:
                view_loss = self._zero_loss(embedding)
            family_mil_loss = self._xray_family_mil_loss(
                xray_embedding=projections[0],
                anatomy_family=list(batch["anatomy_family"]),
            )
            total_xray_loss = (
                output.loss
                + self.config.lambda_xray_view * view_loss
                + self.config.lambda_xray_family_mil * family_mil_loss
            )
        occupancy = self._prototype_occupancy(
            output.assignment_outputs[0].family_assignments,
            output.assignment_outputs[0].family_prototype_counts,
        )
        usage = self._prototype_usage(
            output.assignment_outputs[0].family_assignments,
            output.assignment_outputs[0].family_prototype_counts,
        )
        low_usage_alert, dominant_proto_alert = self._collapsed_prototype_alert(occupancy, usage)
        return {
            "xray_loss": total_xray_loss,
            "xray_swav_loss": output.loss,
            "xray_view_loss": view_loss,
            "xray_family_mil_loss": family_mil_loss,
            "xray_assignment": output.assignment_outputs[0],
            "xray_projection": projections[0],
            "xray_assignment_entropy": self._assignment_entropy(
                output.assignment_outputs[0].family_assignments,
                output.assignment_outputs[0].family_prototype_counts,
            ),
            "xray_prototype_usage": usage,
            "xray_low_usage_alert": low_usage_alert,
            "xray_dominant_proto_alert": dominant_proto_alert,
            "xray_embedding_norm": self._embedding_norm(projections[0]),
        }

    def _run_mri_step(
        self,
        batch: dict[str, Any],
        update_queue: bool = False,
        train: bool = True,
        projection_joint_path: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self._run_with_cudnn_fallback(
            self._run_mri_step_impl,
            batch,
            update_queue=update_queue,
            train=train,
            projection_joint_path=projection_joint_path,
        )

    def _run_mri_step_impl(
        self,
        batch: dict[str, Any],
        update_queue: bool = False,
        train: bool = True,
        projection_joint_path: bool = False,
    ) -> dict[str, torch.Tensor]:
        self._projection_joint_path_context = bool(projection_joint_path)
        device_type = _batch_to_cpu_device_type(self.loss_device)
        with torch.amp.autocast(device_type=device_type, enabled=self.scaler.is_enabled()):
            encoded_views = self._encode_mri_views(
                batch,
                compute_projection_bridge=(
                    projection_joint_path and self.config.mri_model.use_mri_projection_bridge
                ),
            )
            if not encoded_views:
                raise RuntimeError("MRI step received no encoded views.")
            projections = [output.embedding for output in encoded_views]
            output = swav_loss(
                projections=projections,
                anatomy_family=batch["anatomy_family"],
                assigner=self.mri_assigner,
                config=self.config.mri_swav_loss,
                update_queue=update_queue,
                advance_assigner_step=train,
            )
            projected_embeddings = encoded_views[0].mri_proj_embs
            embedding = encoded_views[0].embedding
            self._ensure_2d_embedding("mri_embedding", embedding)
            mri_depth_attn = encoded_views[0].mri_depth_attn
            if mri_depth_attn is not None and self.config.lambda_projection_attention_diversity > 0.0:
                projection_attention_reg = projection_attention_regularization(mri_depth_attn)
                projection_attention_diversity_loss = projection_attention_reg["diversity_loss"]
                projection_attention_pairwise_cosine = projection_attention_reg["pairwise_cosine"]
            else:
                projection_attention_diversity_loss = self._zero_loss(embedding)
                projection_attention_pairwise_cosine = self._zero_loss(embedding)
            total_mri_loss = (
                output.loss
                + self.config.lambda_projection_attention_diversity * projection_attention_diversity_loss
            )
            if projected_embeddings is not None:
                # Keep projection-bridge parameters graph-connected when no projection loss term is active.
                total_mri_loss = total_mri_loss + projected_embeddings.sum() * 0.0
        occupancy = self._prototype_occupancy(
            output.assignment_outputs[0].family_assignments,
            output.assignment_outputs[0].family_prototype_counts,
        )
        usage = self._prototype_usage(
            output.assignment_outputs[0].family_assignments,
            output.assignment_outputs[0].family_prototype_counts,
        )
        low_usage_alert, dominant_proto_alert = self._collapsed_prototype_alert(occupancy, usage)
        return {
            "mri_loss": total_mri_loss,
            "mri_swav_loss": output.loss,
            "projection_attention_diversity_loss": projection_attention_diversity_loss,
            "projection_attention_pairwise_cosine": projection_attention_pairwise_cosine,
            "mri_assignment": output.assignment_outputs[0],
            "mri_projection": projections[0],
            "mri_proj_feats": encoded_views[0].mri_proj_feats,
            "mri_proj_embs": projected_embeddings,
            "mri_depth_attn": encoded_views[0].mri_depth_attn,
            "mri_assignment_entropy": self._assignment_entropy(
                output.assignment_outputs[0].family_assignments,
                output.assignment_outputs[0].family_prototype_counts,
            ),
            "mri_prototype_usage": usage,
            "mri_low_usage_alert": low_usage_alert,
            "mri_dominant_proto_alert": dominant_proto_alert,
            "mri_embedding_norm": self._embedding_norm(projections[0]),
            "mri_queue_size": output.assignment_outputs[0].queue_sizes.to(torch.float32).mean(),
        }

    def _backward_step(self, loss: torch.Tensor) -> None:
        self._last_gradient_audit = {}
        self._last_gradient_audit_step = None
        # A non-finite loss propagates to non-finite gradients through the chain
        # rule (no PyTorch op raises on it outside anomaly-detection mode), so the
        # post-backward _has_nonfinite_gradients check below already catches this
        # case without a device sync before the backward pass.
        params = tuple(
            parameter
            for parameter in itertools.chain(
                self.model_system.parameters(),
                self.prototype_banks.parameters(),
                self.xray_view_head.parameters(),
            )
            if parameter.requires_grad
        )

        self.optimizer.zero_grad(set_to_none=True)
        try:
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self._all_reduce_module_grads(self.prototype_banks_module)
                self._all_reduce_module_grads(self.xray_view_head)
                if self._has_nonfinite_gradients(params):
                    self._record_nonfinite_step("non-finite gradients after backward (AMP)")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update()
                    return
                self._capture_gradient_audit()
                if self.config.optimizer.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, self.config.optimizer.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self._all_reduce_module_grads(self.prototype_banks_module)
                self._all_reduce_module_grads(self.xray_view_head)
                if self._has_nonfinite_gradients(params):
                    self._record_nonfinite_step("non-finite gradients after backward")
                    self.optimizer.zero_grad(set_to_none=True)
                    return
                self._capture_gradient_audit()
                if self.config.optimizer.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, self.config.optimizer.grad_clip_norm)
                self.optimizer.step()
        except RuntimeError as error:
            if self._is_nonfinite_backward_error(error):
                self._record_nonfinite_step(f"backward raised non-finite error: {error}")
                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.update()
                return
            raise

    def _record_nonfinite_step(self, reason: str) -> None:
        self.state["skipped_nonfinite_steps"] = int(self.state.get("skipped_nonfinite_steps", 0)) + 1
        warnings.warn(
            f"{_rank_print_label()} skipped optimizer step due to {reason}. "
            f"global_step={self.state.get('global_step', 0)} "
            f"skipped_nonfinite_steps={self.state['skipped_nonfinite_steps']}",
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def _has_nonfinite_gradients(parameters: tuple[torch.nn.Parameter, ...]) -> bool:
        # torch._foreach_norm computes all per-tensor norms in one fused kernel launch,
        # so the whole check costs a single device sync.
        grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not grads:
            return False
        norms = torch.stack(torch._foreach_norm(grads))
        return not bool(torch.isfinite(norms).all().item())

    @staticmethod
    def _is_nonfinite_backward_error(error: RuntimeError) -> bool:
        message = str(error).lower()
        return ("nan" in message or "inf" in message) and "backward" in message

    @staticmethod
    def _unwrap_module(module: torch.nn.Module | None) -> torch.nn.Module | None:
        if module is None:
            return None
        if isinstance(module, DistributedDataParallel):
            return module.module
        return module

    @staticmethod
    def _combined_grad_norm(grads: list[torch.Tensor]) -> float:
        """L2 norm of the concatenation of several gradient tensors.

        torch._foreach_norm computes every per-tensor norm in one fused
        multi-tensor-apply kernel launch; combining them needs one more norm
        and a single final sync, instead of a Python loop over parameters that
        did one torch.sum(...).item() blocking sync per parameter (an audit
        step here walks every parameter of an entire encoder, ~1000+ syncs
        under the old implementation).
        """
        if not grads:
            return 0.0
        per_tensor_norms = torch.stack(torch._foreach_norm(grads))
        return float(torch.linalg.vector_norm(per_tensor_norms).item())

    def _module_grad_norm(self, module: torch.nn.Module | None) -> float:
        unwrapped_module = self._unwrap_module(module)
        if unwrapped_module is None:
            return 0.0
        grads = [parameter.grad for parameter in unwrapped_module.parameters() if parameter.grad is not None]
        return self._combined_grad_norm(grads)

    def _named_parameter_grad_norm(self, module: torch.nn.Module | None, substring: str) -> float:
        unwrapped_module = self._unwrap_module(module)
        if unwrapped_module is None:
            return 0.0
        grads = [
            parameter.grad
            for parameter_name, parameter in unwrapped_module.named_parameters()
            if substring in parameter_name and parameter.grad is not None
        ]
        return self._combined_grad_norm(grads)

    def _xray_backbone_grad_norm(self) -> float:
        xray_encoder = self._unwrap_module(self.model_system.xray_encoder)
        if xray_encoder is None:
            return 0.0
        if getattr(xray_encoder, "backbone_name", "").startswith("resnet"):
            return self._named_parameter_grad_norm(self.model_system.xray_encoder, "layer4")

        candidate_substrings = (
            "vit_backbone.model.encoder.layer.11",
            "vit_backbone.model.layers.11",
            "vit_backbone.model.layernorm",
        )
        for substring in candidate_substrings:
            grad_norm = self._named_parameter_grad_norm(self.model_system.xray_encoder, substring)
            if grad_norm > 0.0:
                return grad_norm
        return self._module_grad_norm(self.model_system.xray_encoder)

    def _capture_gradient_audit(self) -> None:
        if not self.config.log_bridge_gradient_audit:
            return
        step_index = int(self.state.get("global_step", 0)) + 1
        every_steps = max(1, int(self.config.bridge_gradient_audit_every_steps))
        if step_index % every_steps != 0:
            return
        mri_encoder = self._unwrap_module(self.model_system.mri_encoder)
        self._last_gradient_audit = {
            "grad_norm/xray_encoder_total": self._module_grad_norm(self.model_system.xray_encoder),
            "grad_norm/mri_encoder_total": self._module_grad_norm(self.model_system.mri_encoder),
            "grad_norm/mri_projection_bridge": self._module_grad_norm(
                getattr(mri_encoder, "mri_projection_bridge", None)
            ),
            "grad_norm/mri_projection_head": self._module_grad_norm(
                getattr(mri_encoder, "mri_projection_head", None)
            ),
            "grad_norm/mri_layer4": self._named_parameter_grad_norm(self.model_system.mri_encoder, "layer4"),
            "grad_norm/xray_layer4": self._xray_backbone_grad_norm(),
            "grad_norm/prototype_banks": self._module_grad_norm(self.prototype_banks),
            "grad_norm/xray_view_head": self._module_grad_norm(self.xray_view_head),
        }
        self._last_gradient_audit_step = step_index

    def _append_gradient_audit_metrics(self, metrics: dict[str, list[float]]) -> None:
        if not self.config.log_bridge_gradient_audit:
            return
        step_index = int(self.state.get("global_step", 0)) + 1
        if self._last_gradient_audit_step != step_index:
            return
        for metric_name in BRIDGE_GRADIENT_AUDIT_KEYS:
            metrics.setdefault(metric_name, []).append(float(self._last_gradient_audit.get(metric_name, 0.0)))

    def _all_reduce_module_grads(self, module: torch.nn.Module) -> None:
        """Synchronize gradients for a non-DDP module across ranks.

        Flattens every trainable parameter's gradient into one buffer and
        issues a single collective, instead of one dist.all_reduce per
        parameter (the prototype banks have one parameter per anatomy family,
        the X-ray view head a few -- each all_reduce call has a
        fixed per-call latency regardless of payload size, so ~20 tiny
        separate collectives/step were paying that fixed cost 20x for no
        benefit; this is exactly the pattern DDP's own gradient bucketing
        exists to avoid).
        """
        if not self.is_distributed:
            return
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if not parameters:
            return
        grads: list[torch.Tensor] = []
        for parameter in parameters:
            if parameter.grad is None:
                # Keep optimizer behavior consistent across ranks when a
                # parameter is unused locally this step (e.g. an anatomy
                # family with no samples in this rank's batch) -- contribute
                # an explicit zero so the all-reduce denominator (world_size)
                # still reflects every rank.
                parameter.grad = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            grads.append(parameter.grad)
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= float(world_size)
        for grad, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            grad.copy_(synced)

    def _log(self, payload: dict[str, Any]) -> None:
        self.wandb.log(payload, step=self.state["global_step"])

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format a duration in seconds into a compact human-readable string."""
        total_seconds = max(0, int(seconds))
        days, rem = divmod(total_seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        if days > 0:
            return f"{days}d{hours:02d}h{minutes:02d}m"
        if hours > 0:
            return f"{hours}h{minutes:02d}m"
        if minutes > 0:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    def _estimate_run_eta(self) -> tuple[float, float, float] | None:
        """Estimate remaining and total runtime from observed step throughput."""
        start_time = getattr(self, "train_start_time", None)
        start_global_step = getattr(self, "run_start_global_step", None)
        if start_time is None or start_global_step is None:
            return None
        elapsed = max(0.0, time.time() - float(start_time))
        current_global_step = int(self.state.get("global_step", 0))
        progressed_steps = max(0, current_global_step - int(start_global_step))
        if progressed_steps <= 0:
            return None
        sec_per_step = elapsed / float(progressed_steps)
        total_steps = max(1, int(self.total_training_steps_estimate))
        remaining_steps = max(0, total_steps - current_global_step)
        remaining_seconds = sec_per_step * float(remaining_steps)
        estimated_total_seconds = sec_per_step * float(total_steps)
        eta_unix_time = time.time() + remaining_seconds
        return remaining_seconds, estimated_total_seconds, eta_unix_time

    def _print_iteration_log(
        self,
        stage: str,
        step_index: int,
        num_steps: int,
        lr: float,
        epoch_start_time: float,
        total_loss: float,
        xray_loss: float | None = None,
        mri_loss: float | None = None,
    ) -> None:
        elapsed = max(0.0, time.time() - epoch_start_time)
        parts = [
            f"{_rank_print_label()}",
            f"stage={stage}",
            f"step={step_index}/{num_steps}",
            f"lr={lr:.3e}",
            f"total={total_loss:.4f}",
        ]
        if xray_loss is not None:
            parts.append(f"xray={xray_loss:.4f}")
        if mri_loss is not None:
            parts.append(f"mri={mri_loss:.4f}")
        parts.append(f"elapsed={elapsed/60.0:.1f}m")
        eta_info = self._estimate_run_eta()
        if eta_info is not None:
            remaining_seconds, estimated_total_seconds, eta_unix_time = eta_info
            parts.append(f"eta_remaining={self._format_duration(remaining_seconds)}")
            parts.append(f"eta_total={self._format_duration(estimated_total_seconds)}")
            parts.append(f"eta_end={time.strftime('%Y-%m-%d %H:%M', time.localtime(eta_unix_time))}")
        print(" ".join(parts), flush=True)

    def _save_checkpoint(self, epoch: int, tag: str = "latest", epoch_copy: bool = True) -> Path:
        if self.rank != 0:
            return Path(self.config.checkpoint.output_dir) / f"{tag}.pt"
        output_dir = Path(self.config.checkpoint.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{tag}.pt"
        payload = {
            "config": asdict(self.config),
            "state": self.state,
            "xray_encoder": self.xray_encoder_module.state_dict(),
            "mri_encoder": self.mri_encoder_module.state_dict(),
            "prototype_banks": self.prototype_banks_module.state_dict(),
            "xray_assigner": self.xray_assigner.state_dict(),
            "mri_assigner": self.mri_assigner.state_dict(),
            "xray_view_head": self.xray_view_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
        }
        torch.save(payload, path)
        if epoch_copy and not self.config.checkpoint.keep_last_only and self.config.checkpoint.save_every_epochs > 0:
            epoch_path = output_dir / f"epoch_{epoch:04d}.pt"
            torch.save(payload, epoch_path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume trainer state from a checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        self.xray_encoder_module.load_state_dict(checkpoint["xray_encoder"])
        self.mri_encoder_module.load_state_dict(checkpoint["mri_encoder"])
        self.prototype_banks_module.load_state_dict(checkpoint["prototype_banks"])
        self.xray_assigner.load_state_dict(checkpoint["xray_assigner"])
        self.mri_assigner.load_state_dict(checkpoint["mri_assigner"])
        self.xray_view_head.load_state_dict(checkpoint["xray_view_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        scaler_state = checkpoint.get("scaler")
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)
        self.state.update(checkpoint.get("state", {}))

    def _load_pretrained_mri_backbone(self) -> None:
        """Load a Models-Genesis-style backbone checkpoint into the MRI encoder, in place.

        Runs on every rank (not just rank 0) so each process' local weights are
        consistent before the DDP wrap and optimizer are constructed. Only
        conv1/bn1/layer1-4 keys are loaded; projectors, the projection bridge, and
        the projection head stay randomly initialized.
        """
        checkpoint_path = self.config.mri_model.pretrained_checkpoint
        if not checkpoint_path:
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_encoder_type = checkpoint.get("encoder_type", "volume_3d")
        if checkpoint_encoder_type != self.config.mri_model.encoder_type:
            raise ValueError(
                f"Pretrained MRI checkpoint '{checkpoint_path}' holds a {checkpoint_encoder_type!r} encoder, "
                f"but mri_model.encoder_type is {self.config.mri_model.encoder_type!r}."
            )
        checkpoint_schedule = checkpoint.get("stage_stride_schedule")
        if checkpoint_encoder_type == "volume_3d" and checkpoint_schedule != self.config.mri_model.stage_stride_schedule:
            raise ValueError(
                f"Pretrained MRI checkpoint '{checkpoint_path}' was trained with "
                f"stage_stride_schedule={checkpoint_schedule!r}, but mri_model.stage_stride_schedule="
                f"{self.config.mri_model.stage_stride_schedule!r}. These must match."
            )
        backbone_prefixes = tuple(MRI_BACKBONE_STAGE_PREFIXES.values())
        backbone_state = {
            key: value
            for key, value in checkpoint["mri_encoder_state_dict"].items()
            if key.startswith(backbone_prefixes)
        }
        result = self.mri_encoder_module.load_state_dict(backbone_state, strict=False)
        loaded = len(backbone_state) - len(result.unexpected_keys)
        if loaded == 0:
            raise RuntimeError(
                f"Loading pretrained MRI checkpoint '{checkpoint_path}' matched zero backbone keys "
                "on the MRI encoder; refusing to silently no-op. Check that the checkpoint's backbone "
                "architecture matches mri_model."
            )
        if self.rank == 0:
            print(
                f"[mri_pretrained_checkpoint] path={checkpoint_path} loaded={loaded} "
                f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}",
                flush=True,
            )

    def _set_trainability_for_stage(self, stage: str) -> None:
        trainable = stage in {"warm_start", "joint"}
        for module in (self.xray_encoder_module, self.mri_encoder_module, self.prototype_banks_module):
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)

    @staticmethod
    def _gradual_unfreeze_trainable_indices(
        *,
        epoch: int,
        total_units: int,
        start_epoch: int,
        epochs_per_stage: int,
        units_per_stage: int,
    ) -> set[int]:
        """Compute which of `total_units` ordered units (blocks/stages) are trainable.

        Shared math for the ViT-block and CNN-stage gradual-unfreeze schedules: `units`
        are indexed input->output (index 0 closest to input), and units become trainable
        `units_per_stage` at a time, every `epochs_per_stage` epochs, starting at
        `start_epoch`, from the output end (closest to the head) towards the input.
        """
        if epoch < start_epoch:
            num_trainable = 0
        else:
            stage_index = (epoch - start_epoch) // max(1, epochs_per_stage)
            num_trainable = min(total_units, (stage_index + 1) * units_per_stage)

        return set(range(total_units - num_trainable, total_units))

    def _apply_xray_vit_gradual_unfreeze(self, epoch: int, stage: str) -> None:
        """Narrow ViT block trainability within an already-active X-ray encoder.

        Runs after `_set_trainability_for_stage`, which governs the whole-encoder
        warm_start/joint on-off switch and the projector heads. This
        method only restricts which transformer blocks inside the ViT are trainable,
        so the pretrained backbone isn't immediately overwritten by joint fine-tuning.
        """
        schedule = self.config.schedule
        if not schedule.enable_xray_vit_gradual_unfreeze:
            return
        if self.xray_encoder_module.backbone_name != "dinov3_vitb16":
            return
        if stage not in {"warm_start", "joint"}:
            return

        vit_backbone = self.xray_encoder_module.vit_backbone
        blocks = get_transformer_blocks(vit_backbone)
        total_blocks = len(blocks)

        trainable_indices = self._gradual_unfreeze_trainable_indices(
            epoch=epoch,
            total_units=total_blocks,
            start_epoch=schedule.xray_vit_unfreeze_start_epoch,
            epochs_per_stage=schedule.xray_vit_unfreeze_epochs_per_stage,
            units_per_stage=schedule.xray_vit_unfreeze_blocks_per_stage,
        )

        for index, block in enumerate(blocks):
            block.requires_grad_(index in trainable_indices)

        embeddings, final_norm = get_embedding_and_norm_modules(vit_backbone)
        fully_unfrozen = len(trainable_indices) >= total_blocks
        if embeddings is not None:
            embeddings.requires_grad_(
                fully_unfrozen or not schedule.xray_vit_freeze_patch_embed_until_full
            )
        if final_norm is not None:
            final_norm.requires_grad_((total_blocks - 1) in trainable_indices)

    def _apply_xray_cnn_gradual_unfreeze(self, epoch: int, stage: str) -> None:
        """Narrow ResNet stage trainability within an already-active X-ray encoder.

        Same role as `_apply_xray_vit_gradual_unfreeze` but for resnet X-ray backbones:
        treats [stem, layer1, layer2, layer3, layer4] as 5 ordered units and unfreezes
        them progressively. The stem is index 0, so top-down order (the default) leaves
        it frozen until the backbone is fully unfrozen, without needing a separate
        "freeze embeddings until full" flag the way the ViT path does.
        """
        schedule = self.config.schedule
        if not schedule.enable_xray_cnn_gradual_unfreeze:
            return
        if not self.xray_encoder_module.backbone_name.startswith("resnet"):
            return
        if stage not in {"warm_start", "joint"}:
            return

        stages = get_resnet_stage_modules(self.xray_encoder_module)
        total_stages = len(stages)

        trainable_indices = self._gradual_unfreeze_trainable_indices(
            epoch=epoch,
            total_units=total_stages,
            start_epoch=schedule.xray_cnn_unfreeze_start_epoch,
            epochs_per_stage=schedule.xray_cnn_unfreeze_epochs_per_stage,
            units_per_stage=schedule.xray_cnn_unfreeze_stages_per_step,
        )

        for index, resnet_stage in enumerate(stages):
            resnet_stage.requires_grad_(index in trainable_indices)

    def _train_epoch(self, epoch: int, stage: str, cross_modal_weight: float, lr: float) -> dict[str, float]:
        self._projection_joint_path_context = (stage == "joint")
        self.model_system.train()
        self.prototype_banks.train()
        self._set_trainability_for_stage(stage)
        self._apply_xray_vit_gradual_unfreeze(epoch, stage)
        self._apply_xray_cnn_gradual_unfreeze(epoch, stage)
        epoch_start_time = time.time()

        metrics: dict[str, list[float]] = {
            "train/xray_loss": [],
            "train/xray_swav_loss": [],
            "train/mri_loss": [],
            "train/centroid_loss": [],
            "train/xray_view_loss": [],
            "train/total_loss": [],
            "train/xray_assignment_entropy": [],
            "train/mri_assignment_entropy": [],
            "train/xray_prototype_usage": [],
            "train/mri_prototype_usage": [],
            "train/xray_embedding_norm": [],
            "train/mri_embedding_norm": [],
            "train/mri_queue_size": [],
            "train/projection_attention_diversity_loss": [],
            "train/projection_attention_pairwise_cosine": [],
            "train/xray_low_usage_alert": [],
            "train/mri_low_usage_alert": [],
            "train/xray_dominant_proto_alert": [],
            "train/mri_dominant_proto_alert": [],
            "train/projection_mil_loss": [],
            "train/projection_mil_valid_xray_anchors": [],
            "train/projection_mil_positive_fraction": [],
        }
        if self.config.lambda_xray_family_mil > 0.0:
            metrics["train/xray_family_mil_loss"] = []
        if self.config.log_bridge_gradient_audit:
            metrics.update({metric_name: [] for metric_name in BRIDGE_GRADIENT_AUDIT_KEYS})

        if stage == "warm_start":
            loader_pairs = [("xray", self.xray_train_loader)]
            if self.config.ablations.enable_mri:
                loader_pairs.append(("mri", self.mri_train_loader))
            for loader_name, loader in loader_pairs:
                num_steps = len(loader)
                for step_index, batch in enumerate(loader, start=1):
                    if loader_name == "xray":
                        # Warm-up is the only stage with the X-ray view-prediction loss.
                        outputs = self._run_xray_step(batch, update_queue=False, train=True, view_loss_active=True)
                        total_loss = outputs["xray_loss"]
                        metrics["train/xray_loss"].append(float(total_loss.item()))
                        metrics["train/xray_swav_loss"].append(float(outputs["xray_swav_loss"].item()))
                        metrics["train/xray_view_loss"].append(float(outputs["xray_view_loss"].item()))
                        if "train/xray_family_mil_loss" in metrics:
                            metrics["train/xray_family_mil_loss"].append(
                                float(outputs["xray_family_mil_loss"].item())
                            )
                        metrics["train/xray_assignment_entropy"].append(float(outputs["xray_assignment_entropy"].item()))
                        metrics["train/xray_prototype_usage"].append(float(outputs["xray_prototype_usage"].item()))
                        metrics["train/xray_embedding_norm"].append(float(outputs["xray_embedding_norm"].item()))
                        metrics["train/xray_low_usage_alert"].append(float(outputs["xray_low_usage_alert"].item()))
                        metrics["train/xray_dominant_proto_alert"].append(float(outputs["xray_dominant_proto_alert"].item()))
                        metrics["train/total_loss"].append(float(total_loss.item()))
                    else:
                        outputs = self._run_mri_step(
                            batch,
                            update_queue=True,
                            train=True,
                            projection_joint_path=False,
                        )
                        total_loss = self.config.lambda_mri_loss * outputs["mri_loss"]
                        metrics["train/mri_loss"].append(float(outputs["mri_loss"].item()))
                        metrics["train/mri_assignment_entropy"].append(float(outputs["mri_assignment_entropy"].item()))
                        metrics["train/mri_prototype_usage"].append(float(outputs["mri_prototype_usage"].item()))
                        metrics["train/mri_embedding_norm"].append(float(outputs["mri_embedding_norm"].item()))
                        metrics["train/mri_queue_size"].append(float(outputs["mri_queue_size"].item()))
                        metrics["train/projection_attention_diversity_loss"].append(
                            float(outputs["projection_attention_diversity_loss"].item())
                        )
                        metrics["train/projection_attention_pairwise_cosine"].append(
                            float(outputs["projection_attention_pairwise_cosine"].item())
                        )
                        metrics["train/mri_low_usage_alert"].append(float(outputs["mri_low_usage_alert"].item()))
                        metrics["train/mri_dominant_proto_alert"].append(float(outputs["mri_dominant_proto_alert"].item()))
                        metrics["train/total_loss"].append(float(total_loss.item()))
                    self._backward_step(total_loss)
                    self._append_gradient_audit_metrics(metrics)
                    self.state["global_step"] += 1
                    if self.config.log_every_steps and (step_index % self.config.log_every_steps == 0):
                        self._print_iteration_log(
                            stage=f"{stage}/{loader_name}",
                            step_index=step_index,
                            num_steps=num_steps,
                            lr=lr,
                            epoch_start_time=epoch_start_time,
                            total_loss=float(total_loss.item()),
                            xray_loss=float(outputs["xray_loss"].item()) if loader_name == "xray" else None,
                            mri_loss=float(outputs["mri_loss"].item()) if loader_name == "mri" else None,
                        )
                        minimal_payload = self._build_minimal_step_log(
                            stage=stage,
                            lr=lr,
                            total_loss=float(total_loss.item()),
                            xray_loss=float(outputs["xray_loss"].item()) if loader_name == "xray" else None,
                            mri_loss=float(outputs["mri_loss"].item()) if loader_name == "mri" else None,
                            xray_assignment_entropy=(
                                float(outputs["xray_assignment_entropy"].item()) if loader_name == "xray" else None
                            ),
                            mri_assignment_entropy=(
                                float(outputs["mri_assignment_entropy"].item()) if loader_name == "mri" else None
                            ),
                        )
                        self._log_iteration(minimal_payload)
            result = {key: float(np.mean(value)) if value else 0.0 for key, value in metrics.items()}
            if self.is_distributed:
                result = {key: self._all_reduce_mean_scalar(val, self.loss_device) for key, val in result.items()}
            return result | {
                "train/stage": stage,
                "train/cross_modal_weight": cross_modal_weight,
                "train/lr": lr,
                "train/skipped_nonfinite_steps": float(self.state.get("skipped_nonfinite_steps", 0)),
            }

        if stage == "joint":
            joint_x_loader = self.xray_joint_train_loader or self.xray_train_loader
            joint_m_loader = self.mri_joint_train_loader or self.mri_train_loader
            x_iter = _cycle(joint_x_loader)
            m_iter = _cycle(joint_m_loader)
            num_steps = max(len(joint_x_loader), len(joint_m_loader))
            for step_index in range(num_steps):
                x_batch = next(x_iter)
                m_batch = next(m_iter)
                # No X-ray view-prediction loss after warm-up.
                x_outputs = self._run_xray_step(x_batch, update_queue=False, train=True, view_loss_active=False)
                m_outputs = self._run_mri_step(
                    m_batch,
                    update_queue=True,
                    train=True,
                    projection_joint_path=True,
                )
                centroid_loss = self._centroid_alignment_loss(x_outputs, m_outputs, ramp_weight=cross_modal_weight)
                projection_mil_raw_loss, projection_mil_stats = self._projection_mil_loss(
                    xray_batch=x_batch,
                    mri_batch=m_batch,
                    xray_outputs=x_outputs,
                    mri_outputs=m_outputs,
                )
                projection_mil_loss = float(self.config.lambda_projection_mil_contrastive) * projection_mil_raw_loss
                total_loss = (
                    x_outputs["xray_loss"]
                    + self.config.lambda_mri_loss * m_outputs["mri_loss"]
                    + centroid_loss
                    + projection_mil_loss
                )
                self._backward_step(total_loss)
                self._append_gradient_audit_metrics(metrics)
                self.state["global_step"] += 1
                metrics["train/xray_loss"].append(float(x_outputs["xray_loss"].item()))
                metrics["train/xray_swav_loss"].append(float(x_outputs["xray_swav_loss"].item()))
                metrics["train/mri_loss"].append(float(m_outputs["mri_loss"].item()))
                metrics["train/centroid_loss"].append(float(centroid_loss.item()))
                metrics["train/total_loss"].append(float(total_loss.item()))
                metrics["train/xray_assignment_entropy"].append(float(x_outputs["xray_assignment_entropy"].item()))
                metrics["train/mri_assignment_entropy"].append(float(m_outputs["mri_assignment_entropy"].item()))
                metrics["train/xray_prototype_usage"].append(float(x_outputs["xray_prototype_usage"].item()))
                metrics["train/mri_prototype_usage"].append(float(m_outputs["mri_prototype_usage"].item()))
                metrics["train/xray_embedding_norm"].append(float(x_outputs["xray_embedding_norm"].item()))
                metrics["train/mri_embedding_norm"].append(float(m_outputs["mri_embedding_norm"].item()))
                metrics["train/mri_queue_size"].append(float(m_outputs["mri_queue_size"].item()))
                metrics["train/projection_attention_diversity_loss"].append(
                    float(m_outputs["projection_attention_diversity_loss"].item())
                )
                metrics["train/projection_attention_pairwise_cosine"].append(
                    float(m_outputs["projection_attention_pairwise_cosine"].item())
                )
                metrics["train/xray_low_usage_alert"].append(float(x_outputs["xray_low_usage_alert"].item()))
                metrics["train/mri_low_usage_alert"].append(float(m_outputs["mri_low_usage_alert"].item()))
                metrics["train/xray_dominant_proto_alert"].append(float(x_outputs["xray_dominant_proto_alert"].item()))
                metrics["train/mri_dominant_proto_alert"].append(float(m_outputs["mri_dominant_proto_alert"].item()))
                metrics["train/projection_mil_loss"].append(float(projection_mil_loss.item()))
                metrics["train/projection_mil_valid_xray_anchors"].append(
                    float(projection_mil_stats["valid_xray_anchors"].item())
                )
                metrics["train/projection_mil_positive_fraction"].append(
                    float(projection_mil_stats["positive_fraction"].item())
                )
                if self.config.log_every_steps and (step_index + 1) % self.config.log_every_steps == 0:
                    self._print_iteration_log(
                        stage=stage,
                        step_index=step_index + 1,
                        num_steps=num_steps,
                        lr=lr,
                        epoch_start_time=epoch_start_time,
                        total_loss=float(total_loss.item()),
                        xray_loss=float(x_outputs["xray_loss"].item()),
                        mri_loss=float(m_outputs["mri_loss"].item()),
                    )
                    minimal_payload = self._build_minimal_step_log(
                        stage=stage,
                        lr=lr,
                        total_loss=float(total_loss.item()),
                        xray_loss=float(x_outputs["xray_loss"].item()),
                        mri_loss=float(m_outputs["mri_loss"].item()),
                        projection_mil_loss=float(projection_mil_loss.item()),
                        projection_mil_positive_fraction=float(projection_mil_stats["positive_fraction"].item()),
                        centroid_loss=float(centroid_loss.item()),
                        xray_assignment_entropy=float(x_outputs["xray_assignment_entropy"].item()),
                        mri_assignment_entropy=float(m_outputs["mri_assignment_entropy"].item()),
                    )
                    self._log_iteration(minimal_payload)
        else:
            raise ValueError(f"Unsupported stage '{stage}'.")

        result = {key: float(np.mean(value)) if value else 0.0 for key, value in metrics.items()}
        if self.is_distributed:
            result = {key: self._all_reduce_mean_scalar(value, self.loss_device) for key, value in result.items()}
        return result | {
            "train/stage": stage,
            "train/cross_modal_weight": cross_modal_weight,
            "train/lr": lr,
            "train/skipped_nonfinite_steps": float(self.state.get("skipped_nonfinite_steps", 0)),
        }

    @torch.no_grad()
    def _validate(self, epoch: int, stage: str, cross_modal_weight: float) -> dict[str, float]:
        self._projection_joint_path_context = (stage == "joint")
        self.model_system.eval()
        self.prototype_banks.eval()
        metrics: dict[str, list[float]] = {
            "val/xray_loss": [],
            "val/xray_swav_loss": [],
            "val/mri_loss": [],
            "val/centroid_loss": [],
            "val/xray_view_loss": [],
            "val/total_loss": [],
            "val/xray_assignment_entropy": [],
            "val/mri_assignment_entropy": [],
            "val/xray_prototype_usage": [],
            "val/mri_prototype_usage": [],
            "val/xray_embedding_norm": [],
            "val/mri_embedding_norm": [],
            "val/mri_queue_size": [],
            "val/projection_attention_diversity_loss": [],
            "val/projection_attention_pairwise_cosine": [],
            "val/xray_low_usage_alert": [],
            "val/mri_low_usage_alert": [],
            "val/xray_dominant_proto_alert": [],
            "val/mri_dominant_proto_alert": [],
            "val/projection_mil_loss": [],
            "val/projection_mil_valid_xray_anchors": [],
            "val/projection_mil_positive_fraction": [],
        }
        if self.xray_val_loader is None and self.mri_val_loader is None:
            return {key: 0.0 for key in metrics} | {"val/stage": stage}

        if stage == "joint" and self.xray_val_loader is not None and self.mri_val_loader is not None:
            x_iter = _cycle(self.xray_val_loader)
            m_iter = _cycle(self.mri_val_loader)
            num_steps = max(len(self.xray_val_loader), len(self.mri_val_loader))
            for _ in range(num_steps):
                x_batch = next(x_iter)
                m_batch = next(m_iter)
                x_outputs = self._run_xray_step(x_batch, update_queue=False, train=False, view_loss_active=False)
                m_outputs = self._run_mri_step(
                    m_batch,
                    update_queue=False,
                    train=False,
                    projection_joint_path=True,
                )
                centroid_loss = self._centroid_alignment_loss(x_outputs, m_outputs, ramp_weight=cross_modal_weight)
                projection_mil_raw_loss, projection_mil_stats = self._projection_mil_loss(
                    xray_batch=x_batch,
                    mri_batch=m_batch,
                    xray_outputs=x_outputs,
                    mri_outputs=m_outputs,
                )
                projection_mil_loss = float(self.config.lambda_projection_mil_contrastive) * projection_mil_raw_loss
                total_loss = (
                    x_outputs["xray_loss"]
                    + self.config.lambda_mri_loss * m_outputs["mri_loss"]
                    + centroid_loss
                    + projection_mil_loss
                )
                metrics["val/xray_loss"].append(float(x_outputs["xray_loss"].item()))
                metrics["val/xray_swav_loss"].append(float(x_outputs["xray_swav_loss"].item()))
                metrics["val/mri_loss"].append(float(m_outputs["mri_loss"].item()))
                metrics["val/centroid_loss"].append(float(centroid_loss.item()))
                metrics["val/total_loss"].append(float(total_loss.item()))
                metrics["val/xray_assignment_entropy"].append(float(x_outputs["xray_assignment_entropy"].item()))
                metrics["val/mri_assignment_entropy"].append(float(m_outputs["mri_assignment_entropy"].item()))
                metrics["val/xray_prototype_usage"].append(float(x_outputs["xray_prototype_usage"].item()))
                metrics["val/mri_prototype_usage"].append(float(m_outputs["mri_prototype_usage"].item()))
                metrics["val/xray_embedding_norm"].append(float(x_outputs["xray_embedding_norm"].item()))
                metrics["val/mri_embedding_norm"].append(float(m_outputs["mri_embedding_norm"].item()))
                metrics["val/mri_queue_size"].append(float(m_outputs["mri_queue_size"].item()))
                metrics["val/projection_attention_diversity_loss"].append(
                    float(m_outputs["projection_attention_diversity_loss"].item())
                )
                metrics["val/projection_attention_pairwise_cosine"].append(
                    float(m_outputs["projection_attention_pairwise_cosine"].item())
                )
                metrics["val/xray_low_usage_alert"].append(float(x_outputs["xray_low_usage_alert"].item()))
                metrics["val/mri_low_usage_alert"].append(float(m_outputs["mri_low_usage_alert"].item()))
                metrics["val/xray_dominant_proto_alert"].append(float(x_outputs["xray_dominant_proto_alert"].item()))
                metrics["val/mri_dominant_proto_alert"].append(float(m_outputs["mri_dominant_proto_alert"].item()))
                metrics["val/projection_mil_loss"].append(float(projection_mil_loss.item()))
                metrics["val/projection_mil_valid_xray_anchors"].append(
                    float(projection_mil_stats["valid_xray_anchors"].item())
                )
                metrics["val/projection_mil_positive_fraction"].append(
                    float(projection_mil_stats["positive_fraction"].item())
                )
        else:
            if self.xray_val_loader is not None:
                for batch in self.xray_val_loader:
                    outputs = self._run_xray_step(
                        batch, update_queue=False, train=False, view_loss_active=(stage == "warm_start")
                    )
                    metrics["val/xray_loss"].append(float(outputs["xray_loss"].item()))
                    metrics["val/xray_swav_loss"].append(float(outputs["xray_swav_loss"].item()))
                    metrics["val/xray_view_loss"].append(float(outputs["xray_view_loss"].item()))
                    metrics["val/total_loss"].append(float(outputs["xray_loss"].item()))
                    metrics["val/xray_assignment_entropy"].append(float(outputs["xray_assignment_entropy"].item()))
                    metrics["val/xray_prototype_usage"].append(float(outputs["xray_prototype_usage"].item()))
                    metrics["val/xray_embedding_norm"].append(float(outputs["xray_embedding_norm"].item()))
                    metrics["val/xray_low_usage_alert"].append(float(outputs["xray_low_usage_alert"].item()))
                    metrics["val/xray_dominant_proto_alert"].append(float(outputs["xray_dominant_proto_alert"].item()))
            if self.mri_val_loader is not None and self.config.ablations.enable_mri:
                for batch in self.mri_val_loader:
                    outputs = self._run_mri_step(batch, update_queue=False, train=False)
                    metrics["val/mri_loss"].append(float(outputs["mri_loss"].item()))
                    metrics["val/total_loss"].append(float(outputs["mri_loss"].item()))
                    metrics["val/mri_assignment_entropy"].append(float(outputs["mri_assignment_entropy"].item()))
                    metrics["val/mri_prototype_usage"].append(float(outputs["mri_prototype_usage"].item()))
                    metrics["val/mri_embedding_norm"].append(float(outputs["mri_embedding_norm"].item()))
                    metrics["val/mri_queue_size"].append(float(outputs["mri_queue_size"].item()))
                    metrics["val/projection_attention_diversity_loss"].append(
                        float(outputs["projection_attention_diversity_loss"].item())
                    )
                    metrics["val/projection_attention_pairwise_cosine"].append(
                        float(outputs["projection_attention_pairwise_cosine"].item())
                    )
                    metrics["val/mri_low_usage_alert"].append(float(outputs["mri_low_usage_alert"].item()))
                    metrics["val/mri_dominant_proto_alert"].append(float(outputs["mri_dominant_proto_alert"].item()))

        result = {key: float(np.mean(value)) if value else 0.0 for key, value in metrics.items()}
        if self.is_distributed:
            result = {key: self._all_reduce_mean_scalar(value, self.loss_device) for key, value in result.items()}
        result["val/stage"] = stage
        return result

    def train(self) -> dict[str, Any]:
        """Run the full warm-up + joint training schedule."""
        start_time = time.time()
        self.train_start_time = start_time
        self.run_start_global_step = int(self.state.get("global_step", 0))
        schedule = self.config.schedule
        total_epochs = schedule.warm_start_epochs + schedule.joint_epochs

        history: list[dict[str, Any]] = []
        for epoch in range(int(self.state["epoch"]), total_epochs):
            if self.xray_train_sampler is not None:
                self.xray_train_sampler.set_epoch(epoch)
            if self.mri_train_sampler is not None:
                self.mri_train_sampler.set_epoch(epoch)
            if self.xray_joint_train_sampler is not None:
                self.xray_joint_train_sampler.set_epoch(epoch)
            if self.mri_joint_train_sampler is not None:
                self.mri_joint_train_sampler.set_epoch(epoch)
            if self.xray_val_sampler is not None:
                self.xray_val_sampler.set_epoch(epoch)
            if self.mri_val_sampler is not None:
                self.mri_val_sampler.set_epoch(epoch)
            stage, cross_modal_weight = self._iter_stage(epoch)
            if stage == "done":
                break
            lr = self._set_lr(epoch)
            train_metrics = self._train_epoch(epoch, stage, cross_modal_weight, lr)
            self._distributed_barrier()
            record: dict[str, Any] = {"epoch": epoch + 1} | train_metrics

            if self.config.val_every_epochs > 0 and ((epoch + 1) % self.config.val_every_epochs == 0):
                print("Starting validation", flush=True)
                record |= self._validate(epoch, stage, cross_modal_weight)
                print("Finished validation", flush=True)
                self._distributed_barrier()

            self.state["epoch"] = epoch + 1
            if self.config.checkpoint.save_every_epochs > 0 and (
                (epoch + 1) % self.config.checkpoint.save_every_epochs == 0
            ):
                self._save_checkpoint(epoch + 1, tag="latest")
            self._distributed_barrier()

            if self.rank == 0:
                self._log_epoch_summary(self._build_epoch_log(record))
                self._print_epoch_gradient_audit(epoch, record)
            self._distributed_barrier()
            history.append(record)
            epoch_summary = {
                "epochs_completed": int(self.state["epoch"]),
                "global_step": int(self.state["global_step"]),
                "elapsed_seconds": time.time() - start_time,
                "ablations": asdict(self.config.ablations),
                "history": history,
            }
            summary_path = self.output_dir / "training_summary.json"
            if (not self.is_distributed) or (self.rank == 0):
                _write_json_atomic(summary_path, epoch_summary)
            self._distributed_barrier()

        self._save_checkpoint(int(self.state["epoch"]), tag="latest")
        # The final-epoch checkpoint is the one evaluated downstream (no checkpoint selection).
        self._save_checkpoint(int(self.state["epoch"]), tag="final", epoch_copy=False)
        self._distributed_barrier()
        elapsed = time.time() - start_time
        summary = {
            "epochs_completed": int(self.state["epoch"]),
            "global_step": int(self.state["global_step"]),
            "final_checkpoint": str(Path(self.config.checkpoint.output_dir) / "final.pt"),
            "elapsed_seconds": elapsed,
            "ablations": asdict(self.config.ablations),
            "history": history,
        }
        summary_path = self.output_dir / "training_summary.json"
        # Only rank 0 writes shared run artifacts in distributed mode.
        if (not self.is_distributed) or (self.rank == 0):
            _write_json_atomic(summary_path, summary)
        self._distributed_barrier()
        self.wandb.finish()
        if self.is_distributed:
            self._cleanup_distributed()
        return summary
