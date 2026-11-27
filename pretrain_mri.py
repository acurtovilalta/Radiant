"""Models Genesis restoration pretraining for the MRI encoder.

Corrupts a clean MRI crop (non-linear intensity remap, local pixel shuffle,
in/out-painting) and trains the MRI encoder + a light decoder to reconstruct the
clean original. Only the encoder backbone is kept afterwards.

`mri_model.encoder_type` selects the encoder:
  - `volume_3d` (default): the 3D MRI encoder of RADIANT, on 3D crops
    (configs/mri_pretraining/mri_encoder_genesis.yaml);
  - `slice_2d`: the 2D slice encoder of control C4 (SliceRADIANT), on the same 2D slices that C4
    uses; every slice is corrupted and reconstructed on its own
    (configs/mri_pretraining/slice_encoder_genesis.yaml).
"""

from __future__ import annotations

import os

# Cap BLAS thread pools *before* numpy/torch are imported. Without this, every
# DataLoader worker process independently tries to use all logical cores for
# BLAS ops, causing severe oversubscription/thrashing once more than a couple
# of workers run concurrently (each worker only ever does single-sample work).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import math
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings(
    "ignore", message=r".*cuda\.cudart module is deprecated.*", category=FutureWarning
)

from radiant.data.datasets import MRIDataConfig, build_mri_dataloader, load_dataclass_config
from radiant.data.slices import SliceMultiCropTransform
from radiant.data.transforms import MRIAugmentationConfig, XRayAugmentationConfig, build_mri_transform
from radiant.models.mri_encoder import MRI3DEncoder, MRIEncoderConfig, STAGE_STRIDE_SCHEDULES
from radiant.models.slice_encoder import MRISlice2DEncoder
from radiant.training import OptimizerConfig, load_yaml
from radiant.training.trainer import StageTrainer, _import_wandb, cosine_with_warmup, set_seed

wandb = _import_wandb()
# 3D encoder: conv1/bn1/layer1-4; 2D slice encoder: stem/layer1-4.
_BACKBONE_PREFIXES = ("conv1.", "bn1.", "stem.", "layer1.", "layer2.", "layer3.", "layer4.")

@dataclass(slots=True)
class GenesisAugmentationConfig:
    """Independent per-transform probabilities and parameters for Models Genesis."""
    nonlinear_prob: float = 0.9
    nonlinear_control_points: int = 4
    local_shuffle_prob: float = 0.5
    local_shuffle_num_windows: int = 200
    local_shuffle_max_window_size: int = 8
    inpaint_prob: float = 0.5
    inpaint_num_cuboids: int = 5
    inpaint_size_range: tuple[float, float] = (0.1, 0.2)
    outpaint_prob: float = 0.2
    outpaint_keep_size_range: tuple[float, float] = (0.6, 0.8)

@dataclass(slots=True)
class PretrainConfig:
    """Top-level config: the mri_data / mri_model / optimizer blocks of the RADIANT configs plus `genesis`."""
    mri_data: MRIDataConfig
    mri_model: MRIEncoderConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    genesis: GenesisAugmentationConfig = field(default_factory=GenesisAugmentationConfig)
    epochs: int = 100
    output_dir: str = "outputs/mri_pretraining"
    wandb_project: str = "mri_pretraining"
    wandb_run_name: str | None = None
    num_visual_examples: int = 10
    log_every_steps: int = 20
    seed: int = 13
    use_amp: bool = True

def _load_config(path: str | Path) -> PretrainConfig:
    payload = load_yaml(path)
    return PretrainConfig(
        mri_data=load_dataclass_config(payload.get("mri_data", {}), MRIDataConfig),
        mri_model=load_dataclass_config(payload.get("mri_model", {}), MRIEncoderConfig),
        optimizer=load_dataclass_config(payload.get("optimizer", {}), OptimizerConfig),
        genesis=load_dataclass_config(payload.get("genesis", {}), GenesisAugmentationConfig),
        epochs=int(payload.get("epochs", 100)),
        output_dir=str(payload.get("output_dir", "outputs/mri_pretraining")),
        wandb_project=str(payload.get("wandb_project", "mri_pretraining")),
        wandb_run_name=payload.get("wandb_run_name"),
        num_visual_examples=int(payload.get("num_visual_examples", 10)),
        log_every_steps=int(payload.get("log_every_steps", 20)),
        seed=int(payload.get("seed", 13)),
        use_amp=bool(payload.get("use_amp", True)),
    )

def _nonlinear_intensity(volume: np.ndarray, num_points: int, rng: np.random.Generator, num_samples: int = 1000) -> np.ndarray:
    """Remap intensities through a random monotonically increasing Bezier curve."""
    v_min, v_max = float(volume.min()), float(volume.max())
    if v_max - v_min < 1e-6:
        return volume
    xs, ys = np.sort(rng.uniform(0.0, 1.0, size=num_points)), np.sort(rng.uniform(0.0, 1.0, size=num_points))
    xs[0], xs[-1], ys[0], ys[-1] = 0.0, 1.0, 0.0, 1.0
    degree, index = num_points - 1, np.arange(num_points)
    t = np.linspace(0.0, 1.0, num_samples)
    coeff = np.array([math.comb(degree, i) for i in index])[:, None]
    bernstein = coeff * (t[None, :] ** index[:, None]) * ((1.0 - t[None, :]) ** (degree - index[:, None]))
    curve_x, curve_y = xs @ bernstein, ys @ bernstein
    order = np.argsort(curve_x)
    normalized = (volume - v_min) / (v_max - v_min)
    mapped = np.interp(normalized, curve_x[order], curve_y[order])
    return (mapped * (v_max - v_min) + v_min).astype(np.float32)

def _local_pixel_shuffle(volume: np.ndarray, num_windows: int, max_window_size: int, rng: np.random.Generator) -> np.ndarray:
    volume = volume.copy()
    dims = volume.shape
    max_size = [max(1, min(max_window_size, dim)) for dim in dims]
    for _ in range(num_windows):
        size = [int(rng.integers(1, limit + 1)) for limit in max_size]
        start = [int(rng.integers(0, dim - extent + 1)) for dim, extent in zip(dims, size)]
        block = tuple(slice(s, s + extent) for s, extent in zip(start, size))
        flat = volume[block].reshape(-1)
        rng.shuffle(flat)
        volume[block] = flat.reshape(size)
    return volume

def _random_cuboid(dims: tuple[int, ...], size_range: tuple[float, float], rng: np.random.Generator) -> tuple[slice, ...]:
    size = [max(1, int(rng.uniform(*size_range) * dim)) for dim in dims]
    start = [int(rng.integers(0, dim - extent + 1)) for dim, extent in zip(dims, size)]
    return tuple(slice(s, s + extent) for s, extent in zip(start, size))

def _inpaint(volume: np.ndarray, num_cuboids: int, size_range: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    volume = volume.copy()
    v_min, v_max = float(volume.min()), float(volume.max())
    for _ in range(num_cuboids):
        cuboid = _random_cuboid(volume.shape, size_range, rng)
        volume[cuboid] = rng.uniform(v_min, v_max, size=volume[cuboid].shape).astype(np.float32)
    return volume

def _outpaint(volume: np.ndarray, keep_size_range: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    v_min, v_max = float(volume.min()), float(volume.max())
    noise = rng.uniform(v_min, v_max, size=volume.shape).astype(np.float32)
    keep = _random_cuboid(volume.shape, keep_size_range, rng)
    noise[keep] = volume[keep]
    return noise

def _corrupt_volume(volume: np.ndarray, cfg: GenesisAugmentationConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply Models Genesis corruption to a single [D, H, W] clean volume."""
    if rng.random() < cfg.local_shuffle_prob:
        volume = _local_pixel_shuffle(volume, cfg.local_shuffle_num_windows, cfg.local_shuffle_max_window_size, rng)
    if rng.random() < cfg.nonlinear_prob:
        volume = _nonlinear_intensity(volume, cfg.nonlinear_control_points, rng)
    apply_inpaint = rng.random() < cfg.inpaint_prob
    apply_outpaint = rng.random() < cfg.outpaint_prob
    if apply_inpaint and apply_outpaint:
        # Mutually exclusive, per the original method: resolve the conflict with a coin flip.
        apply_inpaint, apply_outpaint = (True, False) if rng.random() < 0.5 else (False, True)
    if apply_inpaint:
        volume = _inpaint(volume, cfg.inpaint_num_cuboids, cfg.inpaint_size_range, rng)
    elif apply_outpaint:
        volume = _outpaint(volume, cfg.outpaint_keep_size_range, rng)
    return volume


def genesis_corrupt_one(clean: torch.Tensor, cfg: GenesisAugmentationConfig, rng: np.random.Generator) -> torch.Tensor:
    """Apply Models Genesis corruption to a single [1, D, H, W] clean volume (or a [1, H, W] slice)."""
    corrupted = clean.clone()
    corrupted[0] = torch.from_numpy(_corrupt_volume(corrupted[0].numpy(), cfg, rng))
    return corrupted


class _GenesisPairTransform:
    """Crops one clean MRI view and pairs it with an independently corrupted copy.

    Passed as ``build_mri_dataloader``'s ``transform`` hook, so MRIPretrainingDataset
    calls it inside each DataLoader worker process: corruption runs in parallel across
    workers and overlaps with GPU compute, instead of blocking the main training loop.
    """

    def __init__(self, mri_data: MRIDataConfig, genesis_cfg: GenesisAugmentationConfig) -> None:
        overrides = dict(mri_data.augmentations or {})
        overrides.pop("num_global_crops", None)
        overrides.pop("num_local_crops", None)
        augmentation_config = MRIAugmentationConfig(
            global_size=mri_data.global_crop_size,
            local_size=mri_data.local_crop_size,
            num_global_crops=1,
            num_local_crops=0,
            **overrides,
        )
        self._crop_transform = build_mri_transform(augmentation_config)
        self._genesis_cfg = genesis_cfg

    def __call__(self, volume: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        global_views, _ = self._crop_transform(volume)
        clean = global_views[0]
        corrupted = genesis_corrupt_one(clean, self._genesis_cfg, np.random.default_rng())
        return [clean, corrupted], []

class _GenesisSlicePairTransform:
    """2D counterpart of `_GenesisPairTransform` for the C4 slice encoder.

    The [K, H, W] slices of a volume (selected exactly as in C4) get one shared crop, and every
    slice is corrupted independently. Returns the clean and corrupted [K, 1, h, w] views.
    """

    def __init__(self, mri_data: MRIDataConfig, genesis_cfg: GenesisAugmentationConfig) -> None:
        overrides = dict(mri_data.augmentations or {})
        overrides.pop("num_global_crops", None)
        overrides.pop("num_local_crops", None)
        size = lambda value: int(value[-1]) if isinstance(value, (list, tuple)) else int(value)  # noqa: E731
        self._crop_transform = SliceMultiCropTransform(
            XRayAugmentationConfig(
                global_size=size(mri_data.global_crop_size),
                local_size=size(mri_data.local_crop_size),
                num_global_crops=1,
                num_local_crops=0,
                **overrides,
            )
        )
        self._genesis_cfg = genesis_cfg

    def __call__(self, slices: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        global_views, _ = self._crop_transform(slices)
        clean = global_views[0]
        if clean.ndim == 3:  # a single slice comes back as [1, h, w]
            clean = clean.unsqueeze(0)
        rng = np.random.default_rng()
        corrupted = torch.stack([genesis_corrupt_one(view, self._genesis_cfg, rng) for view in clean])
        return [clean, corrupted], []


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, stride: tuple[int, int, int]) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=stride, stride=stride)
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.up(inputs), skip], dim=1))

class MRIGenesisDecoder(nn.Module):
    """Transposed-conv decoder mirroring MRI3DEncoder's downsampling, with encoder skips."""

    def __init__(self, base_channels: int, in_channels: int, stage_strides: tuple[tuple[int, int, int], ...]) -> None:
        super().__init__()
        b = base_channels
        self.up_layer3 = _UpBlock(8 * b, 4 * b, 4 * b, stage_strides[3])
        self.up_layer2 = _UpBlock(4 * b, 2 * b, 2 * b, stage_strides[2])
        self.up_layer1 = _UpBlock(2 * b, b, b, stage_strides[1])
        self.up_stem = _UpBlock(b, b, b, (1, 2, 2))
        self.to_input_res = nn.ConvTranspose3d(b, b, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.out_conv = nn.Conv3d(b, in_channels, kernel_size=1)

    def forward(self, intermediate: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.up_layer3(intermediate["layer4"], intermediate["layer3"])
        x = self.up_layer2(x, intermediate["layer2"])
        x = self.up_layer1(x, intermediate["layer1"])
        x = self.up_stem(x, intermediate["stem"])
        return self.out_conv(self.to_input_res(x))

class MRIGenesisModel(nn.Module):
    """MRI3DEncoder plus a light restoration decoder. Projectors and bridge are unused."""

    def __init__(self, mri_model_config: MRIEncoderConfig) -> None:
        super().__init__()
        self.encoder = MRI3DEncoder(mri_model_config)
        stage_strides = STAGE_STRIDE_SCHEDULES[mri_model_config.stage_stride_schedule]
        self.decoder = MRIGenesisDecoder(mri_model_config.base_channels, mri_model_config.in_channels, stage_strides)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(volume, compute_projection_bridge=False)
        return self.decoder(encoded.intermediate_feature_maps)

class _UpBlock2D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(inputs)
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([upsampled, skip], dim=1))


class MRISliceGenesisModel(nn.Module):
    """C4 2D slice encoder plus a light U-Net-style restoration decoder, applied to every slice.

    Input [B, K, 1, H, W]; output [B * K, 1, H, W]. Projector and projection head are unused.
    """

    def __init__(self, mri_model_config: MRIEncoderConfig) -> None:
        super().__init__()
        self.encoder = MRISlice2DEncoder(mri_model_config)
        self.encoder.eval()  # probe the stage widths without touching the BatchNorm statistics
        with torch.no_grad():
            features = self._encode(torch.zeros(1, mri_model_config.in_channels, 64, 64))
        self.encoder.train()
        channels = {name: int(feature.shape[1]) for name, feature in features.items()}
        self.up_layer3 = _UpBlock2D(channels["layer4"], channels["layer3"], channels["layer3"])
        self.up_layer2 = _UpBlock2D(channels["layer3"], channels["layer2"], channels["layer2"])
        self.up_layer1 = _UpBlock2D(channels["layer2"], channels["layer1"], channels["layer1"])
        self.up_stem = _UpBlock2D(channels["layer1"], channels["stem"], channels["stem"])
        self.to_input_res = nn.ConvTranspose2d(channels["stem"], channels["stem"], kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(channels["stem"], mri_model_config.in_channels, kernel_size=1)

    def _encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.encoder.stem
        features = {"stem": stem[2](stem[1](stem[0](images)))}  # conv1, bn1, relu: 1/2 resolution
        x = self.encoder.layer1(stem[3](features["stem"]))  # maxpool: 1/4
        features["layer1"] = x
        features["layer2"] = x = self.encoder.layer2(x)
        features["layer3"] = x = self.encoder.layer3(x)
        features["layer4"] = self.encoder.layer4(x)
        return features

    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        images = slices.flatten(0, 1)
        features = self._encode(images)
        x = self.up_layer3(features["layer4"], features["layer3"])
        x = self.up_layer2(x, features["layer2"])
        x = self.up_layer1(x, features["layer1"])
        x = self.up_stem(x, features["stem"])
        x = self.to_input_res(x)
        if x.shape[-2:] != images.shape[-2:]:
            x = F.interpolate(x, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return self.out_conv(x)


def _reconstruction_target(clean: torch.Tensor, encoder_type: str) -> torch.Tensor:
    """The clean views in the reconstruction's layout ([B * K, 1, H, W] for the slice encoder)."""
    return clean.flatten(0, 1) if encoder_type == "slice_2d" else clean


def _backbone_state_dict(encoder: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value for key, value in encoder.state_dict().items() if key.startswith(_BACKBONE_PREFIXES)}

def _save_checkpoint(model: nn.Module, mri_model_config: MRIEncoderConfig, epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mri_encoder_state_dict": _backbone_state_dict(model.encoder),
            "encoder_type": mri_model_config.encoder_type,
            "stage_stride_schedule": mri_model_config.stage_stride_schedule,
            "mri_model_config": asdict(mri_model_config),
            "epoch": epoch,
        },
        path,
    )

def _save_reconstruction_panel(
    corrupted: torch.Tensor, reconstruction: torch.Tensor, clean: torch.Tensor, max_examples: int, path: Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    num_examples = min(max_examples, corrupted.shape[0])
    if corrupted.ndim == 4:  # 2D slices [N, 1, H, W]: add a depth axis of 1
        corrupted, reconstruction, clean = (tensor.unsqueeze(2) for tensor in (corrupted, reconstruction, clean))
    mid_depth = corrupted.shape[2] // 2
    figure, axes = plt.subplots(num_examples, 3, figsize=(6, 2 * num_examples), squeeze=False)
    for row in range(num_examples):
        for col, (name, tensor) in enumerate((("corrupted", corrupted), ("reconstruction", reconstruction), ("clean", clean))):
            axis = axes[row][col]
            axis.imshow(tensor[row, 0, mid_depth].numpy(), cmap="gray")
            axis.axis("off")
            if row == 0:
                axis.set_title(name)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path

def _run_validation(
    model: nn.Module, val_loader: Any, loss_fn: Any, device: torch.device, use_amp: bool, encoder_type: str = "volume_3d"
) -> tuple[float, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    total_loss, num_batches = 0.0, 0
    first: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    autocast_enabled = use_amp and device.type == "cuda"
    with torch.no_grad():
        for batch in val_loader:
            clean, corrupted = batch["global_views"][0], batch["global_views"][1]
            clean = _reconstruction_target(clean, encoder_type)
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                reconstruction = model(corrupted.to(device))
                loss = loss_fn(reconstruction, clean.to(device))
            total_loss += float(loss.item())
            num_batches += 1
            if first is None:
                first = (
                    _reconstruction_target(corrupted, encoder_type).cpu(),
                    reconstruction.detach().float().cpu(),
                    clean.cpu(),
                )
    val_loss = total_loss / max(1, num_batches)
    return (val_loss, *first) if first is not None else (val_loss, None, None, None)

def train(config: PretrainConfig, use_l1: bool) -> None:
    set_seed(config.seed)
    torch.set_num_threads(1)  # main process barely does CPU work; leave cores to DataLoader workers.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)

    # Corruption runs inside this transform, which build_mri_dataloader hands to
    # MRIPretrainingDataset — so it executes in the parallel DataLoader workers
    # instead of blocking the main process between GPU steps.
    encoder_type = config.mri_model.encoder_type
    if encoder_type == "slice_2d":
        if not config.mri_data.slice_mode:
            raise ValueError("The slice_2d encoder needs mri_data.slice_mode (the slices C4 is trained on).")
        pair_transform = _GenesisSlicePairTransform(config.mri_data, config.genesis)
    elif encoder_type == "volume_3d":
        pair_transform = _GenesisPairTransform(config.mri_data, config.genesis)
    else:
        raise ValueError(f"Models Genesis pretraining supports volume_3d and slice_2d, not {encoder_type!r}.")
    train_loader = build_mri_dataloader(config.mri_data, transform=pair_transform)
    val_loader = build_mri_dataloader(replace(config.mri_data, split="val", shuffle=False), transform=pair_transform)

    model = (MRISliceGenesisModel if encoder_type == "slice_2d" else MRIGenesisModel)(config.mri_model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
    )
    loss_fn = F.l1_loss if use_l1 else F.mse_loss
    autocast_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled)

    run = None
    if wandb is not None:
        run = wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=asdict(config))

    global_step = 0
    train_start_time = time.time()
    for epoch in range(config.epochs):
        lr = cosine_with_warmup(
            epoch, config.epochs, config.optimizer.lr, config.optimizer.warmup_epochs,
            config.optimizer.lr * config.optimizer.min_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        running_loss, num_batches = 0.0, 0
        for batch in train_loader:
            clean, corrupted = batch["global_views"][0], batch["global_views"][1]
            clean = _reconstruction_target(clean, encoder_type)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                reconstruction = model(corrupted.to(device))
                loss = loss_fn(reconstruction, clean.to(device))
            scaler.scale(loss).backward()
            if config.optimizer.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            num_batches += 1
            global_step += 1
            if run is not None and global_step % config.log_every_steps == 0:
                run.log({"train/loss": float(loss.item()), "train/lr": lr}, step=global_step)
        train_loss = running_loss / max(1, num_batches)

        model.eval()
        val_loss, corrupted_ex, recon_ex, clean_ex = _run_validation(model, val_loader, loss_fn, device, config.use_amp, encoder_type)

        completed_epochs = epoch + 1
        elapsed = time.time() - train_start_time
        remaining = (elapsed / completed_epochs) * (config.epochs - completed_epochs)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.3e} "
            f"elapsed={StageTrainer._format_duration(elapsed)} eta_remaining={StageTrainer._format_duration(remaining)}",
            flush=True,
        )

        if run is not None:
            payload: dict[str, Any] = {"val/loss": val_loss, "train/epoch_loss": train_loss, "epoch": epoch}
            if corrupted_ex is not None:
                panel_path = _save_reconstruction_panel(
                    corrupted_ex, recon_ex, clean_ex, config.num_visual_examples,
                    output_dir / "reconstructions" / f"epoch_{epoch:04d}.png",
                )
                payload["val/reconstructions"] = wandb.Image(str(panel_path))
            run.log(payload, step=global_step)

        _save_checkpoint(model, config.mri_model, epoch, output_dir / "checkpoint_latest.pt")

    if run is not None:
        run.finish()

def _mean_pairwise_cosine(depth_tokens: torch.Tensor) -> float:
    normalized = F.normalize(depth_tokens, dim=-1, eps=1e-6)
    cosine = torch.einsum("bdc,bec->bde", normalized, normalized)
    off_diagonal = ~torch.eye(cosine.shape[-1], dtype=torch.bool, device=cosine.device)
    return float(cosine[:, off_diagonal].mean().item())

def _participation_ratio_effective_rank(depth_tokens: torch.Tensor) -> float:
    centered = depth_tokens - depth_tokens.mean(dim=1, keepdim=True)
    eigenvalues = torch.linalg.svdvals(centered) ** 2
    numerator = eigenvalues.sum(dim=-1) ** 2
    denominator = (eigenvalues**2).sum(dim=-1).clamp_min(1e-12)
    return float((numerator / denominator).mean().item())

def diagnose(checkpoint_path: str, config: PretrainConfig) -> None:
    """Load a checkpoint, run one batch, and report layer4 depth-token diversity."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("encoder_type", "volume_3d") != "volume_3d":
        raise ValueError("--diagnose reports depth-token diversity and applies to the 3D MRI encoder only.")
    mri_model_config = MRIEncoderConfig(**checkpoint["mri_model_config"])
    encoder = MRI3DEncoder(mri_model_config)
    encoder.load_state_dict(checkpoint["mri_encoder_state_dict"], strict=False)
    encoder.eval()

    loader = build_mri_dataloader(config.mri_data)
    batch = next(iter(loader))
    volume = batch["global_views"][0]

    with torch.no_grad():
        output = encoder(volume, compute_projection_bridge=False)
    depth_tokens = output.feature_map.mean(dim=(3, 4)).permute(0, 2, 1)  # [B, D, C]

    mean_cosine = _mean_pairwise_cosine(depth_tokens)
    effective_rank = _participation_ratio_effective_rank(depth_tokens)
    reference_k = int(mri_model_config.mri_projection_num_projections)
    print(f"checkpoint={checkpoint_path}")
    print(f"stage_stride_schedule={checkpoint.get('stage_stride_schedule')}")
    print(f"depth_tokens_shape={tuple(depth_tokens.shape)}")
    print(f"mean_pairwise_cosine={mean_cosine:.4f}")
    print(f"participation_ratio_effective_rank={effective_rank:.4f} (K={reference_k})")
    print("OK: comfortably above K" if effective_rank > 1.5 * reference_k else "WARNING: too close to K")

def main() -> None:
    parser = argparse.ArgumentParser(description="Models Genesis restoration pretraining for the MRI encoder (3D, or the 2D slice encoder of C4).")
    parser.add_argument("--config", required=True, help="Path to YAML pretraining config.")
    parser.add_argument("--l1", action="store_true", help="Use L1 reconstruction loss instead of MSE.")
    parser.add_argument("--diagnose", metavar="CKPT", default=None, help="Diagnose a checkpoint instead of training.")
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.diagnose:
        diagnose(args.diagnose, config)
    else:
        train(config, use_l1=args.l1)

if __name__ == "__main__":
    main()
