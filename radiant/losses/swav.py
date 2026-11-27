"""SwAV with anatomy-family prototype banks.

For every crop that provides targets ("assignment crop"), each sample's embedding is scored against
the prototypes of its own anatomy family, and the Sinkhorn-Knopp algorithm turns those scores into
balanced soft assignments (codes). Every other crop of the same sample then has to predict these
codes (swapped prediction, cross-entropy). Sinkhorn runs separately per family, optionally on top
of a per-family queue of recent embeddings so that small families still get balanced codes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from radiant.models.prototypes import AnatomyPrototypeBanks


@dataclass(slots=True)
class SwAVAssignmentConfig:
    """Sinkhorn and feature-queue settings (`xray_assignment` / `mri_assignment` in the configs)."""

    sinkhorn_epsilon: float = 0.05
    sinkhorn_iterations: int = 3
    use_feature_queue: bool = False
    queue_size: int = 0  # embeddings kept per anatomy family
    queue_start_steps: int = 0  # assigner steps before the queue is used


@dataclass(slots=True)
class SwAVAssignmentOutput:
    """Sinkhorn codes for one crop of a batch."""

    family_names: list[str]  # routed anatomy family of every sample
    family_assignments: torch.Tensor  # [B, P_max] codes, zero beyond each family's prototype count
    family_prototype_counts: torch.Tensor  # [B] number of prototypes in each sample's family bank
    queue_sizes: torch.Tensor  # [B] queued embeddings used for each sample's Sinkhorn


@dataclass(slots=True)
class SwAVLossConfig:
    """Swapped-prediction settings (`xray_swav_loss` / `mri_swav_loss` in the configs)."""

    temperature: float = 0.1
    num_assignment_crops: int = 2  # the first N crops (the global crops) provide the codes


@dataclass(slots=True)
class SwAVLossOutput:
    loss: torch.Tensor
    assignment_outputs: list[SwAVAssignmentOutput]  # one per assignment crop


def sinkhorn_knopp(logits: torch.Tensor, epsilon: float = 0.05, num_iters: int = 3) -> torch.Tensor:
    """Balanced soft assignments of a batch to prototypes, [N, P] -> [N, P].

    Alternately normalizes over prototypes and over samples so that every prototype receives an
    equal share of the batch. Runs in fp32: embeddings and prototypes are L2-normalized, so
    |logits| <= 1 and exp(logits / epsilon) stays well inside the fp32 range. Global scalar
    factors (the max-shift, the uniform marginals) cancel in the row/column normalizations and
    are therefore omitted.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, prototypes).")
    if logits.shape[0] == 0:
        return logits.new_zeros(logits.shape)

    original_dtype = logits.dtype
    scaled = (logits / epsilon).float()
    scaled = scaled - scaled.max()
    assignments = torch.exp(scaled).t()
    assignments = assignments / assignments.sum().clamp_min(1e-12)
    for _ in range(num_iters):
        assignments = assignments / assignments.sum(dim=1, keepdim=True).clamp_min(1e-12)
        assignments = assignments / assignments.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return assignments.t().to(original_dtype)


def _swav_prediction_loss(
    logits: torch.Tensor,
    assignments: torch.Tensor,
    counts: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Cross-entropy between predicted prototype scores and target codes.

    Samples can belong to families with different prototype counts; prototype slots beyond a
    sample's count are padding and are masked out.
    """
    if logits.shape != assignments.shape:
        raise ValueError("logits and assignments must have identical shapes.")
    mask = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0) < counts.unsqueeze(1)
    log_probs = F.log_softmax(logits.masked_fill(~mask, float("-inf")) / temperature, dim=-1)
    return -(assignments * mask.to(assignments.dtype) * log_probs).sum(dim=-1).mean()


class FamilyFeatureQueue(nn.Module):
    """Fixed-size FIFO (ring buffer) of recent embeddings for every anatomy family."""

    def __init__(self, families: tuple[str, ...], embedding_dim: int, queue_size: int) -> None:
        super().__init__()
        if queue_size < 0:
            raise ValueError("queue_size must be non-negative.")
        self.embedding_dim = int(embedding_dim)
        self.queue_size = int(queue_size)
        self.family_to_slot = {family: index for index, family in enumerate(families)}
        self.register_buffer("queues", torch.zeros(len(families), max(1, queue_size), embedding_dim))
        self.register_buffer("counts", torch.zeros(len(families), dtype=torch.long))
        self.register_buffer("pointers", torch.zeros(len(families), dtype=torch.long))

    def get(self, family_name: str, device: torch.device | None = None) -> torch.Tensor:
        """Queued embeddings of one family, [n, d] with n <= queue_size."""
        target_device = device or self.queues.device
        slot = self.family_to_slot[family_name]
        count = int(self.counts[slot].item()) if self.queue_size > 0 else 0
        if count == 0:
            return torch.zeros(0, self.embedding_dim, dtype=self.queues.dtype, device=target_device)
        # Later crops enqueue before the joint loss is backpropagated. Keep
        # the embeddings used by earlier assignment graphs independent of
        # the mutable ring buffer.
        return self.queues[slot, :count].to(target_device).clone()

    @torch.no_grad()
    def enqueue(self, family_name: str, embeddings: torch.Tensor) -> None:
        """Append embeddings to a family's queue, overwriting the oldest entries when full."""
        if self.queue_size == 0 or embeddings.numel() == 0:
            return
        slot = self.family_to_slot[family_name]
        embeddings = embeddings.detach().to(self.queues.device, dtype=self.queues.dtype)
        # Only the last queue_size entries would survive anyway; this also keeps the
        # index_copy_ indices below unique.
        embeddings = embeddings[-self.queue_size :]
        n = embeddings.shape[0]
        indices = (self.pointers[slot] + torch.arange(n, device=self.queues.device)) % self.queue_size
        self.queues[slot].index_copy_(0, indices, embeddings)
        self.pointers[slot] = (self.pointers[slot] + n) % self.queue_size
        self.counts[slot] = torch.clamp(self.counts[slot] + n, max=self.queue_size)


class SwAVAssigner(nn.Module):
    """Per-family Sinkhorn codes for a batch of embeddings, with an optional per-family queue."""

    def __init__(self, prototype_banks: AnatomyPrototypeBanks, config: SwAVAssignmentConfig) -> None:
        super().__init__()
        self.prototype_banks = prototype_banks
        self.config = config
        # Number of assigner calls with advance_step=True; the queue is used from queue_start_steps on.
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        self.queue = (
            FamilyFeatureQueue(
                families=tuple(prototype_banks.family_to_id),
                embedding_dim=prototype_banks.config.embedding_dim,
                queue_size=config.queue_size,
            )
            if config.use_feature_queue and config.queue_size > 0
            else None
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        anatomy_family: list[str] | tuple[str, ...],
        update_queue: bool = False,
        advance_step: bool = True,
    ) -> SwAVAssignmentOutput:
        """Codes for [B, d] embeddings; `update_queue` pushes this batch into the family queues."""
        prototype_output = self.prototype_banks(embeddings=embeddings, anatomy_family=anatomy_family)
        family_names = prototype_output.family_names  # normalized to the known families
        batch_size = embeddings.shape[0]
        family_assignments = embeddings.new_zeros(batch_size, prototype_output.family_logits.shape[1])
        queue_sizes = torch.zeros(batch_size, dtype=torch.long, device=embeddings.device)
        use_queue = self.queue is not None and int(self.step.item()) >= self.config.queue_start_steps

        indices_by_family: dict[str, list[int]] = {}
        for index, family_name in enumerate(family_names):
            indices_by_family.setdefault(family_name, []).append(index)

        for family_name, sample_indices in indices_by_family.items():
            index_tensor = torch.tensor(sample_indices, dtype=torch.long, device=embeddings.device)
            prototype_count = self.prototype_banks.family_counts[family_name]
            current_logits = prototype_output.family_logits.index_select(0, index_tensor)[:, :prototype_count]

            # Queued embeddings join the Sinkhorn batch (as extra rows) but only the current
            # samples' codes are kept.
            queue_embeddings = self.queue.get(family_name, device=embeddings.device) if use_queue else None
            if queue_embeddings is not None:
                queue_sizes[index_tensor] = queue_embeddings.shape[0]
            if queue_embeddings is not None and queue_embeddings.shape[0] > 0:
                queue_logits, _ = self.prototype_banks.get_family_bank(family_name)(queue_embeddings)
                sinkhorn_logits = torch.cat([queue_logits, current_logits], dim=0)
            else:
                sinkhorn_logits = current_logits
            codes = sinkhorn_knopp(
                sinkhorn_logits,
                epsilon=self.config.sinkhorn_epsilon,
                num_iters=self.config.sinkhorn_iterations,
            )
            family_assignments[index_tensor, :prototype_count] = codes[-len(sample_indices) :]

            if update_queue and self.queue is not None:
                self.queue.enqueue(family_name, embeddings.index_select(0, index_tensor))

        if advance_step:
            self.step += 1
        return SwAVAssignmentOutput(
            family_names=family_names,
            family_assignments=family_assignments,
            family_prototype_counts=prototype_output.family_prototype_counts,
            queue_sizes=queue_sizes,
        )


def swav_loss(
    projections: list[torch.Tensor],
    anatomy_family: list[str] | tuple[str, ...],
    assigner: SwAVAssigner,
    config: SwAVLossConfig,
    update_queue: bool = False,
    advance_assigner_step: bool = True,
) -> SwAVLossOutput:
    """Multi-crop SwAV swapped-prediction loss.

    `projections` holds the L2-normalized embeddings of every crop ([B, d] each, global crops
    first). Each of the first `num_assignment_crops` crops provides codes that all *other* crops
    predict; the loss is the mean over assignment crops of the mean over predicting crops.
    """
    if len(projections) < 2:
        raise ValueError("SwAV requires at least two crop projections.")
    num_assignment_crops = min(config.num_assignment_crops, len(projections))
    if num_assignment_crops <= 0:
        raise ValueError("num_assignment_crops must be positive.")

    assignment_outputs: list[SwAVAssignmentOutput] = []
    per_assignment_crop_losses: list[torch.Tensor] = []
    for assignment_index in range(num_assignment_crops):
        assignment = assigner(
            embeddings=projections[assignment_index],
            anatomy_family=anatomy_family,
            update_queue=update_queue,
            # The step counter advances once per loss call, not once per assignment crop.
            advance_step=advance_assigner_step and assignment_index == num_assignment_crops - 1,
        )
        assignment_outputs.append(assignment)
        targets = assignment.family_assignments.detach()

        swapped_losses = []
        for prediction_index, prediction in enumerate(projections):
            if prediction_index == assignment_index:
                continue
            prediction_logits = assigner.prototype_banks(
                embeddings=prediction, anatomy_family=assignment.family_names
            ).family_logits
            swapped_losses.append(
                _swav_prediction_loss(
                    logits=prediction_logits,
                    assignments=targets,
                    counts=assignment.family_prototype_counts,
                    temperature=config.temperature,
                )
            )
        per_assignment_crop_losses.append(torch.stack(swapped_losses).mean())

    return SwAVLossOutput(loss=torch.stack(per_assignment_crop_losses).mean(), assignment_outputs=assignment_outputs)
