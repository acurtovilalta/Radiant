"""Multiple instance contrastive loss for X-rays and MRI projection bags."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def projection_mil_contrastive_loss(
    z_x: torch.Tensor,
    z_m_proj: torch.Tensor,
    xray_families: list[str],
    mri_families: list[str],
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MIL contrastive loss from X-ray anchors to MRI projection bags (Eq. 4 of the paper).

    For X-ray i, the positive bag is every projection k of every MRI j in the batch from the same
    anatomy family; all other projections are negatives:
        L = -mean_i log( sum_{(j,k) in P_i} exp(s_ijk) / sum_{j,k} exp(s_ijk) ),  s_ijk = <z_i, r_jk> / tau.
    The mean is over all B_X X-rays of the per-GPU batch, as in Eq. 4. An X-ray without a same-family
    MRI in the batch has an empty positive bag; it contributes 0 to the sum but still counts in B_X.
    (With family-synchronised batches this is rare.)
    """
    if float(temperature) <= 0.0:
        raise ValueError(f"projection_mil_temperature must be > 0.0, got {temperature}.")
    if z_x.ndim != 2:
        raise ValueError(f"z_x must have shape [B_x, d], got {tuple(z_x.shape)}.")
    if z_m_proj.ndim != 3:
        raise ValueError(f"z_m_proj must have shape [B_m, K, d], got {tuple(z_m_proj.shape)}.")

    zero = z_m_proj.sum() * 0.0 + z_x.sum() * 0.0
    zero_stat = zero.detach()
    if z_x.shape[0] == 0 or z_m_proj.shape[0] == 0:
        return zero, {"valid_xray_anchors": zero_stat, "positive_fraction": zero_stat}

    z_x_norm = F.normalize(z_x, dim=-1, eps=1e-6)
    z_m_proj_norm = F.normalize(z_m_proj, dim=-1, eps=1e-6)
    sim = torch.einsum("bd,mkd->bmk", z_x_norm, z_m_proj_norm) / float(temperature)

    # Map family names to small integer ids so the positive mask is one vectorized comparison.
    xray_labels = [str(label) for label in xray_families[: int(z_x.shape[0])]]
    mri_labels = [str(label) for label in mri_families[: int(z_m_proj.shape[0])]]
    label_to_id = {label: index for index, label in enumerate(dict.fromkeys(xray_labels + mri_labels))}
    x_ids = torch.tensor([label_to_id[label] for label in xray_labels], device=sim.device)
    m_ids = torch.tensor([label_to_id[label] for label in mri_labels], device=sim.device)
    positive_mask = x_ids.unsqueeze(1) == m_ids.unsqueeze(0)  # [B_x, B_m]
    if positive_mask.shape != sim.shape[:2]:
        raise ValueError(
            "projection MIL family labels must match batch dimensions. "
            f"Got positive_mask={tuple(positive_mask.shape)}, sim={tuple(sim.shape)}."
        )

    positive_fraction = positive_mask.to(torch.float32).mean()
    valid_xray = positive_mask.any(dim=1)
    if bool(valid_xray.any().item()):
        x_sim = sim[valid_xray]
        x_mask = positive_mask[valid_xray].unsqueeze(-1).expand_as(x_sim)
        numerator = torch.logsumexp(x_sim.masked_fill(~x_mask, float("-inf")).reshape(x_sim.shape[0], -1), dim=-1)
        denominator = torch.logsumexp(x_sim.reshape(x_sim.shape[0], -1), dim=-1)
        # Sum over anchors with a positive bag, divided by all B_X anchors (Eq. 4).
        loss = -(numerator - denominator).sum() / float(z_x.shape[0])
    else:
        loss = zero

    return loss, {
        "valid_xray_anchors": valid_xray.to(torch.float32).sum(),
        "positive_fraction": positive_fraction,
    }

def xray_family_mil_loss(
    z_x: torch.Tensor,
    xray_families: list[str],
    temperature: float,
) -> torch.Tensor:
    """X-ray-only counterpart of the projection MIL loss (control C1).

    The same objective as `projection_mil_contrastive_loss` with the MRI projection bags replaced by
    X-ray embeddings: for X-ray i the positive bag is every other X-ray in the batch from the same
    anatomy family, and X-rays from other families are the negatives. The anchor itself is excluded
    from both sums. As in Eq. 4, the mean is over all B_X X-rays; an anchor without a same-family
    partner in the batch contributes 0 but still counts in B_X.
    """
    if float(temperature) <= 0.0:
        raise ValueError(f"projection_mil_temperature must be > 0.0, got {temperature}.")
    if z_x.ndim != 2:
        raise ValueError(f"z_x must have shape [B_x, d], got {tuple(z_x.shape)}.")
    batch_size = int(z_x.shape[0])
    if batch_size < 2:
        return z_x.sum() * 0.0

    z_norm = F.normalize(z_x, dim=-1, eps=1e-6)
    sim = (z_norm @ z_norm.t()) / float(temperature)

    if len(xray_families) < batch_size:
        raise ValueError(f"Got {len(xray_families)} anatomy families for {batch_size} X-ray embeddings.")
    labels = [str(label) for label in xray_families[:batch_size]]
    label_to_id = {label: index for index, label in enumerate(dict.fromkeys(labels))}
    ids = torch.tensor([label_to_id[label] for label in labels], device=sim.device)
    not_self = ~torch.eye(batch_size, dtype=torch.bool, device=sim.device)
    positive_mask = (ids.unsqueeze(1) == ids.unsqueeze(0)) & not_self

    valid = positive_mask.any(dim=1)
    if not bool(valid.any().item()):
        return z_x.sum() * 0.0
    sim, positive_mask, not_self = sim[valid], positive_mask[valid], not_self[valid]
    numerator = torch.logsumexp(sim.masked_fill(~positive_mask, float("-inf")), dim=-1)
    denominator = torch.logsumexp(sim.masked_fill(~not_self, float("-inf")), dim=-1)
    return -(numerator - denominator).sum() / float(batch_size)
