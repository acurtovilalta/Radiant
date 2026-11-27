"""Anatomy-family-specific prototype banks for the embedding space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from radiant.data.anatomy import UNKNOWN_ANATOMY_LABEL

from .projection_head import normalize_embedding


@dataclass(slots=True)
class PrototypeBankConfig:
    """Configuration for anatomy-routed prototype banks."""

    embedding_dim: int
    prototypes_per_family: int | dict[str, int] = 64
    families: tuple[str, ...] = (UNKNOWN_ANATOMY_LABEL,)
    normalize_prototypes: bool = True


@dataclass(slots=True)
class PrototypeBankOutput:
    """Outputs for anatomy-routed prototype lookup."""

    family_names: list[str]
    family_ids: torch.Tensor
    family_prototype_counts: torch.Tensor
    family_logits: torch.Tensor
    family_assignments: torch.Tensor
    family_prototypes: torch.Tensor


class PrototypeBank(nn.Module):
    """Prototype bank for one anatomy family."""

    def __init__(
        self,
        embedding_dim: int,
        num_prototypes: int,
        normalize_prototypes: bool = True,
    ) -> None:
        super().__init__()
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive.")
        self.embedding_dim = int(embedding_dim)
        self.num_prototypes = int(num_prototypes)
        self.normalize_prototypes = bool(normalize_prototypes)
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.embedding_dim) * 0.02)

    def normalized_prototypes(self) -> torch.Tensor:
        """Return prototypes with optional L2 normalization."""
        if not self.normalize_prototypes:
            return self.prototypes
        return normalize_embedding(self.prototypes)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project embeddings onto prototypes and return logits and normalized weights."""
        prototypes = self.normalized_prototypes()
        logits = embeddings @ prototypes.t()
        assignments = torch.softmax(logits, dim=-1)
        return logits, assignments


class AnatomyPrototypeBanks(nn.Module):
    """One prototype bank per anatomy family."""

    def __init__(self, config: PrototypeBankConfig) -> None:
        super().__init__()
        families = tuple(config.families)
        if isinstance(config.prototypes_per_family, Mapping):
            if (not families) or (
                len(families) == 1
                and families[0] == UNKNOWN_ANATOMY_LABEL
                and UNKNOWN_ANATOMY_LABEL not in config.prototypes_per_family
            ):
                families = tuple(config.prototypes_per_family.keys())
        families = tuple(dict.fromkeys(families))
        if not families:
            families = (UNKNOWN_ANATOMY_LABEL,)
        self.config = config
        family_counts = self._resolve_family_counts(
            families=families,
            prototypes_per_family=config.prototypes_per_family,
        )
        self.family_counts = family_counts
        self._global_max_prototypes = max(family_counts.values())
        self._is_uniform_prototype_count = len(set(family_counts.values())) == 1
        self.family_to_id = {family: index for index, family in enumerate(families)}
        self.id_to_family = {family_id: family for family, family_id in self.family_to_id.items()}
        self.fallback_family = UNKNOWN_ANATOMY_LABEL if UNKNOWN_ANATOMY_LABEL in families else families[0]
        self.prototype_banks = nn.ModuleDict(
            {
                family: PrototypeBank(
                    embedding_dim=config.embedding_dim,
                    num_prototypes=family_counts[family],
                    normalize_prototypes=config.normalize_prototypes,
                )
                for family in families
            }
        )

    @staticmethod
    def _resolve_family_counts(
        families: tuple[str, ...],
        prototypes_per_family: int | Mapping[str, int],
    ) -> dict[str, int]:
        if isinstance(prototypes_per_family, int):
            return {family: int(prototypes_per_family) for family in families}

        counts: dict[str, int] = {}
        for family in families:
            if family not in prototypes_per_family:
                raise ValueError(f"Missing prototype count for family '{family}'.")
            counts[family] = int(prototypes_per_family[family])
        return counts

    def _normalize_family_name(self, family_name: str) -> str:
        if family_name in self.prototype_banks:
            return family_name
        return self.fallback_family

    def get_family_bank(self, family_name: str) -> PrototypeBank:
        """Return the bank for one anatomy family."""
        return self.prototype_banks[self._normalize_family_name(family_name)]

    def _stacked_prototypes(self, reference: torch.Tensor) -> torch.Tensor:
        """Stack every bank's normalized prototypes into one [F, global_max, d] tensor (padded with zeros)."""
        bank_protos = [bank.normalized_prototypes() for bank in self.prototype_banks.values()]
        if self._is_uniform_prototype_count:
            return torch.stack(bank_protos, dim=0)
        stacked = reference.new_zeros(len(bank_protos), self._global_max_prototypes, self.config.embedding_dim)
        for index, protos in enumerate(bank_protos):
            stacked[index, : protos.shape[0]] = protos
        return stacked

    def forward(
        self,
        embeddings: torch.Tensor,
        anatomy_family: list[str] | tuple[str, ...] | None = None,
        anatomy_family_ids: torch.Tensor | None = None,
    ) -> PrototypeBankOutput:
        """Route each embedding to its anatomy family's prototype bank (batched over samples)."""
        if anatomy_family is None and anatomy_family_ids is None:
            raise ValueError("Either anatomy_family or anatomy_family_ids must be provided.")

        if anatomy_family is None:
            anatomy_family = [
                self.id_to_family.get(int(family_id), self.fallback_family)
                for family_id in anatomy_family_ids.detach().cpu().tolist()
            ]

        family_names = [self._normalize_family_name(name) for name in anatomy_family]
        family_ids = torch.tensor(
            [self.family_to_id[name] for name in family_names],
            dtype=torch.long,
            device=embeddings.device,
        )
        # Pad only to the largest bank among the families present in this batch.
        prototype_counts = [self.family_counts[name] for name in family_names]
        max_prototypes = max(prototype_counts)
        counts_tensor = torch.tensor(prototype_counts, dtype=torch.long, device=embeddings.device)

        stacked_prototypes = self._stacked_prototypes(embeddings)  # [F, global_max, d]
        per_sample_prototypes = stacked_prototypes.index_select(0, family_ids)[:, :max_prototypes, :]
        # Under autocast bmm returns fp16; cast back to the embedding dtype, which the
        # Sinkhorn / assignment code downstream expects.
        family_logits = torch.bmm(per_sample_prototypes, embeddings.unsqueeze(-1)).squeeze(-1)
        family_logits = family_logits.to(embeddings.dtype)

        if self._is_uniform_prototype_count:
            # Every routed family has exactly `max_prototypes` real entries, so
            # nothing is padding -- skip the mask build entirely.
            family_assignments = torch.softmax(family_logits, dim=-1)
        else:
            valid_mask = (
                torch.arange(max_prototypes, device=embeddings.device).unsqueeze(0)
                < counts_tensor.unsqueeze(1)
            )
            family_logits = family_logits.masked_fill(~valid_mask, float("-inf"))
            family_assignments = torch.softmax(family_logits, dim=-1) * valid_mask.to(embeddings.dtype)

        return PrototypeBankOutput(
            family_names=family_names,
            family_ids=family_ids,
            family_prototype_counts=counts_tensor,
            family_logits=family_logits,
            family_assignments=family_assignments,
            family_prototypes=per_sample_prototypes,
        )
