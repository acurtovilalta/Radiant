"""CLI entry point for staged multimodal SwAV training."""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*cuda\.cudart module is deprecated.*",
    category=FutureWarning,
)

from radiant.training import train_from_config
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RADIANT with a staged schedule.")
    parser.add_argument("--config", required=True, help="Path to YAML trainer config.")
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Enable torch autograd anomaly detection (default: disabled).",
    )
    args = parser.parse_args()
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    summary = train_from_config(args.config)
    print(summary)


if __name__ == "__main__":
    main()
