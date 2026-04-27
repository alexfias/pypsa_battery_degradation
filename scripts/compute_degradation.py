from __future__ import annotations

import argparse
from pathlib import Path

import pypsa

from pypsa_battery_degradation import compute_degradation_from_network


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("network", type=Path, help="Path to solved PyPSA network .nc file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/degradation_results.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--carrier",
        default="battery",
        help="Storage carrier to evaluate. Use 'all' for all storage units.",
    )
    args = parser.parse_args()

    network = pypsa.Network(args.network)

    carrier = None if args.carrier.lower() == "all" else args.carrier

    results = compute_degradation_from_network(
        network,
        temperature=None,
        storage_carrier=carrier,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    print(f"Saved degradation results to {args.output}")
    print(results.head())


if __name__ == "__main__":
    main()