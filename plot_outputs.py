"""
Quick plotting helpers for the load-flow assignment deliverables.
Requires matplotlib: install with
    & ".venv/Scripts/python.exe" -m pip install matplotlib

Generates PNGs in the repo root using existing CSV outputs:
- nr_results_bus.csv: base voltage profile
- sens_results_cases.csv: voltage profiles under +/-10% load changes
- sens_results_ranking.csv: sensitivity ranking bars
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required; install it with pip install matplotlib") from exc

ROOT = Path(__file__).parent

# -------------------------- Loaders --------------------------

def load_nr_bus(path: Path) -> List[Tuple[int, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append((int(r["bus"]), float(r["Vmag_pu"])))
    rows.sort(key=lambda x: x[0])
    return rows


def load_sens_cases(path: Path) -> Dict[int, Dict[float, List[Tuple[int, float]]]]:
    """Return mapping: varied_bus -> delta_pct -> list of (bus, Vmag)."""
    data: Dict[int, Dict[float, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            vb = int(r["varied_bus"])
            d = float(r["delta_pct"])
            bus = int(r["bus"])
            vmag = float(r["Vmag_pu"])
            data[vb][d].append((bus, vmag))
    # sort by bus for consistency
    for vb in data:
        for d in data[vb]:
            data[vb][d].sort(key=lambda x: x[0])
    return data


def load_sens_ranking(path: Path) -> List[Tuple[int, float, int, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(
                (
                    int(r["varied_bus"]),
                    float(r["max_std_any_bus"]),
                    int(r["max_std_at_bus"]),
                    float(r["mean_std_all_buses"]),
                )
            )
    rows.sort(key=lambda x: x[0])
    return rows

# -------------------------- Plots --------------------------

def plot_base_voltage(profile: List[Tuple[int, float]], out_path: Path) -> None:
    buses = [b for b, _ in profile]
    mags = [v for _, v in profile]
    plt.figure(figsize=(6, 3.5))
    plt.plot(buses, mags, marker="o", label="NR (Python)")
    plt.xlabel("Bus")
    plt.ylabel("|V| (pu)")
    plt.title("Base Voltage Profile (NR)")
    plt.grid(True, linestyle=":", linewidth=0.6)
    plt.ylim(0.94, 1.06)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_sensitivity_profiles(data: Dict[int, Dict[float, List[Tuple[int, float]]]], out_dir: Path) -> None:
    for vb, delta_map in data.items():
        plt.figure(figsize=(6.5, 3.5))
        for delta, rows in sorted(delta_map.items()):
            buses = [b for b, _ in rows]
            mags = [v for _, v in rows]
            plt.plot(buses, mags, marker="o", label=f"ΔP,Q={delta:.0f}%")
        plt.xlabel("Bus")
        plt.ylabel("|V| (pu)")
        plt.title(f"Voltage Profiles vs Load Variation at Bus {vb}")
        plt.grid(True, linestyle=":", linewidth=0.6)
        plt.ylim(0.93, 1.07)
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"sens_profile_bus{vb}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_sensitivity_ranking(rows: List[Tuple[int, float, int, float]], out_path: Path) -> None:
    varied = [r[0] for r in rows]
    max_std = [r[1] for r in rows]
    mean_std = [r[3] for r in rows]
    x = range(len(varied))
    width = 0.35

    plt.figure(figsize=(6.5, 3.5))
    plt.bar([i - width / 2 for i in x], max_std, width=width, label="Max std across buses")
    plt.bar([i + width / 2 for i in x], mean_std, width=width, label="Mean std across buses")
    plt.xticks(list(x), [str(v) for v in varied])
    plt.xlabel("Varied load bus")
    plt.ylabel("Std of |V| (pu)")
    plt.title("Voltage Sensitivity Ranking")
    plt.grid(True, axis="y", linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -------------------------- Main --------------------------

def main() -> None:
    nr_bus_csv = ROOT / "nr_results_bus.csv"
    sens_cases_csv = ROOT / "sens_results_cases.csv"
    sens_rank_csv = ROOT / "sens_results_ranking.csv"

    if not nr_bus_csv.exists():
        raise SystemExit("nr_results_bus.csv not found. Run write_results('nr_results') first.")
    if not sens_cases_csv.exists() or not sens_rank_csv.exists():
        raise SystemExit("Sensitivity CSVs not found. Run write_sensitivity_results('sens_results') first.")

    out_dir = ROOT

    # Base profile
    base_profile = load_nr_bus(nr_bus_csv)
    plot_base_voltage(base_profile, out_dir / "plot_base_voltage.png")

    # Sensitivity profiles
    sens_data = load_sens_cases(sens_cases_csv)
    plot_sensitivity_profiles(sens_data, out_dir)

    # Sensitivity ranking
    rank_rows = load_sens_ranking(sens_rank_csv)
    plot_sensitivity_ranking(rank_rows, out_dir / "plot_sensitivity_ranking.png")

    print("Saved plots to:")
    print(" - plot_base_voltage.png")
    for vb in sorted(sens_data):
        print(f" - sens_profile_bus{vb}.png")
    print(" - plot_sensitivity_ranking.png")


if __name__ == "__main__":
    main()
