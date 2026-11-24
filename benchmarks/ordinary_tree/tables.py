"""Print the two ordinary-tree simulation tables from archived or regenerated results."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "output"


def print_tables(split_file, coverage_file):
    with np.load(split_file) as data:
        rates = 100 * data["outcomes"][..., 0].mean(axis=(0, 1))

    print("Data 2: change in query leaf, n=499")
    print("| Candidate cuts | Range reduction | Squared-error reduction |")
    print("|---|---:|---:|")
    for label, start in [("Dyadic midpoint", 0), ("All admissible cuts", 2)]:
        print(f"| {label} | {rates[start]:.2f}% | {rates[start+1]:.2f}% |")

    print("\n| Dataset | Leaf | Selected | Penalty (pp) | Lower bound | Coverage | 95% interval |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for ds, result in json.loads(coverage_file.read_text()).items():
        cells = [r for r in result["cells"] if r["cell"] in result["selected"]]
        for r in sorted(cells, key=lambda r: (r["interval"][0], -r["interval"][1])):
            left, right = r["interval"]
            leaf = f"{'[' if left == 0 else '('}{left:g},{right:g}]"
            lo, hi = r["coverage_ci"]
            print(f"| Data {ds} | {leaf} | {100*r['rho']:.2f}% | {100*r['penalty']:.2f} | "
                  f"{100*r['plugin']:.2f}% | {100*r['coverage']:.2f}% | "
                  f"{100*lo:.2f}–{100*hi:.2f}% |")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rerun", action="store_true", help="use results from the latest simulation")
    rerun = p.parse_args().rerun
    split_dir = OUTPUT / "ordinary_tree" if rerun else HERE
    coverage_dir = OUTPUT / "ordinary_selected_leaf" if rerun else HERE
    print_tables(split_dir / "data2_split_comparison.npz", coverage_dir / "results.json")
