"""Prepare the fixed benchmark scores and covariates used for the real-data tables."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from run import HERE, OUTPUT

sys.path.insert(0, str(HERE.parent))
from _benchmark_data import load_dataset

SKIN_CLASSES = ["psoriasis", "seboreic dermatitis", "lichen planus",
                "pityriasis rosea", "cronic dermatitis", "pityriasis rubra pilaris"]


def prepare(data_dir):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    design = json.loads((HERE / "design.json").read_text())
    with np.load(HERE.parent / "tree_forest_inputs.npz") as saved:
        for name, setting in design.items():
            if name == "legislator":
                # At most six splits are possible with seven leaves. The center of
                # each depth-six bin routes identically to the original coordinate.
                with np.load(HERE / "legislator_scores.npz") as archive:
                    x = -1 + (archive["bins"].astype(float) + 0.5) / 32
                    scores = archive["scores"]
            elif name == "skin":
                data = HERE.parents[1] / "data"
                frame = pd.read_csv(data / "dermatology.data", header=None, na_values="?")
                age = frame.iloc[:, 33].astype(float)
                frame.iloc[:, 33] = age.fillna(age.mean())
                x = frame.iloc[:, list(range(11)) + [33]].to_numpy(dtype=float)
                labels = frame.iloc[:, 34].to_numpy(dtype=int) - 1
                outputs = json.loads((data / "dermdata.json").read_text())
                scores = 1 - np.array([float(row[SKIN_CLASSES[y]]) for row, y in zip(outputs, labels)])
            else:
                key = name + "_000__"
                features, _ = load_dataset(name, data_dir, int(saved[key+"seeds"][0]))
                digest = hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
                if digest != str(saved[key+"feature_sha256"]):
                    raise ValueError(f"{name}: features differ from the original benchmark")
                keep = np.r_[saved[key+"calibration_indices"], saved[key+"test_indices"]]
                x = features[keep]
                scores = np.r_[saved[key+"calibration_scores"], saved[key+"test_scores"]]
            if len(x) != setting["pool"] or len(scores) != len(x):
                raise ValueError(f"{name}: unexpected number of observations")
            np.savez(OUTPUT / f"{name}.npz", x=x, s=scores, bounds=setting["bounds"],
                     n=setting["n"], m=setting["m"], k=setting["K"],
                     depth=24 if name in ("skin", "legislator") else 8)
    subprocess.run(["c++", "-O3", "-shared", "-fPIC", "-std=c++17",
                    str(HERE / "trees.cpp"), "-o", str(OUTPUT / "trees.so")], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    prepare(parser.parse_args().data_dir)
