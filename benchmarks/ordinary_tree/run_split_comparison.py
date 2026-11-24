"""Data 2 query-leaf stability at n=499 with the test observation excluded."""
import ctypes
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "cut_criterion_stability"
sys.path.insert(0, str(BASE))
from run import draw, predictor, SEED, NAMES


def simulate(compare):
    outcomes = np.empty((8, 1500, 4, 3), dtype=np.uint8)
    result = np.empty((4, 8))
    for model in range(8):
        scale, predict = predictor(2, model)
        rng = np.random.default_rng([SEED, 1, 2, model, 499])
        for start in range(0, 1500, 48):
            x, y = draw(rng, 2, (min(48, 1500-start), 500))
            scores = abs(y/scale - predict(x)) + rng.uniform(0, 1e-12, x.shape)
            order = np.argsort(x, axis=1)
            xs, ss = [np.take_along_axis(v, order, axis=1) for v in (x, scores)]
            for i, query in enumerate((order == 499).argmax(axis=1)):
                compare(xs[i], ss[i], 500, query, 8, 0., result)
                outcomes[model, start+i] = result[:, :3]
        print(f"Completed predictor {model+1}/8", flush=True)
    return outcomes


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        lib = Path(tmp) / "ordinary.so"
        subprocess.run(["c++", "-std=c++17", "-O3", "-shared", "-fPIC", "-DORDINARY_TREE",
                        str(BASE / "tree.cpp"), "-o", str(lib)], check=True)
        compare = ctypes.CDLL(str(lib)).compare
        doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        compare.argtypes = [doubles, doubles, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, doubles]
        compare.restype = None
        outcomes = simulate(compare)
    dest = HERE.parent / "output" / "ordinary_tree"
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "data2_split_comparison.npz", outcomes=outcomes, methods=NAMES,
                        metrics=["query_leaf_changed", "partition_changed", "ordinary_leaves"])
    for name, rate in zip(NAMES, outcomes[:, :, :, 0].mean(axis=(0, 1))):
        print(f"{name}: {100*rate:.2f}%")
