#!/usr/bin/env python3
"""Compare the all-direction dyadic tree with a forest using random direction subsets."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from conformal_tree import ConformalForest, choose_leaf_cap
from conformal_tree.forest import aggregate_offsets
from ._benchmark_data import DATASETS, load_dataset

HERE = Path(__file__).resolve().parent
ALPHA, M, DEPTH, ETA, TREES = 0.1, 20, 8, 0.05, 100


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def run_trial(dataset, repetition, args):
    key = f"{dataset}_{repetition:03d}__"
    with np.load(HERE / "tree_forest_inputs.npz", allow_pickle=False) as saved:
        values = {name: saved[key+name] for name in
                  ("calibration_scores", "test_scores", "train_indices", "calibration_indices",
                   "test_indices", "seeds", "uniform_thresholds", "feature_sha256", "response_sha256")}
    seeds = [int(s) for s in values["seeds"]]
    X, y = load_dataset(dataset, args.data_dir, seeds[0])
    if array_hash(X) != str(values["feature_sha256"]) or array_hash(y) != str(values["response_sha256"]):
        raise ValueError(f"{dataset}: data or preprocessing differ from the reference benchmark")
    train, cal, test = [values[k] for k in ("train_indices", "calibration_indices", "test_indices")]
    scores, residual = values["calibration_scores"], values["test_scores"]
    if args.refit_model:
        scale = float(np.mean(np.abs(y[train])))
        y = y/scale
        model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_features="sqrt",
                                      random_state=seeds[3], n_jobs=1)
        model.fit(X[train], y[train])
        scores = np.abs(y[cal]-model.predict(X[cal]))
        residual = np.abs(y[test]-model.predict(X[test]))
    domain = np.column_stack((X[train].min(axis=0), X[train].max(axis=0)))
    tree_scores = scores+np.random.default_rng(seeds[4]).uniform(0, 1e-12, len(scores))
    d = X.shape[1]
    p = math.ceil(math.sqrt(d)) if args.directions == "sqrt" else max(1, math.ceil(math.log2(d)))
    cap = choose_leaf_cap(len(scores), M, p, DATASETS[dataset][1], 0.1)
    rows, predictions = [], dict(test_scores=residual)
    configurations = (("all_d_single", d, 1), (f"{args.directions}_{args.aggregation}_forest", p, TREES))
    for method, q, trees in configurations:
        forest = ConformalForest(M, cap, q, trees, DEPTH, ETA, seeds[5])
        forest.fit(X[cal], scores, domain, tree_scores)
        matrix = forest.predict_component_offsets(X[test], ALPHA)
        if method == "all_d_single":
            offsets = matrix[0]
        else:
            threshold = values["uniform_thresholds"] if args.aggregation == "uniform" else 0.5
            offsets = aggregate_offsets(matrix, threshold)
        if not np.isfinite(offsets).all():
            raise ValueError(f"Unexpected infinite interval in {dataset}, {method}")
        rows.append(dict(dataset=dataset, label=DATASETS[dataset][0], method=method,
                         repetition=repetition, n=len(scores), d=d, p=q, kmax=cap, delta=forest.delta_,
                         width=float(np.mean(2*offsets)), coverage=float(np.mean(residual <= offsets)),
                         isl=float(np.mean(2*offsets+20*np.maximum(residual-offsets, 0)))))
        predictions[method] = offsets
    destination = args.output / "trials" / f"{dataset}_{repetition:03d}"
    np.savez_compressed(destination.with_suffix(".npz"), **predictions)
    save_json(destination.with_suffix(".json"), rows)
    return dataset, repetition


def summarize(args):
    rows = []
    for dataset in args.datasets:
        for rep in range(args.repetitions):
            path = args.output / "trials" / f"{dataset}_{rep:03d}.json"
            if path.exists():
                rows.extend(json.loads(path.read_text()))
    summary = []
    methods = ("all_d_single", f"{args.directions}_{args.aggregation}_forest")
    for method in methods:
        for dataset in args.datasets:
            group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            if not group:
                continue
            result = {k: group[0][k] for k in ("dataset", "label", "method", "n", "d", "p", "kmax", "delta")}
            result["trials"] = len(group)
            for name in ("width", "isl", "coverage"):
                result[name] = statistics.mean(r[name] for r in group)
            summary.append(result)
    save_json(args.output / "summary.json", summary)
    tables = []
    for method in methods:
        lines = [method, "", "| Dataset | Width | ISL | Coverage | delta |",
                 "|---|---:|---:|---:|---:|"]
        for r in summary:
            if r["method"] == method:
                cells = [r["label"], *[f"{r[k]:.3f}" for k in ("width", "isl", "coverage", "delta")]]
                lines.append("| "+" | ".join(cells)+" |")
        tables.append("\n".join(lines))
    (args.output / "tables.txt").write_text("\n\n".join(tables)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="cqr-comparison/datasets directory")
    parser.add_argument("--output", type=Path, default=HERE / "output/tree_forest")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--directions", choices=("sqrt", "log2"), default="sqrt")
    parser.add_argument("--aggregation", choices=("uniform", "majority"), default="uniform")
    parser.add_argument("--refit-model", action="store_true", help="refit black-box models instead of using saved scores")
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 20 or args.jobs < 1:
        parser.error("Require 1 <= repetitions <= 20 and jobs >= 1")
    args.data_dir, args.output = args.data_dir.resolve(), args.output.resolve()
    reference = json.loads((HERE / "tree_forest_reference.json").read_text())
    archive_hash = hashlib.sha256((HERE / "tree_forest_inputs.npz").read_bytes()).hexdigest()
    if archive_hash != reference["input_archive_sha256"]:
        raise ValueError("Saved input archive does not match tree_forest_reference.json")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "trials").mkdir(exist_ok=True)
    import conformal_tree
    package = Path(conformal_tree.__file__).parent
    settings = dict(directions=args.directions, aggregation=args.aggregation, refit_model=args.refit_model,
                    archive_sha256=archive_hash, alpha=ALPHA, m=M, max_depth=DEPTH, eta=ETA,
                    trees=TREES, forest_delta_budget=.1,
                    code_sha256={str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (package / "forest.py", package / "_utils/fast_dyadic.py")},
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    loader_sha256=hashlib.sha256((HERE / "_benchmark_data.py").read_bytes()).hexdigest())
    settingpath = args.output / "settings.json"
    if settingpath.exists() and json.loads(settingpath.read_text()) != settings:
        raise ValueError("Output directory contains different settings or code. Choose a new --output directory.")
    save_json(settingpath, settings)
    tasks = []
    for dataset in args.datasets:
        for rep in range(args.repetitions):
            checkpoint = args.output / "trials" / f"{dataset}_{rep:03d}.json"
            if checkpoint.exists():
                print(f"SKIP {dataset} trial {rep+1}", flush=True)
            else:
                tasks.append((dataset, rep))
    errors = []
    started = time.perf_counter()

    def record_error(dataset, rep, exc):
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        errors.append(dict(dataset=dataset, repetition=rep, error=repr(exc)))
        save_json(args.output / "errors.json", errors)

    if args.jobs == 1:
        for i, (dataset, rep) in enumerate(tasks, 1):
            try:
                run_trial(dataset, rep, args)
                print(f"DONE {i}/{len(tasks)} {dataset} trial {rep+1}, elapsed {time.perf_counter()-started:.0f}s", flush=True)
            except Exception as exc:
                record_error(dataset, rep, exc)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_trial, name, rep, args): (name, rep) for name, rep in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                dataset, rep = futures[future]
                try:
                    future.result()
                    print(f"DONE {i}/{len(tasks)} {dataset} trial {rep+1}, elapsed {time.perf_counter()-started:.0f}s", flush=True)
                except Exception as exc:
                    record_error(dataset, rep, exc)
    summary = summarize(args)
    if errors:
        raise RuntimeError(f"{len(errors)} trials failed. Fix the problem and rerun to resume.")
    if args.repetitions == 20 and args.directions == "sqrt" and args.aggregation == "uniform" and not args.refit_model:
        expected = {(r["dataset"], r["method"]): r for r in reference["summary"]}
        for row in summary:
            ref = expected[(row["dataset"], row["method"])]
            for name in ("width", "isl", "coverage", "delta"):
                if not math.isclose(row[name], ref[name], rel_tol=0, abs_tol=1e-12):
                    raise RuntimeError(f"Reference mismatch: {row['dataset']}, {row['method']}, {name}")
        print("All reported metrics match the reference benchmark before rounding.", flush=True)
    print((args.output / "tables.txt").read_text(), flush=True)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
