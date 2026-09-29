"""Train source-family LMD meta regression/classification on adapted LMD-core features."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .features import build_frame_features, feature_names, iter_frames

WEATHERS = ("clean", "fog", "rain", "snow")


def _read_meta(root):
    root = Path(root).resolve()
    meta = json.loads((root / "candidate_audit.json").read_text(encoding="utf-8"))
    if meta.get("split") != "train":
        raise ValueError("LMD training input must be an official-train extraction")
    if not meta.get("full_split"):
        print("WARNING: training extraction is a smoke/subset cache", flush=True)
    return root, meta


def _reservoir(root, meta, per_weather, threshold, seed):
    blocks = []
    for weather_index, weather in enumerate(WEATHERS):
        order = meta["conditions"][weather]["box_order"]
        x_res, y_res, g_res, seen = [], [], [], 0
        local_rng = np.random.default_rng(seed + weather_index)
        for sample, _, rows in iter_frames(root, weather):
            x, y, _ = build_frame_features(rows, order, threshold)
            scene = str(meta["conditions"][weather]["scene_map"][str(sample)])
            for i in range(len(x)):
                seen += 1
                if per_weather <= 0 or len(x_res) < per_weather:
                    x_res.append(x[i].copy())
                    y_res.append(float(y[i]))
                    g_res.append(scene)
                else:
                    j = int(local_rng.integers(0, seen))
                    if j < per_weather:
                        x_res[j] = x[i].copy()
                        y_res[j] = float(y[i])
                        g_res[j] = scene
        if not x_res:
            raise ValueError(f"No training candidates for {weather}")
        blocks.append((np.asarray(x_res, dtype=np.float32),
                       np.asarray(y_res, dtype=np.float32),
                       np.asarray([f"scene_{g}" for g in g_res], dtype=object),
                       weather, seen))
        print(f"{weather}: sampled {len(x_res)} from {seen} candidates", flush=True)
    x = np.concatenate([b[0] for b in blocks], axis=0)
    y = np.concatenate([b[1] for b in blocks], axis=0)
    groups = np.concatenate([b[2] for b in blocks], axis=0)
    return x, y, groups, {b[3]: {"seen": b[4], "sampled": len(b[0])} for b in blocks}


def _regressors(seed, jobs):
    return {
        "ridge_reg": Ridge(alpha=10.0),
        "random_forest_reg": RandomForestRegressor(
            n_estimators=100, max_depth=18, random_state=seed, n_jobs=jobs),
        "gb_reg": GradientBoostingRegressor(
            loss="squared_error", learning_rate=0.08, n_estimators=80,
            criterion="friedman_mse", max_depth=8, subsample=0.8,
            random_state=seed),
    }


def _classifiers(seed, jobs):
    return {
        "logistic": LogisticRegression(C=0.1, solver="lbfgs", max_iter=5000,
                                        class_weight="balanced", random_state=seed),
        "random_forest_cls": RandomForestClassifier(
            n_estimators=100, max_depth=20, criterion="gini",
            class_weight="balanced", random_state=seed, n_jobs=jobs),
        "gb_cls": GradientBoostingClassifier(
            loss="exponential", learning_rate=0.3, n_estimators=100,
            criterion="friedman_mse", max_depth=8, subsample=0.8,
            random_state=seed),
    }


def _reg_metrics(y, pred):
    return {"r2": float(r2_score(y, pred)),
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5)}


def _cls_metrics(y, prob):
    return {"average_precision": float(average_precision_score(y, prob)),
            "roc_auc": float(roc_auc_score(y, prob))
            if len(np.unique(y)) == 2 else None}


def run(args):
    root, meta = _read_meta(args.train_root)
    x, y, groups, sampling = _reservoir(
        root, meta, args.max_rows_per_weather, args.proposal_iou_threshold, args.seed)
    split = GroupShuffleSplit(n_splits=1, test_size=args.holdout_fraction,
                              random_state=args.seed)
    train_idx, val_idx = next(split.split(x, y, groups))
    if set(groups[train_idx]) & set(groups[val_idx]):
        raise AssertionError("Scene leakage in train-only LMD model selection")
    scaler = StandardScaler().fit(x[train_idx])
    x_train = scaler.transform(x[train_idx])
    x_val = scaler.transform(x[val_idx])

    regression = {}
    best_name, best_score = None, -np.inf
    for name, model in _regressors(args.seed, args.jobs).items():
        model.fit(x_train, y[train_idx])
        pred = model.predict(x_val)
        metrics = _reg_metrics(y[val_idx], pred)
        regression[name] = metrics
        print(name, metrics, flush=True)
        if metrics["r2"] > best_score:
            best_name, best_score = name, metrics["r2"]
    best_reg = clone(_regressors(args.seed, args.jobs)[best_name])
    x_all = scaler.transform(x)
    best_reg.fit(x_all, y)

    classification = {}
    best_cls_name = best_cls = None
    if not args.skip_classification:
        cls = (y >= args.classification_iou_threshold).astype(np.int8)
        best_ap = -np.inf
        for name, model in _classifiers(args.seed, args.jobs).items():
            model.fit(x_train, cls[train_idx])
            prob = model.predict_proba(x_val)[:, 1]
            metrics = _cls_metrics(cls[val_idx], prob)
            classification[name] = metrics
            print(name, metrics, flush=True)
            if metrics["average_precision"] > best_ap:
                best_cls_name, best_ap = name, metrics["average_precision"]
        best_cls = clone(_classifiers(args.seed, args.jobs)[best_cls_name])
        best_cls.fit(x_all, cls)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = {
        "schema": 1,
        "method": "adapted_lmd_core",
        "feature_names": feature_names(),
        "proposal_iou_threshold": float(args.proposal_iou_threshold),
        "classification_iou_threshold": float(args.classification_iou_threshold),
        "scaler": scaler,
        "regressor_name": best_name,
        "regressor": best_reg,
        "classifier_name": best_cls_name,
        "classifier": best_cls,
        "f_checkpoint_sha256": meta["arm_sha256"]["F"],
        "frontend_sha256": meta["frontend_sha256"],
        "v3_checkpoint_sha256": meta["v3_checkpoint_sha256"],
    }
    with (output / "model.pkl").open("wb") as stream:
        pickle.dump(package, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "protocol": {
            "method": "adapted_lmd_core",
            "training_root": str(root),
            "weather_mixed_single_model": True,
            "proposal_iou_threshold": args.proposal_iou_threshold,
            "classification_iou_threshold": args.classification_iou_threshold,
            "max_rows_per_weather": args.max_rows_per_weather,
            "holdout_fraction": args.holdout_fraction,
            "grouping": "OPV2V scene; same scene across four weather conditions stays together",
            "model_selection_uses_validation_split": False,
            "source_family_note": (
                "model families and parameter values are drawn from committed "
                "MetaDetect3D configs; modern sklearn names replace deprecated aliases"),
        },
        "sampling": sampling,
        "rows": {"total": int(len(x)), "train": int(len(train_idx)),
                 "holdout": int(len(val_idx))},
        "selected_regressor": best_name,
        "regression_holdout": regression,
        "selected_classifier": best_cls_name,
        "classification_holdout": classification,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("LMD TRAINING COMPLETE:", output / "model.pkl", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--proposal-iou-threshold", type=float, default=0.2)
    parser.add_argument("--classification-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-rows-per-weather", type=int, default=50000,
                        help="Reservoir cap after scanning full split; <=0 keeps all")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--skip-classification", action="store_true")
    args = parser.parse_args()
    if not 0 < args.holdout_fraction < 0.5:
        parser.error("--holdout-fraction must be in (0,0.5)")
    run(args)


if __name__ == "__main__":
    main()
