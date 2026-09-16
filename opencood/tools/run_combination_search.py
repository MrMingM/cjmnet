#!/usr/bin/env python3
"""Sequential low-fidelity search for robust Where2comm combinations.

The search deliberately uses only source-derived development scenes.  OPV2V-W
must remain untouched until a configuration has been selected and retrained.

The first eight runs form an L8 screen over four binary factors:

* augmentation: V2X-DGW AWA or Physics-Fog
* prediction KD + communication preservation
* weather feature adapter
* agent/fusion alignment + agent contrast

Four focused follow-up runs complete the Physics-Fog KD/adapter/DG factorial.
All runs start from the same clean checkpoint and use the same random seeds.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


L8_CANDIDATES = [
    # name, augmentation, kd, adapter, dg
    ("l8_00_awa_plain", "v2x_dgw_awa", False, False, False),
    ("l8_01_awa_adapter_dg", "v2x_dgw_awa", False, True, True),
    ("l8_02_awa_kd_dg", "v2x_dgw_awa", True, False, True),
    ("l8_03_awa_kd_adapter", "v2x_dgw_awa", True, True, False),
    ("l8_04_fog_dg", "physics_fog", False, False, True),
    ("l8_05_fog_adapter", "physics_fog", False, True, False),
    ("l8_06_fog_kd", "physics_fog", True, False, False),
    ("l8_07_fog_kd_adapter_dg", "physics_fog", True, True, True),
    # Focused completion of the Physics-Fog 2^3 module factorial.  Together
    # with l8_04--l8_07 these four runs cover every KD/adapter/DG combination.
    ("f2_08_fog_plain", "physics_fog", False, False, False),
    ("f2_09_fog_kd_adapter", "physics_fog", True, True, False),
    ("f2_10_fog_kd_dg", "physics_fog", True, False, True),
    ("f2_11_fog_adapter_dg", "physics_fog", False, True, True),
]

TRAIN_FINISHED_RE = re.compile(
    r"Training Finished, checkpoints saved to (.+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a staged robust-perception combination search.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--hypes_yaml", required=True)
    parser.add_argument("--development_dir", required=True)
    parser.add_argument("--fog_lookup_dir", required=True)
    parser.add_argument("--output_dir", default="combination_search/l8_seed20260729")
    parser.add_argument(
        "--gpu", default="2",
        help="Physical GPU/HCU index. The subprocess sees it as logical cuda:0.")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--augmentation_seed", type=int, default=420)
    parser.add_argument("--evaluation_seed", type=int, default=620)
    parser.add_argument("--train_steps", type=int, default=800)
    parser.add_argument("--val_steps", type=int, default=100)
    parser.add_argument("--eval_frames", type=int, default=200)
    parser.add_argument("--max_cav", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-6)
    parser.add_argument("--clean_drop_tolerance", type=float, default=0.01)
    parser.add_argument(
        "--selection_domains", nargs="+",
        choices=["awa", "fog", "rain", "snow"],
        default=["fog", "rain", "snow"],
        help="Weather domains included in model ranking. AWA is excluded by "
             "default because it is a training augmentation rather than an "
             "unseen target weather domain.")
    parser.add_argument(
        "--full_eval_top_k", type=int, default=0,
        help="After screening, re-evaluate the top K candidates on a separate "
             "full-evaluation output tree; zero disables this stage.")
    parser.add_argument(
        "--full_eval_frames", type=int, default=0,
        help="Frames used by the Top-K re-evaluation; zero uses the complete "
             "development split.")
    parser.add_argument("--skip_base_eval", action="store_true")
    parser.add_argument(
        "--only", action="append", default=[],
        help="Run only the named candidate; may be supplied repeatedly.")
    return parser.parse_args()


def absolute(path, root):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def require(path, description):
    if not path.exists():
        raise FileNotFoundError("%s not found: %s" % (description, path))


def run_logged(command, log_path, env, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(str(item) for item in command), flush=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1)
        lines = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
            lines.append(line.rstrip())
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            "Command failed with exit code %d; see %s" %
            (return_code, log_path))
    return lines


def read_result(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError("Expected one result row in %s, got %d" %
                         (csv_path, len(rows)))
    row = rows[0]
    return {
        "ap30": float(row["ap30"]),
        "ap50": float(row["ap50"]),
        "ap70": float(row["ap70"]),
        "communication_rate": float(row["communication_rate"]),
        "samples": int(row["samples"]),
    }


def evaluate_condition(
        python, eval_script, model_dir, checkpoint, development_dir,
        output_dir, condition, env, repo_root, num_workers, eval_frames,
        evaluation_seed, fog_lookup_dir, max_cav, hypes_yaml=None):
    csv_path = output_dir / ("%s.csv" % condition)
    if csv_path.exists():
        return read_result(csv_path)

    command = [
        python, "-u", eval_script,
        "--model_dir", model_dir,
        "--checkpoint_path", checkpoint,
        "--weather", "%s=%s" % (condition, development_dir),
        "--weather_reliability", "off",
        "--num_workers", str(num_workers),
        "--max_frames", str(eval_frames),
        "--max_cav", str(max_cav),
        "--device", "cuda",
        "--output_csv", csv_path,
    ]
    if hypes_yaml is not None:
        command.extend(["--hypes_yaml", hypes_yaml])
    if condition != "clean":
        command.extend([
            "--input_branch", "weather_augmented",
            "--weather_augmentation_seed", str(evaluation_seed),
        ])
    if condition == "awa":
        command.extend(["--weather_augmentation_mode", "v2x_dgw_awa"])
    elif condition == "fog":
        command.extend([
            "--weather_augmentation_mode", "physics_fog",
            "--fog_lookup_dir", fog_lookup_dir,
            "--fog_alpha", "0.02",
        ])
    elif condition == "rain":
        command.extend([
            "--weather_augmentation_mode", "physics_rain",
            "--rain_rate", "40",
        ])
    elif condition == "snow":
        command.extend([
            "--weather_augmentation_mode", "physics_snow",
            "--snowfall_rate", "1.0",
        ])
    run_logged(
        command, output_dir / ("%s.log" % condition), env, repo_root)
    return read_result(csv_path)


def evaluate_suite(
        python, eval_script, model_dir, checkpoint, development_dir,
        output_dir, env, repo_root, num_workers, eval_frames,
        evaluation_seed, fog_lookup_dir, max_cav, hypes_yaml=None):
    results = {}
    for condition in ("clean", "awa", "fog", "rain", "snow"):
        results[condition] = evaluate_condition(
            python, eval_script, model_dir, checkpoint, development_dir,
            output_dir, condition, env, repo_root, num_workers, eval_frames,
            evaluation_seed, fog_lookup_dir, max_cav, hypes_yaml)
    return results


def train_candidate(
        python, train_script, candidate, base_model_dir, hypes_yaml,
        output_dir, env, repo_root, opt):
    name, augmentation, use_kd, use_adapter, use_dg = candidate
    state_path = output_dir / "candidate_state.json"
    if state_path.exists():
        with state_path.open(encoding="utf-8") as stream:
            state = json.load(stream)
        checkpoint = Path(state["checkpoint"])
        require(checkpoint, "resumable candidate checkpoint")
        return Path(state["model_dir"]), checkpoint

    command = [
        python, "-u", train_script,
        "--model_dir", base_model_dir,
        "--teacher_model_dir", base_model_dir,
        "--hypes_yaml", hypes_yaml,
        "--weather_augmentation_mode", augmentation,
        "--epochs", "1",
        "--batch_size", "1",
        "--num_workers", str(opt.num_workers),
        "--lr", str(opt.lr),
        "--warmup_epochs", "0",
        "--max_train_steps", str(opt.train_steps),
        "--max_val_steps", str(opt.val_steps),
        "--save_every_steps", str(opt.train_steps),
        "--seed", str(opt.seed),
        "--weather_augmentation_seed", str(opt.augmentation_seed),
        "--lambda_weather_det", "1.0",
        "--lambda_clean_det", "0.5",
        "--lambda_pred_cls", "0.5" if use_kd else "0.0",
        "--lambda_pred_reg", "0.5" if use_kd else "0.0",
        "--lambda_clean_comm", "1.0" if use_kd else "0.0",
        "--lambda_weather_comm", "1.0" if use_kd else "0.0",
        "--lambda_weather_feature", "0.5" if use_adapter else "0.0",
        "--lambda_clean_feature", "0.5" if use_adapter else "0.0",
        "--adapter_hidden_channels", "64",
        "--feature_background_floor", "0.05",
        "--lambda_dg_agent_align", "0.1" if use_dg else "0.0",
        "--lambda_dg_fusion_align", "1.0" if use_dg else "0.0",
        "--lambda_dg_agent_contrast", "0.01" if use_dg else "0.0",
        "--dg_contrast_temperature", "0.2",
        "--dg_pool_size", "4",
        "--dg_background_floor", "0.05",
        "--save_name", "combo_%s" % name,
    ]
    if augmentation == "physics_fog":
        command.extend([
            "--fog_lookup_dir", opt.fog_lookup_dir,
            "--fog_alpha_low", "0.005",
            "--fog_alpha_high", "0.03",
            "--fog_alpha_sampling", "lookup_discrete",
        ])
    lines = run_logged(
        command, output_dir / "train.log", env, repo_root)
    model_dir = None
    for line in lines:
        match = TRAIN_FINISHED_RE.search(line)
        if match:
            model_dir = Path(match.group(1).strip()).resolve()
    if model_dir is None:
        raise RuntimeError("Could not discover saved model directory")
    checkpoint = model_dir / ("net_step%d.pth" % opt.train_steps)
    require(checkpoint, "candidate checkpoint")
    state = {
        "name": name,
        "augmentation": augmentation,
        "prediction_kd_and_comm": use_kd,
        "feature_adapter": use_adapter,
        "dg_alignment_and_contrast": use_dg,
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
    }
    with state_path.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2, ensure_ascii=False)
    return model_dir, checkpoint


def summarize(
        candidate_rows, base_results, output_dir, tolerance,
        output_stem="search",
        selection_domains=("fog", "rain", "snow")):
    base_clean50 = base_results["clean"]["ap50"]
    weather_names = tuple(selection_domains)
    if not weather_names:
        raise ValueError("selection_domains cannot be empty")
    for name in weather_names:
        if name not in base_results:
            raise KeyError("Selection domain missing from base results: %s" %
                           name)
    for row in candidate_rows:
        results = row["results"]
        row["weather_macro_ap50"] = sum(
            results[name]["ap50"] for name in weather_names) / \
            float(len(weather_names))
        row["weather_macro_ap70"] = sum(
            results[name]["ap70"] for name in weather_names) / \
            float(len(weather_names))
        row["robustness_score"] = 0.5 * (
            row["weather_macro_ap50"] + row["weather_macro_ap70"])
        row["clean_ap50_drop"] = (
            base_clean50 - results["clean"]["ap50"])
        row["passes_clean_constraint"] = (
            row["clean_ap50_drop"] <= tolerance)

    viable = [row for row in candidate_rows
              if row["passes_clean_constraint"]]
    ranked_pool = viable if viable else candidate_rows
    ranked_pool.sort(
        key=lambda item: (
            item["robustness_score"],
            item["weather_macro_ap50"],
            item["results"]["clean"]["ap50"]),
        reverse=True)
    rank_by_name = {
        row["name"]: rank for rank, row in enumerate(ranked_pool, 1)}
    for row in candidate_rows:
        row["rank"] = rank_by_name.get(row["name"])

    summary = {
        "selection_protocol": {
            "weather_domains": list(weather_names),
            "score": "0.5 * macro_AP50 + 0.5 * macro_AP70",
            "clean_reference_ap50": base_clean50,
            "maximum_allowed_clean_ap50_drop": tolerance,
            "opv2vw_used_for_selection": False,
        },
        "base_results": base_results,
        "candidates": candidate_rows,
        "winner": ranked_pool[0] if ranked_pool else None,
    }
    summary_path = output_dir / ("%s_summary.json" % output_stem)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    ranking_path = output_dir / ("%s_ranking.csv" % output_stem)
    with ranking_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "rank", "name", "augmentation", "kd", "adapter", "dg",
            "clean_ap50", "awa_ap50", "fog_ap50", "rain_ap50",
            "snow_ap50", "weather_macro_ap50", "weather_macro_ap70",
            "robustness_score", "clean_ap50_drop",
            "passes_clean_constraint", "model_dir", "checkpoint",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
                candidate_rows,
                key=lambda item: item["rank"]
                if item["rank"] is not None else 10 ** 9):
            writer.writerow({
                "rank": row["rank"],
                "name": row["name"],
                "augmentation": row["augmentation"],
                "kd": row["kd"],
                "adapter": row["adapter"],
                "dg": row["dg"],
                "clean_ap50": row["results"]["clean"]["ap50"],
                "awa_ap50": row["results"]["awa"]["ap50"],
                "fog_ap50": row["results"]["fog"]["ap50"],
                "rain_ap50": row["results"]["rain"]["ap50"],
                "snow_ap50": row["results"]["snow"]["ap50"],
                "weather_macro_ap50": row["weather_macro_ap50"],
                "weather_macro_ap70": row["weather_macro_ap70"],
                "robustness_score": row["robustness_score"],
                "clean_ap50_drop": row["clean_ap50_drop"],
                "passes_clean_constraint": row["passes_clean_constraint"],
                "model_dir": row["model_dir"],
                "checkpoint": row["checkpoint"],
            })
    return summary_path, ranking_path, ranked_pool


def main():
    opt = parse_args()
    repo_root = Path(opt.repo_root).expanduser().resolve()
    base_model_dir = absolute(opt.base_model_dir, repo_root)
    hypes_yaml = absolute(opt.hypes_yaml, repo_root)
    development_dir = absolute(opt.development_dir, repo_root)
    fog_lookup_dir = absolute(opt.fog_lookup_dir, repo_root)
    output_dir = absolute(opt.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_script = repo_root / "opencood/tools/train_prediction_distillation.py"
    eval_script = repo_root / "opencood/tools/test_weather_where2comm.py"
    for path, description in (
            (repo_root / "opencood", "OpenCOOD package"),
            (base_model_dir / "net_epoch50.pth", "clean Epoch50 checkpoint"),
            (hypes_yaml, "training YAML"),
            (development_dir, "development split"),
            (fog_lookup_dir, "Physics-Fog lookup directory"),
            (train_script, "training script"),
            (eval_script, "evaluation script")):
        require(path, description)
    fog_tables = list(fog_lookup_dir.glob("*alpha_*.pickle"))
    if not fog_tables:
        raise FileNotFoundError(
            "No *alpha_*.pickle lookup tables in %s. Point "
            "--fog_lookup_dir to the converted OpenCOOD lookup directory, "
            "not merely the upstream integral_lookup_tables directory." %
            fog_lookup_dir)

    candidates = L8_CANDIDATES
    if opt.only:
        selected = set(opt.only)
        candidates = [item for item in candidates if item[0] in selected]
        missing = selected - {item[0] for item in candidates}
        if missing:
            raise ValueError("Unknown candidate(s): %s" %
                             ", ".join(sorted(missing)))

    env = os.environ.copy()
    # On ROCm, filter the physical device first and then address the only
    # exposed device as logical cuda:0.  Setting HIP_VISIBLE_DEVICES and
    # CUDA_VISIBLE_DEVICES to the physical index at the same time can produce
    # ambiguous/double remapping on some cluster wrappers.
    env["ROCR_VISIBLE_DEVICES"] = str(opt.gpu)
    env["HIP_VISIBLE_DEVICES"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get(
        "PYTHONPATH", "")
    python = sys.executable

    print("GPU physical index:", opt.gpu)
    print(
        "Child visibility: ROCR_VISIBLE_DEVICES=%s "
        "HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0" % opt.gpu)
    print("Candidates:", len(candidates))
    print("Selection data:", development_dir)
    print("OPV2V-W is not used by this search.")

    base_eval_dir = output_dir / "base"
    base_checkpoint = base_model_dir / "net_epoch50.pth"
    base_results_path = base_eval_dir / "results.json"
    if base_results_path.exists():
        with base_results_path.open(encoding="utf-8") as stream:
            base_results = json.load(stream)
    elif opt.skip_base_eval:
        raise ValueError(
            "--skip_base_eval requires an existing base/results.json")
    else:
        base_results = evaluate_suite(
            python, eval_script, base_model_dir, base_checkpoint,
            development_dir, base_eval_dir, env, repo_root, opt.num_workers,
            opt.eval_frames, opt.evaluation_seed, fog_lookup_dir,
            opt.max_cav, hypes_yaml)
        with base_results_path.open("w", encoding="utf-8") as stream:
            json.dump(base_results, stream, indent=2)

    rows = []
    for candidate in candidates:
        name, augmentation, use_kd, use_adapter, use_dg = candidate
        candidate_dir = output_dir / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print("Candidate:", name)
        print("augmentation=%s kd=%s adapter=%s dg=%s" %
              (augmentation, use_kd, use_adapter, use_dg))
        model_dir, checkpoint = train_candidate(
            python, train_script, candidate, base_model_dir, hypes_yaml,
            candidate_dir, env, repo_root, opt)
        result_path = candidate_dir / "results.json"
        if result_path.exists():
            with result_path.open(encoding="utf-8") as stream:
                results = json.load(stream)
        else:
            results = evaluate_suite(
                python, eval_script, model_dir, checkpoint, development_dir,
                candidate_dir / "eval", env, repo_root, opt.num_workers,
                opt.eval_frames, opt.evaluation_seed, fog_lookup_dir,
                opt.max_cav)
            with result_path.open("w", encoding="utf-8") as stream:
                json.dump(results, stream, indent=2)
        rows.append({
            "name": name,
            "augmentation": augmentation,
            "kd": use_kd,
            "adapter": use_adapter,
            "dg": use_dg,
            "model_dir": str(model_dir),
            "checkpoint": str(checkpoint),
            "results": results,
        })

    summary_path, ranking_path, ranked = summarize(
        rows, base_results, output_dir, opt.clean_drop_tolerance,
        output_stem="search",
        selection_domains=opt.selection_domains)
    print("\nCombination search finished")
    print("Ranking:", ranking_path)
    print("Summary:", summary_path)
    if ranked:
        winner = ranked[0]
        print(
            "Winner: %s score=%.4f weather_AP50=%.4f "
            "weather_AP70=%.4f clean_AP50=%.4f" %
            (winner["name"], winner["robustness_score"],
             winner["weather_macro_ap50"], winner["weather_macro_ap70"],
             winner["results"]["clean"]["ap50"]))

    if opt.full_eval_top_k > 0:
        top_k = min(opt.full_eval_top_k, len(ranked))
        full_root = output_dir / "full_eval"
        full_root.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print(
            "Full development re-evaluation: top %d, frames=%s" %
            (top_k, "all" if opt.full_eval_frames == 0
             else opt.full_eval_frames))

        full_base_path = full_root / "base_results.json"
        if full_base_path.exists():
            with full_base_path.open(encoding="utf-8") as stream:
                full_base_results = json.load(stream)
        else:
            full_base_results = evaluate_suite(
                python, eval_script, base_model_dir, base_checkpoint,
                development_dir, full_root / "base", env, repo_root,
                opt.num_workers, opt.full_eval_frames,
                opt.evaluation_seed, fog_lookup_dir, opt.max_cav,
                hypes_yaml)
            with full_base_path.open("w", encoding="utf-8") as stream:
                json.dump(full_base_results, stream, indent=2)

        full_rows = []
        for screened in ranked[:top_k]:
            name = screened["name"]
            print("\nFull evaluation candidate:", name)
            full_result_path = full_root / name / "results.json"
            if full_result_path.exists():
                with full_result_path.open(encoding="utf-8") as stream:
                    full_results = json.load(stream)
            else:
                full_results = evaluate_suite(
                    python, eval_script, screened["model_dir"],
                    screened["checkpoint"], development_dir,
                    full_root / name / "eval", env, repo_root,
                    opt.num_workers, opt.full_eval_frames,
                    opt.evaluation_seed, fog_lookup_dir, opt.max_cav)
                full_result_path.parent.mkdir(parents=True, exist_ok=True)
                with full_result_path.open("w", encoding="utf-8") as stream:
                    json.dump(full_results, stream, indent=2)
            full_rows.append({
                "name": name,
                "augmentation": screened["augmentation"],
                "kd": screened["kd"],
                "adapter": screened["adapter"],
                "dg": screened["dg"],
                "model_dir": screened["model_dir"],
                "checkpoint": screened["checkpoint"],
                "results": full_results,
            })

        full_summary, full_ranking, full_ranked = summarize(
            full_rows, full_base_results, output_dir,
            opt.clean_drop_tolerance, output_stem="full_eval",
            selection_domains=opt.selection_domains)
        print("\nFull evaluation ranking:", full_ranking)
        print("Full evaluation summary:", full_summary)
        if full_ranked:
            winner = full_ranked[0]
            print(
                "Full evaluation winner: %s score=%.4f "
                "weather_AP50=%.4f weather_AP70=%.4f clean_AP50=%.4f" %
                (winner["name"], winner["robustness_score"],
                 winner["weather_macro_ap50"],
                 winner["weather_macro_ap70"],
                 winner["results"]["clean"]["ap50"]))


if __name__ == "__main__":
    main()
