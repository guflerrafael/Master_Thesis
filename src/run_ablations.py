"""Run the ablation study (notes.md 6.7): every experiment x seed is one src/train.py run.

Each experiment changes ONE setting of the reference configuration ('ref' = v1: hourly steps,
7 channels, 2 x 32 ConvLSTM, focal loss, 32 x 32 crops, 20 epochs). Runs are written to
models/ablations/<experiment>_s<seed>.pt and outputs/ablations/<experiment>_s<seed>_{log.txt,
history.csv,done.json}. Finished runs (done.json present) are skipped, so the script can simply be
restarted after a Colab disconnect. An interrupted run continues after its last finished epoch
(train.py --resume, models/<run>_last.pt), also on another machine. Summaries: src/summarize_ablations.py.

All years up to 2023 (notes.md 9.14): the tier-7 experiments train on train + train2 + train3
(2004-2014, 2017-2023) and validate on 2015-2016, so they compare with the runs trained on 2004-2014.
--final trains the named experiments on every non-test year (train,val,train2,train3, --no-val, last
epoch); never into --folder final (the models of notes.md 6.17).

Mac + Colab at the same time: give each machine its own seeds and the SAME --folder, e.g.
  Mac:    python src/run_ablations.py --only cnn3d_st_all --seeds 0 --folder ablations_all
  Colab:  python src/run_ablations.py --only cnn3d_st_all --seeds 1 2 --folder ablations_all --extra="--workers 2"
then merge the Colab results with src/sync_runs.py (notes.md 9.14).

Examples (from the repository root):
  python src/run_ablations.py --tier 1 --seeds 0 1 2              # Tier 1, three seeds
  python src/run_ablations.py --only ref births_plus --seeds 0      # selected experiments
  python src/run_ablations.py --tier 1 2 3 --dry-run                # list the commands only
  python src/run_ablations.py --tier 1 2 3 --smoke                  # 1 epoch, 3 batches each: does every option run?
  python src/run_ablations.py --tier 1 --extra="--workers 2"        # extra train.py options (note the '=')
  python src/run_ablations.py --only cnn3d_st --final --seeds 0 1 2 --folder final_2023   # final models
"""
import argparse
import shlex
import subprocess
import sys
import time

import eda_figures as E

ROOT = E.ROOT
SRC = ROOT / "src"
BASE = "frp,confidence,detected,period,time,pt,persistent"
ALL_YEARS = "--train-splits train,train2,train3"                 # 2004-2014 + 2017-2023, validated on 2015-2016
FINAL = "--train-splits train,val,train2,train3 --no-val"        # every non-test year, last epoch
CNN3D_ST = "--arch cnn3d_st --time-bins 12 --sigma-t 1 --neg-ratio 2"

# name: (tier, train.py options, question)
EXPERIMENTS = {
    # ---- Tier 1: main claims
    "ref":              (1, "", "reference = v1 (hourly)"),
    "ref15":            (1, "--step-agg 1", "native 15-min steps (v2)"),
    "births_plus":      (1, f"--channels {BASE},births", "raw input + clustering (nucleus births) channel"),
    "births_only":      (1, "--channels births,period,time,pt,persistent", "clustering only (births instead of raw hotspots)"),
    "no_time":          (1, "--channels frp,confidence,detected,pt,persistent", "without period flag and time"),
    "no_static":        (1, "--channels frp,confidence,detected,period,time", "without mainland and persistent masks"),
    "no_conf":          (1, "--channels frp,detected,period,time,pt,persistent", "without confidence"),
    "no_frp":           (1, "--channels confidence,detected,period,time,pt,persistent", "without FRP intensity"),
    "presence_only":    (1, "--channels detected,period,time,pt,persistent", "detections only (no FRP, no confidence)"),
    "hist0":            (1, "--history-h 0", "no history before the period"),
    "tail0":            (1, "--tail-h 0", "no tail after the period"),
    # ---- Tier 2: architecture / training design
    "layers1":          (2, "--layers 1", "1 ConvLSTM layer"),
    "layers3":          (2, "--layers 3", "3 ConvLSTM layers"),
    "hidden16":         (2, "--hidden 16", "16 hidden channels"),
    "hidden64":         (2, "--hidden 64", "64 hidden channels"),
    "kernel5":          (2, "--kernel 5", "5 x 5 ConvLSTM kernels"),
    "readout_max":      (2, "--readout max", "head on max over time"),
    "readout_mean":     (2, "--readout mean", "head on mean over time"),
    "cnn3d":            (2, "--arch cnn3d", "no recurrence: 3D CNN"),
    "crop16":           (2, "--crop 16", "16 x 16 training crops"),
    "crop64":           (2, "--crop 64", "64 x 64 training crops"),
    # ---- Tier 3: labels, loss, sampling
    "sigma05":          (3, "--sigma 0.5", "narrower targets (sigma 0.5 px)"),
    "sigma2":           (3, "--sigma 2", "wider targets (sigma 2 px)"),
    "loss_bce":         (3, "--loss bce", "weighted BCE instead of focal loss"),
    "loss_mse":         (3, "--loss mse", "MSE instead of focal loss"),
    "neg0":             (3, "--neg-ratio 0", "no hard negatives"),
    "neg05":            (3, "--neg-ratio 0.5", "0.5 hard negatives per positive"),
    "neg2":             (3, "--neg-ratio 2", "2 hard negatives per positive"),
    "ignore0":          (3, "--ignore-r 0", "no ignore regions in training"),
    "ignore4":          (3, "--ignore-r 4", "ignore radius 4 px in training"),
    "target100":        (3, "--min-target-ha 100", "targets >= 100 ha only"),
    "noflip":           (3, "--no-flip", "no flip augmentation"),
    # ---- combinations (candidate final configurations, corrected data, notes.md 6.13)
    "neg2_noflip":      (4, "--neg-ratio 2 --no-flip", "2 hard negatives per positive + no flips"),
    "conv_final":       (4, "--hidden 16 --neg-ratio 2 --readout max", "ConvLSTM candidate: 16 ch + 2 negatives + max readout"),
    "cnn3d_final":      (4, "--arch cnn3d --neg-ratio 2", "3D-CNN candidate: no recurrence + 2 negatives"),
    # ---- ICNF SGIF export (notes.md 8.6)
    "cnn3d_final_rk":   (5, "--arch cnn3d --neg-ratio 2 --rekindle-ignore", "cnn3d_final + ignore regions at all rekindle records"),
    # ---- space-time ignition maps (notes.md 9.6, step 2: first run; step 3: ablations)
    "cnn3d_st":         (6, CNN3D_ST, "space-time 3D CNN, 12 hourly maps + 2D head, 2 negatives"),
    # ---- all training years up to 2023, validated on 2015-2016 (notes.md 7.7 E2, 9.14)
    "cnn3d_st_all":     (7, f"{CNN3D_ST} {ALL_YEARS}", "cnn3d_st trained on 2004-2014 + 2017-2023"),
    "cnn3d_final_all":  (7, f"--arch cnn3d --neg-ratio 2 {ALL_YEARS}", "cnn3d_final trained on 2004-2014 + 2017-2023"),
}
SMOKE = "--epochs 1 --max-batches 3 --val-periods 10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, nargs="*", default=[1])
    ap.add_argument("--only", nargs="*", default=None, help="experiment names (overrides --tier)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--extra", default="", help="extra train.py options for every run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny runs into ablations_smoke/ (with --final: final_smoke/) to test every option")
    ap.add_argument("--folder", default="ablations", help="output sub-folder, e.g. ablations_local for runs on another machine")
    ap.add_argument("--final", action="store_true", help=f"final models: {FINAL} (not into --folder final)")
    args = ap.parse_args()

    names = args.only if args.only else [n for n, (t, _, _) in EXPERIMENTS.items() if t in args.tier]
    unknown = set(names) - set(EXPERIMENTS)
    assert not unknown, f"unknown experiments: {unknown}"
    if args.final:
        assert args.folder != "final", "--folder final holds the models of notes.md 6.17; use e.g. final_2023"
        clash = [n for n in names if "--train-splits" in EXPERIMENTS[n][1] or "--no-val" in EXPERIMENTS[n][1]]
        assert not clash, f"--final sets the training years itself; use the base experiments, not {clash}"
    folder = ("final_smoke" if args.final else "ablations_smoke") if args.smoke else args.folder
    seeds = args.seeds[:1] if args.smoke else args.seeds
    runs = [(n, s) for n in names for s in seeds]
    print(f"{len(runs)} runs ({len(names)} experiments × {len(seeds)} seeds) -> models/{folder}/, outputs/{folder}/", flush=True)

    (ROOT / "outputs" / folder).mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for k, (n, seed) in enumerate(runs, 1):
        run = f"{folder}/{n}_s{seed}"
        if (ROOT / "outputs" / f"{run}_done.json").exists():
            print(f"[{k}/{len(runs)}] {run}: done, skipped", flush=True)
            continue
        opts = f"{EXPERIMENTS[n][1]} {FINAL if args.final else ''} {SMOKE if args.smoke else ''} {args.extra}"
        cmd = [sys.executable, "-u", str(SRC / "train.py"), "--name", run, "--seed", str(seed), "--resume", *shlex.split(opts)]
        print(f"[{k}/{len(runs)}] {run}: {EXPERIMENTS[n][2]} | {' '.join(cmd[2:])}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        resuming = (ROOT / "models" / f"{run}_last.pt").exists()
        if resuming:
            print(f"    resuming after the last finished epoch (models/{run}_last.pt)", flush=True)
        with open(ROOT / "outputs" / f"{run}_log.txt", "a" if resuming else "w") as log:
            ret = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        status = "ok" if ret == 0 else f"FAILED (exit {ret}, see outputs/{run}_log.txt)"
        print(f"    {status} in {(time.time() - t0) / 60:.1f} min (total {(time.time() - t_start) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
