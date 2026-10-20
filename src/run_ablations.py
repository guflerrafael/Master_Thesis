"""Run the ablation study (notes.md 6.7): every experiment x seed is one src/train.py run.

Each experiment changes ONE setting of the reference configuration ('ref' = v1: hourly steps,
7 channels, 2 x 32 ConvLSTM, focal loss, 32 x 32 crops, 20 epochs). Runs are written to
models/ablations/<experiment>_s<seed>.pt and outputs/ablations/<experiment>_s<seed>_{log.txt,
history.csv,done.json}. Finished runs (done.json present) are skipped, so the script can simply be
restarted after a Colab disconnect. Summaries: src/summarize_ablations.py.

Examples (from the repository root):
  python src/run_ablations.py --tier 1 --seeds 0 1 2              # Tier 1, three seeds
  python src/run_ablations.py --only ref births_plus --seeds 0      # selected experiments
  python src/run_ablations.py --tier 1 2 3 --dry-run                # list the commands only
  python src/run_ablations.py --tier 1 2 3 --smoke                  # 1 epoch, 3 batches each: does every option run?
  python src/run_ablations.py --tier 1 --extra="--workers 2"        # extra train.py options (note the '=')
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
}
SMOKE = "--epochs 1 --max-batches 3 --val-periods 10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, nargs="*", default=[1])
    ap.add_argument("--only", nargs="*", default=None, help="experiment names (overrides --tier)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--extra", default="", help="extra train.py options for every run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny runs into ablations_smoke/ to test every option")
    args = ap.parse_args()

    names = args.only if args.only else [n for n, (t, _, _) in EXPERIMENTS.items() if t in args.tier]
    unknown = set(names) - set(EXPERIMENTS)
    assert not unknown, f"unknown experiments: {unknown}"
    folder = "ablations_smoke" if args.smoke else "ablations"
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
        opts = f"{EXPERIMENTS[n][1]} {SMOKE if args.smoke else ''} {args.extra}"
        cmd = [sys.executable, "-u", str(SRC / "train.py"), "--name", run, "--seed", str(seed), *shlex.split(opts)]
        print(f"[{k}/{len(runs)}] {run}: {EXPERIMENTS[n][2]} | {' '.join(cmd[2:])}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        with open(ROOT / "outputs" / f"{run}_log.txt", "w") as log:
            ret = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        status = "ok" if ret == 0 else f"FAILED (exit {ret}, see outputs/{run}_log.txt)"
        print(f"    {status} in {(time.time() - t0) / 60:.1f} min (total {(time.time() - t_start) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
