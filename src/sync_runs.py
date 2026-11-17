"""Merge finished runs from another machine (Colab: Google Drive) into this repository (notes.md 9.14).

Runs on the Mac and on Colab use the same --folder but different seeds. Colab writes to
MyDrive/master_thesis_runs/{models,outputs}/<folder>/. Download that folder (Drive web: right click ->
Download, then unzip; or Google Drive for desktop) and point --src at the directory that contains
models/ and outputs/ (several --src for split zip downloads).

Copied per FINISHED run (outputs/<folder>/<run>_done.json present): models/<folder>/<run>.pt and
outputs/<folder>/<run>_{done.json,history.csv,log.txt}. Nothing is ever overwritten: a run that
exists here with different content (both machines ran the same seed) or is unfinished here is
reported as a conflict and skipped. Unfinished runs at the source are listed with their last log line.

Run from the repository root:
  .venv/bin/python src/sync_runs.py --src ~/Downloads/master_thesis_runs --folder ablations_all --dry-run
  .venv/bin/python src/sync_runs.py --src ~/Downloads/master_thesis_runs --folder ablations_all
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import eda_figures as E

ROOT = E.ROOT


def files_of(base, folder, run):
    return {"model": base / "models" / folder / f"{run}.pt",
            "done": base / "outputs" / folder / f"{run}_done.json",
            "history": base / "outputs" / folder / f"{run}_history.csv",
            "log": base / "outputs" / folder / f"{run}_log.txt"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="director(y/ies) containing models/ and outputs/")
    ap.add_argument("--folder", required=True, help="run folder, e.g. ablations_all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    assert args.folder != "final", "outputs/final holds the results of notes.md 6.17; not merged by this script"

    copied, same, conflicts, unfinished = [], [], [], []
    for src in map(lambda p: Path(p).expanduser().resolve(), args.src):
        out = src / "outputs" / args.folder
        if not out.is_dir():
            print(f"{src}: no outputs/{args.folder}/ (expects the directory that holds models/ and outputs/)")
            continue
        for done in sorted(out.glob("*_done.json")):
            run = done.name[:-len("_done.json")]
            s, d = files_of(src, args.folder, run), files_of(ROOT, args.folder, run)
            if not s["model"].exists():
                conflicts.append((run, f"finished at {src.name} but models/{args.folder}/{run}.pt is missing there"))
                continue
            here = [k for k, p in d.items() if p.exists()]
            if here:
                if digest(d["done"]) == digest(s["done"]) and digest(d["model"]) == digest(s["model"]):
                    same.append(run)
                elif not d["done"].exists():
                    conflicts.append((run, "unfinished (or running) on this machine: same seed on both machines?"))
                else:
                    conflicts.append((run, "finished on both machines with different results: same seed twice?"))
                continue
            info = json.loads(s["done"].read_text())
            copied.append((run, info.get("host", "?"), info.get("device", "?"), info.get("best_mean_ap")))
            if not args.dry_run:
                for k, p in s.items():
                    if p.exists():
                        d[k].parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, d[k])
        for log in sorted(out.glob("*_log.txt")):
            run = log.name[:-len("_log.txt")]
            if not (out / f"{run}_done.json").exists():
                lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip()]
                state = (src / "models" / args.folder / f"{run}_last.pt").exists()
                unfinished.append((run, src.name, state, lines[-1][:140] if lines else ""))

    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {len(copied)} finished runs into models/{args.folder}/ and outputs/{args.folder}/")
    for run, host, dev, best in copied:
        print(f"  {run}: {host}, {dev}, best mean AP {best if best is None or best < 0 else round(best, 4)}")
    if same:
        print(f"already here, identical: {', '.join(same)}")
    for run, why in conflicts:
        print(f"CONFLICT {run}: {why} (skipped)")
    for run, src, state, last in unfinished:
        print(f"unfinished at {src}: {run} (resume state {'present' if state else 'missing'}); last log line: {last}")


if __name__ == "__main__":
    main()
