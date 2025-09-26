#!/usr/bin/env python3
"""
Lightweight wrapper to run nullgtn-artifact predictor with minimal coupling.

Assumptions/minimal contract:
- A working directory exists that contains:
  - temp_output.json           (graph JSON produced by nullgtn artifact tooling)
  - nullgtn_<model_key>.json.pkl (serialized model file)
- We simply invoke the artifact's predict.py and capture its stdout indices.
- We emit a CFWR-style predictions JSON with best-effort mapping (indices only).

This avoids invasive changes inside nullgtn-artifact while enabling CFWR to
execute it as an additional model option.
"""

import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path


def run_nullgtn_predict(artifact_dir: str, model_key: str, work_dir: str) -> str:
    """Run the artifact predictor and return raw stdout."""
    pred_py = Path(artifact_dir) / "reann_cond_pairs" / "GTN_comb" / "predict.py"
    if not pred_py.exists():
        raise FileNotFoundError(f"nullgtn predict.py not found at {pred_py}")

    # predict.py expects: argv[1]=model_key, argv[2]=directory (work_dir)
    cmd = [sys.executable, str(pred_py), model_key, work_dir]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(pred_py.parent))
    if res.returncode != 0:
        raise RuntimeError(f"nullgtn predict failed: {res.stderr}")
    return res.stdout


def parse_indices(stdout_text: str) -> list:
    """Extract integer indices printed by the artifact (one per line)."""
    indices = []
    for line in stdout_text.splitlines():
        line = line.strip()
        if re.fullmatch(r"\d+", line):
            indices.append(int(line))
    return indices


def write_predictions(indices: list, out_path: str) -> None:
    """Write a simple JSON list; indices are artifact-specific node positions."""
    payload = {
        "model": "nullgtn",
        "predicted_node_indices": indices,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Run nullgtn-artifact prediction and emit JSON")
    ap.add_argument("--artifact_dir", required=True, help="Path to nullgtn-artifact root")
    ap.add_argument("--model_key", required=True, help="Model key used in artifact (suffix in nullgtn_<key>.json.pkl)")
    ap.add_argument("--work_dir", required=True, help="Work dir with temp_output.json and model pickle")
    ap.add_argument("--out_path", required=True, help="Output JSON path for predictions")
    args = ap.parse_args()

    # Sanity checks
    temp_json = Path(args.work_dir) / "temp_output.json"
    # Accept either artifact-named pickle (.json.pkl) or plain .pkl
    model_pkl_json = Path(args.work_dir) / f"nullgtn_{args.model_key}.json.pkl"
    model_pkl_plain = Path(args.work_dir) / f"nullgtn_{args.model_key}.pkl"
    if not temp_json.exists():
        print(f"ERROR: Missing {temp_json}", file=sys.stderr)
        return 2
    if not (model_pkl_json.exists() or model_pkl_plain.exists()):
        print(f"ERROR: Missing model pickle nullgtn_{args.model_key}.json.pkl or .pkl in {args.work_dir}", file=sys.stderr)
        return 2

    try:
        stdout = run_nullgtn_predict(args.artifact_dir, args.model_key, args.work_dir)
        indices = parse_indices(stdout)
        write_predictions(indices, args.out_path)
        print(f"nullgtn predictions written to {args.out_path} ({len(indices)} indices)")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())


