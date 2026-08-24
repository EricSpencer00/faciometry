"""Run every arm of the evaluation, in dependency order, and write the summary.

    python -m evals.run_all            # everything
    python -m evals.run_all 1 2 3      # selected arms
    make -C evals evals                # the same thing

Arm 5 reads arms 2 and 3 from ``evals/results/``, so the order here is a
dependency order and not a preference. Arms 8, 9 and 10 need the FRLL images
and the MediaPipe weights under ``evals/data/``, and arm 11 additionally needs
the YuNet and 6DRepNet weights, which ``faciometry.models.weights`` fetches into
its own cache; ``make -C evals data`` fetches the rest. If they are absent those arms report themselves as not run, with the
reason, rather than being skipped silently.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
import traceback

from evals._bootstrap import RESULTS, provenance, write_json

ARMS = [
    ("1", "arm01_synthetic_geometry", "synthetic geometry control"),
    ("2", "arm02_pose_sweep", "pose sweep"),
    ("3", "arm03_encode_control", "encode/decode control for arm 2"),
    ("4", "arm04_perspective", "perspective sweep"),
    ("5", "arm05_discriminability", "discriminability (needs arms 2 and 3)"),
    ("6", "arm06_niosh", "NIOSH 2003 external check"),
    ("7", "arm07_negative_control", "negative control on the normative model"),
    ("8", "arm08_frll", "FRLL landmark accuracy and spread"),
    ("9", "arm09_test_retest", "test-retest repeatability (needs arm 8's data)"),
    ("10", "arm10_fairness", "fairness stratification (needs arm 8's data)"),
    ("11", "arm11_pose_estimator", "pose estimator error without a labelled benchmark"),
]


def main(selected: list[str] | None = None) -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out, timings, failures = {}, {}, 0
    for num, mod, label in ARMS:
        if selected and num not in selected:
            continue
        print(f"--- arm {num}: {label}", flush=True)
        t0 = time.time()
        try:
            m = importlib.import_module(f"evals.arms.{mod}")
            summary = m.run()
            out[num] = {"label": label, "status": "ran", "summary": summary}
            timings[num] = round(time.time() - t0, 2)
        except FileNotFoundError as exc:
            out[num] = {"label": label, "status": "not run",
                        "reason": f"required data not present: {exc}"}
            print(f"    NOT RUN: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            out[num] = {"label": label, "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()}
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
        else:
            print(f"    ok in {timings[num]}s", flush=True)

    write_json("summary", {"arms": out,
                           "n_ran": sum(1 for v in out.values() if v["status"] == "ran"),
                           "n_not_run": sum(1 for v in out.values() if v["status"] == "not run"),
                           "n_failed": failures})
    # Wall clock lives in its own file. It is the one thing in `results/` that
    # cannot be byte-identical across runs, and keeping it out of summary.json
    # is what lets the reproducibility claim be checked with a checksum rather
    # than qualified in prose.
    write_json("timings", {"seconds_per_arm": timings})
    print(json.dumps({k: v["status"] for k, v in out.items()}, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", help="arm numbers to run; default all")
    raise SystemExit(main(ap.parse_args().arms or None))
