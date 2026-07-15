"""
Thin wrapper around TIPs.py so the web app (and CLI) can turn a raw CBCT volume
into the teeth-instance and pulp-instance segmentations that the registration
pipeline expects.

TIPs (Tooth Instance and Pulp segmentation, https://github.com/TaoZhong11/TIPs)
is an nnU-Net v2 pipeline. To actually run it you need, once:

  1. the nnU-Net stack installed        ->  pip install -e .
  2. the U-Mamba trainer                 ->  see the TIPs repo
  3. the trained weights in ./nnResults/ ->  datasets 803 (teeth binary),
                                             812 (teeth instance), 810 (pulp)

`tips_available()` checks that (1) and (3) are in place; if not, the web button
returns a clear message instead of failing deep inside nnU-Net. Segmentation is
GPU-heavy and takes minutes per scan, so it is never run implicitly.
"""

import os
import gzip
import glob
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TIPS_SCRIPT = os.path.join(HERE, "TIPs.py")
# TIPs.py forces nnUNet_results='nnResults/' relative to its working directory,
# so the weights must live under <TIPS_HOME>/nnResults/. Override the home with
# the TIPS_HOME environment variable if your weights live elsewhere.
TIPS_HOME = os.environ.get("TIPS_HOME", HERE)
NN_RESULTS = os.path.join(TIPS_HOME, "nnResults")


def tips_available():
    """Return (ok, reason). ok=True only if TIPs can plausibly run here."""
    if not os.path.exists(TIPS_SCRIPT):
        return False, "TIPs.py not found in the repository."
    if shutil.which("nnUNetv2_predict") is None:
        return False, ("nnU-Net is not installed (nnUNetv2_predict not on PATH). "
                       "Run `pip install -e .` and set up the U-Mamba trainer.")
    if not os.path.isdir(NN_RESULTS) or not os.listdir(NN_RESULTS):
        return False, (f"Trained weights not found in {NN_RESULTS}. Download the "
                       "TIPs model weights (datasets 803/810/812) into nnResults/.")
    return True, "ok"


def _as_niigz(src, dst_dir, stem="case"):
    """Copy `src` into dst_dir as <stem>.nii.gz (gzipping a plain .nii)."""
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, stem + ".nii.gz")
    if src.endswith(".nii.gz"):
        shutil.copy(src, dst)
    elif src.endswith(".nii"):
        with open(src, "rb") as fin, gzip.open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    else:
        raise ValueError("CBCT must be a NIfTI volume (.nii or .nii.gz).")
    return dst


def _first_niigz(folder):
    hits = sorted(glob.glob(os.path.join(folder, "*.nii.gz")))
    return hits[0] if hits else None


def segment_cbct(cbct_path, workdir, timeout=None):
    """Run TIPs on a single CBCT volume.

    Returns {'teeth': <path>, 'pulp': <path>} pointing at the teeth-instance and
    pulp-instance segmentations (both NIfTI, in the CBCT frame). Raises
    RuntimeError with a readable message if TIPs is unavailable or fails.
    """
    ok, reason = tips_available()
    if not ok:
        raise RuntimeError(reason)

    os.makedirs(workdir, exist_ok=True)
    in_dir = os.path.join(workdir, "cbct_in")
    _as_niigz(cbct_path, in_dir, stem="case")

    # No -c flag -> automatic centroid (TIPs runs without stdin prompts).
    cmd = [sys.executable, TIPS_SCRIPT, in_dir]
    proc = subprocess.run(cmd, cwd=TIPS_HOME, capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError("TIPs segmentation failed:\n" + tail)

    teeth = _first_niigz(in_dir + "_resample_teeth_instance")
    pulp = _first_niigz(in_dir + "_resample_pulps_instance")
    if not teeth or not pulp:
        raise RuntimeError("TIPs finished but no segmentation output was found "
                           f"under {in_dir}_resample_*_instance/.")
    return {"teeth": teeth, "pulp": pulp}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Segment a CBCT with TIPs.")
    ap.add_argument("cbct", help="CBCT volume (.nii / .nii.gz)")
    ap.add_argument("--workdir", default="./outputs/tips_run")
    args = ap.parse_args()
    ok, reason = tips_available()
    if not ok:
        raise SystemExit("TIPs not ready: " + reason)
    out = segment_cbct(args.cbct, args.workdir)
    print("teeth:", out["teeth"])
    print("pulp :", out["pulp"])
