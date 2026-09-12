"""
External registration benchmark on the public MICCAI 2025 STSR Task-2 dataset
(paired CBCT + IOS): https://www.codabench.org/competitions/6470/

For every case and jaw it runs THIS repository's registration and reports:

  * surface agreement (the same metrics as the clinical cohort table):
    NSD@0.5, ASSD, RMSD, computed between our registered IOS and the CBCT
    tooth-surface point cloud;
  * an INDEPENDENT accuracy check whenever a ground-truth transform is provided
    (Labels/<case>/<jaw>_gt.npy): rotation error (deg) and target registration
    error TRE (mm) between our transform and the ground truth. This is the
    landmark-free "independent ground truth" the reviewers asked for.

Expected layout (matching the challenge dataloader, data/dataset.py):
  <root>/Images/<case>/lower.stl , upper.stl , CBCT.nii.gz
  <root>/Labels/<case>/lower_gt.npy , upper_gt.npy        (optional; enables TRE)

The CBCT tooth surface is obtained by thresholding the raw volume (default > 800,
as the challenge does) and mapping the boundary voxels through the affine.

GROUND-TRUTH CONVENTION. --gt-frame raw (default) assumes the .npy maps the STL
vertices in millimetres onto the CBCT world space in millimetres. If the STSR GT
is stored in the centred/scaled frame that data/dataset.py builds, pass
--gt-frame centered. The surface metrics are valid either way; only the GT-based
TRE/rotation depend on this. Confirm against the challenge's own evaluation
before quoting the TRE. Nothing here is hard-coded: all numbers are measured.

Usage:
  python stsr_eval.py --root /path/to/STSR --out ./outputs/stsr
"""

import argparse
import csv
import glob
import os
import numpy as np
import nibabel as nib

import registration as R
from eval_metrics import surface_metrics


def _std(vals):
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def cbct_surface_points(cbct_path, thr=800.0, max_points=60000):
    """Threshold the raw CBCT and map suprathreshold voxels to world mm."""
    img = nib.load(cbct_path)
    data = np.asarray(img.get_fdata())
    coords = np.argwhere(data > thr)
    if coords.shape[0] == 0:
        raise ValueError(f"No voxels above {thr} HU in {cbct_path}.")
    if coords.shape[0] > max_points:                # keep it tractable
        coords = coords[np.random.choice(coords.shape[0], max_points, replace=False)]
    homog = np.hstack([coords, np.ones((coords.shape[0], 1))])
    return (img.affine @ homog.T).T[:, :3]


def rotation_error_deg(Ra, Rb):
    """Geodesic angle between two rotation matrices, in degrees."""
    Rrel = Ra @ Rb.T
    cos = (np.trace(Rrel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def gt_metrics(src_pts, T_pred, T_gt, gt_frame, src_center, tgt_center):
    """Rotation error (deg) and TRE (mm) of T_pred against the ground truth.

    'raw'      : T_gt maps STL mm -> CBCT mm directly.
    'centered' : T_gt is expressed in the challenge's centred frame (source and
                 target each shifted to their centroid); we compare there and the
                 distances are already in mm because we do not rescale.
    """
    if gt_frame == "centered":
        s = src_pts - src_center
        Tg = T_gt.copy()
        Tg[:3, 3] += (tgt_center - src_center)      # as in data/dataset.py
        pred = R.apply_transform(s, T_pred) - (tgt_center - src_center)
        gtp = R.apply_transform(s, Tg) - (tgt_center - src_center)
        rot = rotation_error_deg(T_pred[:3, :3], Tg[:3, :3])
    else:                                           # raw mm
        pred = R.apply_transform(src_pts, T_pred)
        gtp = R.apply_transform(src_pts, T_gt)
        rot = rotation_error_deg(T_pred[:3, :3], T_gt[:3, :3])
    tre = float(np.linalg.norm(pred - gtp, axis=1).mean())
    return rot, tre


def find_cases(root):
    img_dir = os.path.join(root, "Images")
    if not os.path.isdir(img_dir):
        raise SystemExit(f"Expected {img_dir}/. Download the STSR data there first.")
    return sorted(d for d in os.listdir(img_dir)
                  if os.path.isdir(os.path.join(img_dir, d)))


def main():
    ap = argparse.ArgumentParser(description="Benchmark registration on public STSR data.")
    ap.add_argument("--root", required=True, help="STSR dataset root (contains Images/, Labels/)")
    ap.add_argument("--out", default="./outputs/stsr", help="output directory")
    ap.add_argument("--thr", type=float, default=800.0, help="CBCT bone/teeth threshold (HU)")
    ap.add_argument("--gt-frame", choices=["raw", "centered"], default="raw",
                    help="coordinate frame of the GT transform (see module docstring)")
    args = ap.parse_args()
    np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for case in find_cases(args.root):
        cdir = os.path.join(args.root, "Images", case)
        cbct = os.path.join(cdir, "CBCT.nii.gz")
        if not os.path.exists(cbct):
            continue
        tgt = cbct_surface_points(cbct, thr=args.thr)
        tgt_center = tgt.mean(0)
        for jaw in ("lower", "upper"):
            stl = os.path.join(cdir, f"{jaw}.stl")
            if not os.path.exists(stl):
                continue
            row = {"case": case, "jaw": jaw}
            try:
                src = R.load_mesh_points(stl)
                res = R.register(src, tgt)
                T_pred = res["T_ios_to_cbct"]
                m = surface_metrics(R.apply_transform(src, T_pred), tgt)
                row.update(nsd05=m["nsd05"], assd=m["assd"], rmsd=m["rmse"],
                           overlap=res["strict_fitness"], mirrored=res["mirrored"])

                gt = os.path.join(args.root, "Labels", case, f"{jaw}_gt.npy")
                if os.path.exists(gt):
                    rot, tre = gt_metrics(src, T_pred, np.load(gt).astype(float),
                                          args.gt_frame, src.mean(0), tgt_center)
                    row.update(rot_err_deg=rot, tre_mm=tre)
                msg = "NSD=%.3f ASSD=%.3f RMSD=%.3f" % (m["nsd05"], m["assd"], m["rmse"])
                if "tre_mm" in row:
                    msg += "  rotErr=%.2fdeg TRE=%.3fmm" % (row["rot_err_deg"], row["tre_mm"])
                print("%-14s %-6s %s" % (case, jaw, msg), flush=True)
            except Exception as e:
                row["error"] = str(e)
                print("%-14s %-6s ERROR: %s" % (case, jaw, e), flush=True)
            rows.append(row)

    metric_rows = [r for r in rows if "nsd05" in r]
    if not metric_rows:
        raise SystemExit("No cases evaluated. Check --root layout (Images/<case>/...).")

    print("\n--- summary (n=%d) ---" % len(metric_rows))
    for k, lbl in [("nsd05", "NSD@0.5"), ("assd", "ASSD (mm)"), ("rmsd", "RMSD (mm)")]:
        v = [r[k] for r in metric_rows]
        print("  %-10s %.3f +/- %.3f" % (lbl, np.mean(v), _std(v)))
    tre_rows = [r for r in metric_rows if "tre_mm" in r]
    if tre_rows:
        for k, lbl in [("rot_err_deg", "Rot err (deg)"), ("tre_mm", "TRE (mm)")]:
            v = [r[k] for r in tre_rows]
            print("  %-13s %.3f +/- %.3f   (n=%d with GT)" % (lbl, np.mean(v), _std(v), len(tre_rows)))
    else:
        print("  (no ground-truth transforms found -> surface metrics only)")

    csv_path = os.path.join(args.out, "stsr_metrics.csv")
    cols = ["case", "jaw", "nsd05", "assd", "rmsd", "overlap", "mirrored",
            "rot_err_deg", "tre_mm", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nWrote", csv_path)

    tex_path = os.path.join(args.out, "stsr_table.tex")
    write_tex(metric_rows, tre_rows, tex_path)
    print("Wrote", tex_path, "(paste-ready, measured numbers only)")


def write_tex(metric_rows, tre_rows, path):
    """Paste-ready summary table of the external benchmark (measured means)."""
    def line(label, key, subset):
        v = [r[key] for r in subset]
        return "    %s & $%.3f \\pm %.3f$ \\\\" % (label, np.mean(v), _std(v))

    n = len(metric_rows)
    lines = [
        "% Auto-generated by stsr_eval.py from measured data. Do not edit numbers by hand.",
        "\\begin{table}[htbp]", "  \\centering",
        ("  \\caption{External registration benchmark on the public MICCAI STSR "
         "Task-2 paired CBCT/IOS dataset ($n=%d$ jaws). NSD/ASSD/RMSD are surface "
         "metrics as in Table~\\ref{tab:registration_metrics}; rotation error and "
         "TRE are computed against the dataset's ground-truth transforms and are "
         "therefore independent of our surface-overlap score.}" % n),
        "  \\label{tab:stsr}",
        "  \\begin{tabular}{lc}", "    \\toprule",
        "    Metric & Mean $\\pm$ SD \\\\", "    \\midrule",
        line("NSD@0.5\\,mm", "nsd05", metric_rows),
        line("ASSD (mm)", "assd", metric_rows),
        line("RMSD (mm)", "rmsd", metric_rows),
    ]
    if tre_rows:
        lines += ["    \\midrule",
                  line("Rotation error (deg)", "rot_err_deg", tre_rows),
                  line("TRE (mm)", "tre_mm", tre_rows)]
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
