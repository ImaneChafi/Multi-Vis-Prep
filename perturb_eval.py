"""
Robustness / capture-range evaluation by perturbation recovery.

The surface-distance metrics are measured at the pose the pipeline selects, so
they cannot show from how far away that pose is reached. This experiment uses a
transform we control as ground truth, independent of the surface-overlap score.

For each successfully registered case we take the converged alignment as a
reference pose, then apply N random rigid perturbations (rotation up to
--rot-max degrees about a random axis, translation up to --trans-max mm in a
random direction), re-run the FULL pipeline (global init, mirror handling,
coarse-to-fine ICP) from each perturbed scan, and measure the target
registration error (TRE): the mean distance between the re-registered scan and
the reference pose over all scan points. A perturbation is "recovered" when its
TRE < --success-mm. The perturbation range also quantifies the initial
misalignment the pipeline tolerates.

Writes:
  <out>/perturb_metrics.csv   per case: mean/median TRE, recovery rate
  <out>/perturb_table.tex     paste-ready LaTeX summary (measured numbers only)

Usage:
  SEG_ROOT=/path/to/segmentations IOS_ROOT=/path/to/scans \
    python perturb_eval.py --out ./outputs/perturb \
      --n-perturb 20 --rot-max 30 --trans-max 10
"""

import argparse
import csv
import os
import numpy as np
from scipy.spatial.transform import Rotation

import registration as R
from eval_metrics import surface_metrics
from cohort_report import diagnose


def _std(vals):
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def random_rigid(rot_max_deg, trans_max_mm, rng):
    """A 4x4 rigid transform: random axis, angle in [0, rot_max], translation of
    random direction and magnitude in [0, trans_max]."""
    axis = rng.normal(size=3)
    axis /= (np.linalg.norm(axis) + 1e-12)
    angle = np.radians(rng.uniform(0, rot_max_deg))
    Rm = Rotation.from_rotvec(axis * angle).as_matrix()
    tdir = rng.normal(size=3)
    tdir /= (np.linalg.norm(tdir) + 1e-12)
    t = tdir * rng.uniform(0, trans_max_mm)
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = t
    return T


def evaluate(args):
    rng = np.random.default_rng(0)
    pats = R.discover_patients(ios_mode=args.ios_mode)
    if not pats:
        raise SystemExit("No paired cases found. Set SEG_ROOT / IOS_ROOT to your data "
                         "(see the discovery diagnostic above).")
    rows = []
    for num in sorted(pats, key=lambda x: int(x)):
        p = pats[num]
        for ios in p["ios_candidates"]:
            try:
                ref = R.register_patient(p, ios, arch="auto")
                arch = ref["arch"]
                teeth = R.nifti_surface_points(
                    p["teeth_instance"] or p["teeth_binary"],
                    labels=R.arch_labels(arch) if p["teeth_instance"] else None)
                ios_pts = R.load_mesh_points(ios)
                aligned = R.apply_transform(ios_pts, ref["T_ios_to_cbct"])  # reference pose
                if surface_metrics(aligned, teeth)["nsd05"] < args.success_nsd:
                    print("P%-3s %-6s skipped (reference did not register)" % (num, arch),
                          flush=True)
                    continue
            except Exception as e:
                print("P%-3s reference ERROR: %s" % (num, e), flush=True)
                continue

            tres = []
            for _ in range(args.n_perturb):
                P = random_rigid(args.rot_max, args.trans_max, rng)
                perturbed = R.apply_transform(aligned, P)
                try:
                    rec = R.register(perturbed, teeth)
                    back = R.apply_transform(perturbed, rec["T_ios_to_cbct"])
                    tres.append(float(np.linalg.norm(back - aligned, axis=1).mean()))
                except Exception:
                    tres.append(float("inf"))       # a crash counts as a non-recovery
            tres = np.array(tres)
            recovered = tres < args.success_mm
            finite = tres[np.isfinite(tres)]
            row = {"patient": num, "arch": arch, "n": args.n_perturb,
                   "tre_mean": float(finite.mean()) if finite.size else float("inf"),
                   "tre_median": float(np.median(finite)) if finite.size else float("inf"),
                   "recovery_rate": float(recovered.mean())}
            rows.append(row)
            print("P%-3s %-6s TRE(mean)=%.3f mm  recovered=%d/%d"
                  % (num, arch, row["tre_mean"], int(recovered.sum()), args.n_perturb),
                  flush=True)
    return rows


def write_csv(rows, path):
    cols = ["patient", "arch", "n", "tre_mean", "tre_median", "recovery_rate"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_tex(rows, path, args):
    # Cohort-level recovery rate is over all perturbations, weighted by count.
    total = sum(r["n"] for r in rows)
    rec = sum(r["recovery_rate"] * r["n"] for r in rows)
    rate = (100.0 * rec / total) if total else 0.0
    tre_ok = [r["tre_mean"] for r in rows if np.isfinite(r["tre_mean"])]
    mean_tre, sd_tre = (np.mean(tre_ok), _std(tre_ok)) if tre_ok else (0.0, 0.0)
    lines = [
        "% Auto-generated by perturb_eval.py from measured data. Do not edit numbers by hand.",
        "\\begin{table}[htbp]", "  \\centering",
        ("  \\caption{Robustness to initialisation. Each of the %d successfully "
         "registered scans was perturbed by %d random rigid transforms (rotation "
         "up to %g$^\\circ$, translation up to %g\\,mm) and re-registered from "
         "scratch. TRE is the mean distance back to the reference pose; a "
         "perturbation is recovered when TRE $<$ %g\\,mm.}"
         % (len(rows), args.n_perturb, args.rot_max, args.trans_max, args.success_mm)),
        "  \\label{tab:robustness}",
        "  \\begin{tabular}{lc}", "    \\toprule",
        "    Quantity & Value \\\\", "    \\midrule",
        "    Cases $\\times$ perturbations & %d $\\times$ %d \\\\" % (len(rows), args.n_perturb),
        "    Perturbation range & up to %g$^\\circ$, %g\\,mm \\\\" % (args.rot_max, args.trans_max),
        "    Recovery rate (TRE $<$ %g\\,mm) & %.0f\\%% \\\\" % (args.success_mm, rate),
        "    Mean TRE (mm) & $%.3f \\pm %.3f$ \\\\" % (mean_tre, sd_tre),
        "    \\bottomrule", "  \\end{tabular}", "\\end{table}", "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Perturbation-recovery robustness test.")
    ap.add_argument("--out", default="./outputs/perturb")
    ap.add_argument("--n-perturb", type=int, default=20, help="perturbations per case")
    ap.add_argument("--rot-max", type=float, default=30.0, help="max rotation (deg)")
    ap.add_argument("--trans-max", type=float, default=10.0, help="max translation (mm)")
    ap.add_argument("--success-mm", type=float, default=1.0, help="TRE recovery threshold (mm)")
    ap.add_argument("--success-nsd", type=float, default=0.5,
                    help="reference case is used only if its NSD@0.5 >= this")
    ap.add_argument("--ios-mode", choices=["all", "clean"], default="all",
                    help="'all' (default) uses every scan incl. shells/STL, one per "
                         "arch; 'clean' only gingiva-removed .ply files")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    diagnose(args.ios_mode)
    rows = evaluate(args)
    if not rows:
        raise SystemExit("No cases with a usable reference registration.")

    total = sum(r["n"] for r in rows)
    rec = sum(r["recovery_rate"] * r["n"] for r in rows)
    tre_ok = [r["tre_mean"] for r in rows if np.isfinite(r["tre_mean"])]
    print("\n--- summary ---")
    print("Cases: %d   Perturbations each: %d" % (len(rows), args.n_perturb))
    print("Recovery rate (TRE < %g mm): %.0f%%" % (args.success_mm, 100.0 * rec / total))
    if tre_ok:
        print("Mean TRE: %.3f +/- %.3f mm" % (np.mean(tre_ok), _std(tre_ok)))

    write_csv(rows, os.path.join(args.out, "perturb_metrics.csv"))
    write_tex(rows, os.path.join(args.out, "perturb_table.tex"), args)
    print("\nWrote", os.path.join(args.out, "perturb_metrics.csv"))
    print("Wrote", os.path.join(args.out, "perturb_table.tex"), "(paste-ready, measured numbers only)")


if __name__ == "__main__":
    main()
