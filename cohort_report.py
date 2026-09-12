"""
Full-cohort registration report.

Runs the registration pipeline on EVERY discovered paired CBCT/IOS case (point
the pipeline at your data with the SEG_ROOT / IOS_ROOT environment variables, see
registration.py), measures NSD@0.5 / ASSD / RMSD per case, flags each case as a
successful or failed registration, and writes:

  <out>/cohort_metrics.csv   one row per case (real measurements)
  <out>/cohort_table.tex     paste-ready LaTeX table (real numbers only)

It also prints the mean +/- std over all evaluated cases AND over the successful
subset, plus the success rate. Nothing is hard-coded: every value is measured
from your data, so the table cannot contain invented rows.

Usage:
  SEG_ROOT=/path/to/segmentations IOS_ROOT=/path/to/scans \
    python cohort_report.py --out ./outputs/cohort --success-nsd 0.5
"""

import argparse
import csv
import os
import numpy as np

import registration as R
from eval_metrics import surface_metrics


def _std(vals):
    """Sample standard deviation (ddof=1); 0.0 when fewer than two values."""
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _num_key(x):
    return (0, int(x)) if str(x).isdigit() else (1, str(x))


def diagnose(ios_mode):
    """Print why cases are or aren't discovered, so a 31 -> 10 gap is visible."""
    def index(root):
        out = {}
        if os.path.isdir(root):
            for d in sorted(os.listdir(root)):
                full = os.path.join(root, d)
                if os.path.isdir(full):
                    n = R._patient_number(d)
                    if n:
                        out[n] = full
        return out

    seg, ios = index(R.SEG_ROOT), index(R.IOS_ROOT)
    print("=== discovery (ios_mode=%s) ===" % ios_mode)
    print("SEG_ROOT %s  -> %d patient folders" % (R.SEG_ROOT, len(seg)))
    print("IOS_ROOT %s  -> %d patient folders" % (R.IOS_ROOT, len(ios)))
    seg_only = sorted(set(seg) - set(ios), key=_num_key)
    ios_only = sorted(set(ios) - set(seg), key=_num_key)
    if seg_only:
        print("  segmentation but NO matching IOS folder:", ", ".join(seg_only))
    if ios_only:
        print("  IOS but NO matching segmentation folder:", ", ".join(ios_only))
    usable = 0
    for n in sorted(set(seg) & set(ios), key=_num_key):
        teeth = (R._find_seg_file(seg[n], R.TEETH_INSTANCE_SUBDIR)
                 or R._find_seg_file(seg[n], R.TEETH_BINARY_SUBDIR))
        cands = R._find_ios_candidates(ios[n], mode=ios_mode)
        flag = ""
        if teeth is None:
            flag += " [no teeth segmentation]"
        if not cands:
            flag += " [no IOS file matched]"
        if teeth is not None and cands:
            usable += 1
        print("  P%-5s teeth=%-3s ios_candidates=%d%s"
              % (n, "yes" if teeth else "NO", len(cands), flag))
    print("=> %d patient(s) usable for evaluation\n" % usable)


def evaluate(success_nsd, ios_mode):
    pats = R.discover_patients(ios_mode=ios_mode)
    if not pats:
        raise SystemExit("No paired cases found. Set SEG_ROOT / IOS_ROOT to your data "
                         "(run again to see the discovery diagnostic above).")
    rows = []
    for num in sorted(pats, key=lambda x: int(x)):
        p = pats[num]
        for ios in p["ios_candidates"]:
            row = {"patient": num, "ios": os.path.basename(ios)}
            try:
                res = R.register_patient(p, ios, arch="auto")
                arch = res["arch"]
                ios_reg = R.apply_transform(R.load_mesh_points(ios), res["T_ios_to_cbct"])
                teeth = R.nifti_surface_points(
                    p["teeth_instance"] or p["teeth_binary"],
                    labels=R.arch_labels(arch) if p["teeth_instance"] else None)
                m = surface_metrics(ios_reg, teeth)
                row.update(arch=arch, nsd05=m["nsd05"], assd=m["assd"], rmsd=m["rmse"],
                           hd95=m["hd95"], overlap=res["strict_fitness"],
                           mirrored=res["mirrored"])
                row["success"] = bool(m["nsd05"] >= success_nsd)
                print("P%-3s %-6s NSD=%.3f ASSD=%.3f RMSD=%.3f overlap=%.2f %s"
                      % (num, arch, m["nsd05"], m["assd"], m["rmse"], res["strict_fitness"],
                         "ok" if row["success"] else "FAIL"), flush=True)
            except Exception as e:                 # keep going; a failure is a data point
                row.update(arch="?", success=False, error=str(e))
                print("P%-3s ERROR: %s" % (num, e), flush=True)
            rows.append(row)
    return rows


def summarise(rows, label, subset):
    keys = ["nsd05", "assd", "rmsd"]
    have = [r for r in subset if "nsd05" in r]
    if not have:
        return None
    stats = {k: (float(np.mean([r[k] for r in have])), _std([r[k] for r in have]))
             for k in keys}
    print("%-22s n=%d  " % (label, len(have))
          + "  ".join("%s=%.3f+/-%.3f" % (k, *stats[k]) for k in keys), flush=True)
    return len(have), stats


def write_csv(rows, path):
    cols = ["patient", "arch", "ios", "nsd05", "assd", "rmsd", "hd95",
            "overlap", "mirrored", "success", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_tex(rows, path, success_nsd):
    metric_rows = [r for r in rows if "nsd05" in r]
    allm = {k: (np.mean([r[k] for r in metric_rows]), _std([r[k] for r in metric_rows]))
            for k in ("nsd05", "assd", "rmsd")} if metric_rows else None
    good = [r for r in metric_rows if r.get("success")]
    goodm = {k: (np.mean([r[k] for r in good]), _std([r[k] for r in good]))
             for k in ("nsd05", "assd", "rmsd")} if good else None
    n_eval = len(metric_rows)
    n_ok = len(good)

    lines = [
        "% Auto-generated by cohort_report.py from measured data. Do not edit numbers by hand.",
        "\\begin{table}[htbp]", "  \\centering",
        ("  \\caption{Registration quality over the full cohort. Of %d evaluated "
         "cases, %d met the NSD$\\geq$%.2f success criterion (success rate %.0f\\%%). "
         "NSD is the normalized surface Dice at 0.5\\,mm; ASSD and RMSD are the "
         "average and root-mean-square symmetric surface distances (mm).}"
         % (n_eval, n_ok, success_nsd, (100.0 * n_ok / n_eval) if n_eval else 0)),
        "  \\label{tab:registration_metrics}",
        "  \\begin{tabular}{lccc}", "    \\toprule",
        "    Case & NSD@0.5\\,mm & ASSD & RMSD \\\\", "    \\midrule",
    ]
    for r in sorted(metric_rows, key=lambda r: (int(r["patient"]), r.get("arch", ""))):
        flag = "" if r.get("success") else "$^\\ast$"
        lines.append("    %s (%s)%s & %.2f & %.2f & %.2f \\\\"
                     % (r["patient"], str(r.get("arch", "")).capitalize(), flag,
                        r["nsd05"], r["assd"], r["rmsd"]))
    lines.append("    \\midrule")
    if goodm:
        lines.append("    Mean (successful, $n{=}%d$) & $%.2f \\pm %.2f$ & $%.2f \\pm %.2f$ & $%.2f \\pm %.2f$ \\\\"
                     % (n_ok, *goodm["nsd05"], *goodm["assd"], *goodm["rmsd"]))
    if allm:
        lines.append("    Mean (all evaluated, $n{=}%d$) & $%.2f \\pm %.2f$ & $%.2f \\pm %.2f$ & $%.2f \\pm %.2f$ \\\\"
                     % (n_eval, *allm["nsd05"], *allm["assd"], *allm["rmsd"]))
    lines += ["    \\bottomrule", "  \\end{tabular}",
              "  \\\\[2pt] {\\footnotesize $^\\ast$ below the NSD success threshold.}",
              "\\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_summary_tex(rows, path):
    """Compact NSD/ASSD/RMSD summary: mean +/- SD over all evaluated cases and
    over the successful subset (measured numbers only)."""
    metric_rows = [r for r in rows if "nsd05" in r]
    good = [r for r in metric_rows if r.get("success")]

    def cell(subset, key):
        v = [r[key] for r in subset]
        return "$%.2f \\pm %.2f$" % (np.mean(v), _std(v)) if v else "--"

    labels = [("NSD@0.5\\,mm", "nsd05"), ("ASSD (mm)", "assd"), ("RMSD (mm)", "rmsd")]
    lines = [
        "% Auto-generated by cohort_report.py from measured data. Do not edit numbers by hand.",
        "\\begin{table}[htbp]", "  \\centering",
        ("  \\caption{Registration accuracy summarised over the cohort. NSD is the "
         "normalized surface Dice at a 0.5\\,mm tolerance; ASSD and RMSD are the "
         "average and root-mean-square symmetric surface distances (mm), evaluated "
         "over the inlier region. Means are reported over all evaluated cases and "
         "over the successfully registered subset.}"),
        "  \\label{tab:summary}",
        "  \\begin{tabular}{lcc}", "    \\toprule",
        "    Metric & All evaluated ($n=%d$) & Successful ($n=%d$) \\\\"
        % (len(metric_rows), len(good)),
        "    \\midrule",
    ]
    for lbl, key in labels:
        lines.append("    %s & %s & %s \\\\" % (lbl, cell(metric_rows, key), cell(good, key)))
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Evaluate the full CBCT/IOS cohort.")
    ap.add_argument("--out", default="./outputs/cohort", help="output directory")
    ap.add_argument("--success-nsd", type=float, default=0.5,
                    help="a case counts as successful if NSD@0.5 >= this (default 0.5)")
    ap.add_argument("--ios-mode", choices=["all", "clean"], default="all",
                    help="'all' (default) evaluates every scan incl. shells/STL, one "
                         "per arch; 'clean' only gingiva-removed .ply files")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    diagnose(args.ios_mode)
    rows = evaluate(args.success_nsd, args.ios_mode)

    print("\n--- summary ---")
    summarise(rows, "Mean (all evaluated)", rows)
    summarise(rows, "Mean (successful)", [r for r in rows if r.get("success")])
    n_eval = len([r for r in rows if "nsd05" in r])
    n_ok = len([r for r in rows if r.get("success")])
    print("Success rate: %d/%d (%.0f%%)"
          % (n_ok, n_eval, (100.0 * n_ok / n_eval) if n_eval else 0))

    csv_path = os.path.join(args.out, "cohort_metrics.csv")
    tex_path = os.path.join(args.out, "cohort_table.tex")
    summary_path = os.path.join(args.out, "cohort_summary.tex")
    write_csv(rows, csv_path)
    write_tex(rows, tex_path, args.success_nsd)
    write_summary_tex(rows, summary_path)
    print("\nWrote", csv_path)
    print("Wrote", tex_path, "(per-case table, paste-ready)")
    print("Wrote", summary_path, "(NSD/ASSD/RMSD summary, paste-ready)")


if __name__ == "__main__":
    main()
