"""
Compute registration-quality metrics for every case that has an intraoral scan,
including the low-quality ones. Because the IOS is a partial (crown-only) open
surface and the CBCT teeth are a full closed volume, a volumetric Dice is not
meaningful; we report a SURFACE Dice (Normalized Surface Dice, NSD) plus the
standard surface-distance metrics, all evaluated over the region of mutual
overlap (the CBCT tooth surface within BAND mm of the registered IOS, i.e. the
crown region the scan actually covers).

Metrics per case:
  NSD@0.5, NSD@1.0 : surface Dice at 0.5 / 1.0 mm tolerance
  ASSD            : average symmetric surface distance (mm)
  HD95            : 95th-percentile symmetric Hausdorff distance (mm)
  Chamfer         : symmetric mean surface distance sum (mm)
"""

import json
import numpy as np
from scipy.spatial import cKDTree
import registration as R

BAND = 3.0   # mm: defines the overlapping crown region on the CBCT side


MATCH = 0.5   # mm: inlier region for the accuracy metrics (ASSD, RMSE),
              # matching the RMSE reported by the GUI and the montage figure.


def surface_metrics(ios_reg, teeth):
    t_tree = cKDTree(teeth)
    i_tree = cKDTree(ios_reg)
    d_i2t, _ = t_tree.query(ios_reg)          # IOS -> CBCT
    # Restrict CBCT to the crown region the IOS covers.
    d_all_t, _ = i_tree.query(teeth)
    crown = teeth[d_all_t <= BAND]
    d_t2i, _ = i_tree.query(crown)            # CBCT(crown) -> IOS

    n = len(d_i2t) + len(d_t2i)

    # NSD (surface Dice / completeness): over the full IOS and crown surfaces.
    def nsd(tau):
        return ((d_i2t <= tau).sum() + (d_t2i <= tau).sum()) / n

    # ASSD / RMSE (accuracy): over the overlapping region only (matched within
    # MATCH mm), so non-overlapping parts of a failed scan don't dominate. This
    # matches the accuracy the GUI reports.
    mi = d_i2t[d_i2t <= MATCH]
    mt = d_t2i[d_t2i <= MATCH]
    both_m = np.concatenate([mi, mt])

    return {
        "nsd05": float(nsd(0.5)),
        "assd": float((mi.mean() + mt.mean()) / 2),
        "rmse": float(np.sqrt((both_m ** 2).mean())),
        "hd95": float(np.percentile(both_m, 95)),
        "chamfer": float(mi.mean() + mt.mean()),
    }


def main():
    np.random.seed(0)
    pats = R.discover_patients()
    rows = []
    for num in sorted(pats, key=int):
        p = pats[num]
        for ios in p["ios_candidates"]:
            res = R.register_patient(p, ios, arch="auto")
            arch = res["arch"]
            ios_pts = R.load_mesh_points(ios)
            teeth = R.nifti_surface_points(
                p["teeth_instance"] or p["teeth_binary"],
                labels=R.arch_labels(arch) if p["teeth_instance"] else None)
            ios_reg = R.apply_transform(ios_pts, res["T_ios_to_cbct"])
            m = surface_metrics(ios_reg, teeth)
            m.update(patient=num, arch=arch)
            rows.append(m)
            print("P%s %-5s NSD@0.5=%.3f ASSD=%.3f RMSE=%.3f HD95=%.3f Chamfer=%.3f"
                  % (num, arch, m["nsd05"], m["assd"], m["rmse"], m["hd95"], m["chamfer"]),
                  flush=True)

    keys = ["nsd05", "assd", "rmse", "hd95", "chamfer"]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    std = {k: float(np.std([r[k] for r in rows])) for k in keys}
    print("MEAN " + " ".join("%s=%.3f±%.3f" % (k, mean[k], std[k]) for k in keys), flush=True)
    json.dump({"rows": rows, "mean": mean, "std": std}, open("/tmp/eval_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
