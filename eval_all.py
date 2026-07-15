"""Metrics for an extended set of >=10 patients: the gingiva-removed cases plus a
few whose only intraoral scan is the raw OrthoCAD shell (gingiva included)."""

import os
import glob
import re
import numpy as np
import registration as R
from eval_metrics import surface_metrics

np.random.seed(0)
IOS_ROOT = R.IOS_ROOT


def seg_dir(num):
    for d in sorted(os.listdir(R.SEG_ROOT)):
        full = os.path.join(R.SEG_ROOT, d)
        if os.path.isdir(full) and re.search(r"0*%s\b" % num, d):
            return full
    return None


def teeth_instance(num):
    sd = seg_dir(num)
    if not sd:
        return None
    hits = glob.glob(os.path.join(sd, "input_resample_teeth_instance", "*.nii.gz"))
    return hits[0] if hits else None


def shell(num, arch):
    hits = glob.glob(os.path.join(IOS_ROOT, "Patient" + num, "**", "*shell_occlusion_*.ply"),
                     recursive=True)
    hits = [h for h in hits if "registration" not in h.lower()]
    for h in hits:
        if R.arch_from_filename(h) == arch:
            return h
    return None


# (patient, ios_path, arch); shells flagged with gingiva=True
pats = R.discover_patients()


def wg(num, tag):
    p = pats[num]
    cands = [c for c in p["ios_candidates"] if not tag or tag in c.lower()]
    return cands[0]


JOBS = [
    ("05", wg("05", None), "upper", False),
    ("08", wg("08", None), "lower", False),
    ("11", shell("11", "lower"), "lower", True),
    ("12", shell("12", "lower"), "lower", True),
    ("19", wg("19", "_l_"), "lower", False),
    ("19", wg("19", "_u_"), "upper", False),
    ("20", wg("20", "_l_"), "lower", False),
    ("21", wg("21", "_l_"), "lower", False),
    ("22", wg("22", "_l_"), "lower", False),
    ("23", wg("23", None), "upper", False),
    ("26", wg("26", None), "upper", False),
    ("28", wg("28", None), "lower", False),
    ("36", shell("36", "lower"), "lower", True),
]

rows = []
for num, ios, arch, gingiva in JOBS:
    ti = teeth_instance(num)
    p = {"number": num, "teeth_instance": ti, "pulp_instance": None,
         "teeth_binary": None, "pulp_binary": None, "has_arch_split": True,
         "ios_dir": IOS_ROOT}
    res = R.register_patient(p, ios, arch=arch)
    teeth = R.nifti_surface_points(ti, labels=R.arch_labels(res["arch"]))
    ios_reg = R.apply_transform(R.load_mesh_points(ios), res["T_ios_to_cbct"])
    m = surface_metrics(ios_reg, teeth)
    m.update(num=num, arch=res["arch"], gingiva=gingiva)
    rows.append(m)
    print("P%s %-5s %-8s NSD=%.3f ASSD=%.3f RMSE=%.3f"
          % (num, res["arch"], "(shell)" if gingiva else "", m["nsd05"], m["assd"], m["rmse"]),
          flush=True)


def stats(keys, subset):
    return {k: (np.mean([r[k] for r in subset]), np.std([r[k] for r in subset])) for k in keys}


keys = ["nsd05", "assd", "rmse"]
allm = stats(keys, rows)
clean = stats(keys, [r for r in rows if r["num"] not in ("05", "08")])
print("MEAN all: " + " ".join("%s=%.3f±%.3f" % (k, *allm[k]) for k in keys), flush=True)
print("MEAN excl05,08: " + " ".join("%s=%.3f±%.3f" % (k, *clean[k]) for k in keys), flush=True)
