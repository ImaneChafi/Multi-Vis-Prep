"""
Command-line registration: give it the two CBCT segmentations (teeth and pulp)
and an intraoral scan; it writes the transform and the registered IOS mesh.

Example:
  python register_cli.py --teeth teeth.nii.gz --pulp pulp.nii.gz \
      --ios scan.ply --arch auto --out ./outputs/case01
"""

import argparse
import os
import json
import numpy as np
import open3d as o3d

import registration as R


def main():
    ap = argparse.ArgumentParser(description="Register an intraoral scan to a CBCT tooth segmentation.")
    ap.add_argument("--teeth", required=True, help="CBCT tooth segmentation (NIfTI, .nii/.nii.gz)")
    ap.add_argument("--pulp", required=True, help="CBCT pulp segmentation (NIfTI, .nii/.nii.gz)")
    ap.add_argument("--ios", required=True, help="Intraoral scan surface mesh (.ply/.stl)")
    ap.add_argument("--arch", default="auto", choices=["auto", "upper", "lower", "both"],
                    help="Which arch to register (default: auto)")
    ap.add_argument("--out", default="./outputs/case", help="Output directory")
    args = ap.parse_args()

    for f in [args.teeth, args.pulp, args.ios]:
        if not os.path.exists(f):
            raise SystemExit(f"File not found: {f}")

    print("Registering ...")
    res = R.register_files(args.teeth, args.ios, arch=args.arch, pulp_path=args.pulp)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "T_ios_to_cbct.npy"), res["T_ios_to_cbct"])
    np.save(os.path.join(args.out, "T_cbct_to_ios.npy"), res["T_cbct_to_ios"])

    # Save the registered IOS mesh (in CBCT space) for inspection in any viewer.
    m = res["meshes"]["ios"]
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(m["vertices"], dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(m["faces"], dtype=np.int32)))
    o3d.io.write_triangle_mesh(os.path.join(args.out, "ios_registered.ply"), mesh)

    summary = {
        "arch": res["arch"], "mirrored": res["mirrored"],
        "strict_overlap_0.5mm": round(res["strict_fitness"], 3),
        "rmsd_mm": round(res["rmse"], 3),
        "coverage_1.5mm": round(res["fitness"], 3),
        "T_ios_to_cbct": res["T_ios_to_cbct"].tolist(),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"arch={res['arch']}  mirrored={res['mirrored']}  "
          f"overlap@0.5mm={res['strict_fitness']:.3f}  RMSD={res['rmse']:.3f} mm")
    print(f"Saved transform + registered mesh to: {args.out}")


if __name__ == "__main__":
    main()
