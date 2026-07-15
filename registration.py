"""
IOS <-> CBCT-segmentation rigid registration.

Method (surface-based, uses BOTH inputs):
  1. Convert the CBCT teeth segmentation (NIfTI label volume) into a surface
     point cloud in world/mm coordinates using its affine.
  2. Sample the intraoral scan (IOS) mesh into a point cloud.
  3. Coarse global alignment with FPFH features + RANSAC (the two modalities
     live in completely different coordinate frames, so a global step is needed).
  4. Refine with point-to-plane ICP.
  5. The IOS is the ICP *source* (its crowns are fully covered by the CBCT
     crowns, which makes the fitness metric meaningful). The resulting 4x4
     maps IOS -> CBCT world space. The pulp shares the teeth affine, so the
     same/ inverse transform positions teeth + pulp relative to the IOS.

This is the ICP surface-registration approach: the PointNetLK network from the
MICCAI repo never takes the CBCT as input, so it cannot register against your
segmentation. This method actually consumes both modalities.
"""

import os
import re
import glob
import numpy as np
import nibabel as nib
import open3d as o3d
from scipy import ndimage

# ----------------------------------------------------------------------------
# Paths. Override with the SEG_ROOT / IOS_ROOT environment variables, or edit
# the defaults below.
#   SEG_ROOT: directory of TIPs segmentation outputs (one folder per patient)
#   IOS_ROOT: directory of intraoral scans        (one folder per patient)
# ----------------------------------------------------------------------------
SEG_ROOT = os.environ.get("SEG_ROOT", "./data/segmentations")
IOS_ROOT = os.environ.get("IOS_ROOT", "./data/intraoral_scans")

# Instance files label each tooth by FDI number, which lets us split by arch.
TEETH_INSTANCE_SUBDIR = "input_resample_teeth_instance"
PULP_INSTANCE_SUBDIR = "input_resample_pulps_instance"
# Binary fallbacks (whole mouth, no arch split).
TEETH_BINARY_SUBDIR = "input_resample_teeth_binary"
PULP_BINARY_SUBDIR = "input_resample_pulps_segmentation"

# FDI numbering: upper jaw = quadrants 1 & 2 (11-28), lower jaw = 3 & 4 (31-48).
FDI_UPPER = set(range(11, 29))
FDI_LOWER = set(range(31, 49))


def arch_labels(arch):
    """Return the set of FDI labels for the requested arch ('upper'/'lower'),
    or None for 'both' (meaning: keep every non-zero label)."""
    if arch == "upper":
        return FDI_UPPER
    if arch == "lower":
        return FDI_LOWER
    return None


def arch_from_filename(path):
    """Infer the arch from the IOS filename: a delimited 'l' token means lower,
    'u' means upper (e.g. patient_19_l_without_gingiva.ply -> lower). Returns
    'upper'/'lower', or None when the name carries no such marker."""
    name = os.path.basename(path).lower()
    if re.search(r"_l(?:_|\.)", name):
        return "lower"
    if re.search(r"_u(?:_|\.)", name):
        return "upper"
    return None


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------
def _patient_number(name):
    m = re.search(r"(\d+)", name)
    return m.group(1) if m else None


def discover_patients():
    """Return {patient_number: {seg_dir, ios_dir, teeth, pulp, ios_candidates}}
    for every patient that has BOTH a teeth segmentation and at least one IOS."""
    patients = {}

    # Index segmentation folders by patient number
    seg_by_num = {}
    if os.path.isdir(SEG_ROOT):
        for d in sorted(os.listdir(SEG_ROOT)):
            full = os.path.join(SEG_ROOT, d)
            if os.path.isdir(full):
                num = _patient_number(d)
                if num:
                    seg_by_num[num] = full

    # Index IOS folders by patient number
    ios_by_num = {}
    if os.path.isdir(IOS_ROOT):
        for d in sorted(os.listdir(IOS_ROOT)):
            full = os.path.join(IOS_ROOT, d)
            if os.path.isdir(full):
                num = _patient_number(d)
                if num:
                    ios_by_num[num] = full

    for num, seg_dir in seg_by_num.items():
        if num not in ios_by_num:
            continue
        ios_dir = ios_by_num[num]

        teeth_inst = _find_seg_file(seg_dir, TEETH_INSTANCE_SUBDIR)
        pulp_inst = _find_seg_file(seg_dir, PULP_INSTANCE_SUBDIR)
        teeth_bin = _find_seg_file(seg_dir, TEETH_BINARY_SUBDIR)
        pulp_bin = _find_seg_file(seg_dir, PULP_BINARY_SUBDIR)

        # Need at least one teeth source to register against.
        teeth = teeth_inst or teeth_bin
        if teeth is None:
            continue

        ios_candidates = _find_ios_candidates(ios_dir)
        if not ios_candidates:
            continue

        patients[num] = {
            "number": num,
            "seg_dir": seg_dir,
            "ios_dir": ios_dir,
            "teeth_instance": teeth_inst,
            "pulp_instance": pulp_inst,
            "teeth_binary": teeth_bin,
            "pulp_binary": pulp_bin,
            "has_arch_split": teeth_inst is not None,
            "ios_candidates": ios_candidates,
        }
    return patients


def _find_seg_file(seg_dir, subdir):
    d = os.path.join(seg_dir, subdir)
    if not os.path.isdir(d):
        return None
    files = glob.glob(os.path.join(d, "*.nii.gz"))
    return files[0] if files else None


def _find_ios_candidates(ios_dir):
    """Only the cleaned, gingiva-removed IOS ('*without_gingiva*.ply') so the
    tooth crowns overlay the CBCT segmentation without gingiva occluding it.
    Raw OrthoCAD shells (which include gingiva) and '.vtp.ply' duplicates are
    excluded, as are files inside prior-registration / backup subfolders."""
    skip = ("registration", "test_for_metrics", "backup")

    def ok(p):
        rel = os.path.relpath(p, ios_dir).lower()
        if any(s in rel for s in skip):
            return False
        if p.lower().endswith(".vtp.ply"):
            return False
        return True

    cleaned = [p for p in glob.glob(os.path.join(ios_dir, "**", "*.ply"), recursive=True)
               if "without_gingiva" in os.path.basename(p).lower() and ok(p)]
    return sorted(set(cleaned))


# ----------------------------------------------------------------------------
# Loading / conversion
# ----------------------------------------------------------------------------
def nifti_surface_points(path, max_points=40000, labels=None):
    """Extract boundary (surface) voxels of a mask and return their world-mm
    coordinates as an (N,3) array, using the NIfTI affine.

    labels: if given (set/iterable of ints), keep only those label values
    (used to select one arch from the instance segmentation); otherwise keep
    every non-zero voxel."""
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    if labels is not None:
        mask = np.isin(data, list(labels))
    else:
        mask = data > 0
    if mask.sum() == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Surface = mask voxels that touch background (6-connectivity erosion)
    eroded = ndimage.binary_erosion(mask, iterations=1)
    surface = mask & ~eroded
    if surface.sum() == 0:
        surface = mask  # tiny structure; keep everything

    ijk = np.argwhere(surface)  # (N,3) voxel indices
    if ijk.shape[0] > max_points:
        idx = np.random.choice(ijk.shape[0], max_points, replace=False)
        ijk = ijk[idx]

    ijk_h = np.hstack([ijk, np.ones((ijk.shape[0], 1))])
    world = (img.affine @ ijk_h.T).T[:, :3]
    return world.astype(np.float64)


def load_mesh_points(ply_path, n_points=30000):
    """Load an IOS mesh and sample a point cloud (world/mm)."""
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if len(mesh.vertices) == 0:
        pc = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pc.points)
    elif len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        pc = mesh.sample_points_uniformly(number_of_points=n_points)
        pts = np.asarray(pc.points)
    else:
        pts = np.asarray(mesh.vertices)
    return pts.astype(np.float64)


def _to_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


# ----------------------------------------------------------------------------
# Surface meshes (for display)
# ----------------------------------------------------------------------------
def nifti_to_mesh(path, labels=None, target_tris=20000, smooth_iters=3):
    """Marching-cubes surface of a mask, returned as (vertices_world, faces).
    vertices are in world/mm via the NIfTI affine; faces index into vertices."""
    from skimage import measure

    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    if labels is not None:
        vol = np.isin(data, list(labels))
    else:
        vol = data > 0
    if vol.sum() == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)

    vol = np.pad(vol.astype(np.float32), 1)  # closed borders
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)
    verts -= 1.0  # undo pad -> voxel index space

    # voxel index -> world mm
    verts_world = apply_transform(verts, img.affine)

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_world),
        o3d.utility.Vector3iVector(faces.astype(np.int32)))
    if smooth_iters:
        mesh = mesh.filter_smooth_simple(number_of_iterations=smooth_iters)
    if target_tris and len(mesh.triangles) > target_tris:
        mesh = mesh.simplify_quadric_decimation(int(target_tris))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles, dtype=np.int32)


def load_ios_mesh(ply_path, target_tris=60000):
    """Load an IOS surface mesh; decimate if very dense. Returns (vertices, faces)."""
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if len(mesh.triangles) == 0:
        return np.asarray(mesh.vertices), np.empty((0, 3), dtype=np.int32)
    if target_tris and len(mesh.triangles) > target_tris:
        mesh = mesh.simplify_quadric_decimation(int(target_tris))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles, dtype=np.int32)


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------
def _prep(pcd, voxel):
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return down, fpfh


def _ransac(src_d, tgt_d, src_f, tgt_f, distance_threshold):
    # mutual_filter=False keeps more feature correspondences, which makes each
    # RANSAC shot hit the good basin more often (and runs faster).
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_f, tgt_f, False, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 0.999))


def _pca_axes(points):
    """Centroid and principal axes (columns, descending) of a point set,
    forced right-handed."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    V = vt.T
    if np.linalg.det(V) < 0:
        V[:, -1] = -V[:, -1]
    return c, V


def _pca_inits(src_pts, tgt_pts):
    """Candidate rigid inits that align the two point sets' principal frames.
    Covers the flip ambiguities of a near-symmetric dental arch: the 4 sign
    combinations of the first two axes, for both the natural axis order and the
    swap of the first two axes (needed when the arch's width and length are
    close, so PCA may order those axes differently between IOS and CBCT).
    One of these is always the correct orientation."""
    cs, Vs = _pca_axes(src_pts)
    ct, Vt = _pca_axes(tgt_pts)
    inits = []
    for perm in ((0, 1, 2), (1, 0, 2)):
        Vsp = Vs[:, perm].copy()
        if np.linalg.det(Vsp) < 0:
            Vsp[:, 2] = -Vsp[:, 2]          # keep right-handed after the swap
        for s0 in (1.0, -1.0):
            for s1 in (1.0, -1.0):
                F = np.diag([s0, s1, s0 * s1])   # det = +1 (proper rotation)
                Rm = Vt @ F @ Vsp.T
                T = np.eye(4)
                T[:3, :3] = Rm
                T[:3, 3] = ct - Rm @ cs
                inits.append(T)
    return inits


def _icp_refine(src, tgt, T0, thresholds=(3.0, 1.5, 0.8)):
    """Coarse-to-fine point-to-plane ICP."""
    T = np.asarray(T0, dtype=np.float64)
    for thr in thresholds:
        icp = o3d.pipelines.registration.registration_icp(
            src, tgt, thr, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        T = np.asarray(icp.transformation, dtype=np.float64)
    return T


def register(ios_points, teeth_points, voxel=0.6, n_restarts=3, strict_dist=0.5):
    """Register IOS (source) onto CBCT teeth (target).

    Robust against two ambiguities:

    * Arch flip — dental arches are near-symmetric, so it gathers candidate
      global inits from stochastic FPFH+RANSAC plus PCA principal-axis
      alignments covering all flip signs.
    * Handedness — the iTero/OrthoCAD IOS export is mirrored relative to the
      CBCT (a rigid transform cannot correct a reflection), so it tries both the
      IOS as-is AND a mirrored copy, and keeps whichever wins.

    Every candidate is refined with coarse-to-fine ICP and scored by STRICT
    overlap (fraction of IOS points within strict_dist mm of a tooth surface),
    which a flipped/mirrored-wrong alignment cannot fake. When the mirrored copy
    wins, the returned transform includes that reflection (det < 0).

    Returns the transform, its inverse, loose fitness (@1.5 mm), strict fitness
    (@strict_dist), RMSE, and whether a mirror was applied.
    """
    tgt = _to_pcd(teeth_points)
    tgt_d, tgt_f = _prep(tgt, voxel)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    tgt_pts_d = np.asarray(tgt_d.points)
    distance_threshold = voxel * 1.5

    ios = np.asarray(ios_points, dtype=np.float64)
    mirror = np.eye(4)
    mirror[0, 0] = -1.0                       # reflect across X
    ios_mirrored = ios.copy()
    ios_mirrored[:, 0] = -ios_mirrored[:, 0]

    # Try the IOS as-is and mirrored; pre is the map original-IOS -> variant.
    variants = [(ios, np.eye(4)), (ios_mirrored, mirror)]

    best = None
    for src_arr, pre in variants:
        src = _to_pcd(src_arr)
        src_d, src_f = _prep(src, voxel)
        inits = [_ransac(src_d, tgt_d, src_f, tgt_f, distance_threshold).transformation
                 for _ in range(max(1, n_restarts))]
        inits += _pca_inits(np.asarray(src_d.points), tgt_pts_d)
        # Fast candidate selection on the downsampled clouds.
        for T0 in inits:
            T = _icp_refine(src_d, tgt_d, T0, thresholds=(voxel * 3, voxel * 1.5))
            ev = o3d.pipelines.registration.evaluate_registration(src_d, tgt_d, strict_dist, T)
            if best is None or ev.fitness > best["strict"]:
                best = {"strict": ev.fitness, "T": T, "src": src, "pre": pre}

    # Refine the winning variant at full resolution.
    src = best["src"]
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    Tf = _icp_refine(src, tgt, best["T"], thresholds=(1.5, 0.8, 0.4))

    strict = o3d.pipelines.registration.evaluate_registration(src, tgt, strict_dist, Tf)
    loose = o3d.pipelines.registration.evaluate_registration(src, tgt, 1.5, Tf)

    T_full = Tf @ best["pre"]                  # original IOS -> CBCT (incl. mirror if used)
    return {
        "T_ios_to_cbct": T_full,
        "T_cbct_to_ios": np.linalg.inv(T_full),
        "fitness": float(loose.fitness),
        "rmse": float(strict.inlier_rmse),
        "strict_fitness": float(strict.fitness),
        "mirrored": bool(np.linalg.det(best["pre"]) < 0),
    }


def apply_transform(points, T):
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return pts
    h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (T @ h.T).T[:, :3]


# ----------------------------------------------------------------------------
# High-level orchestration (used by the web app)
# ----------------------------------------------------------------------------
def _teeth_source(patient, arch):
    """Return (teeth_surface_points, labels_used). Uses the instance file with
    an arch filter when available; otherwise the whole-mouth binary file."""
    if patient.get("teeth_instance") and arch in ("upper", "lower"):
        return nifti_surface_points(patient["teeth_instance"], labels=arch_labels(arch))
    src = patient.get("teeth_instance") or patient.get("teeth_binary")
    return nifti_surface_points(src)


def _pulp_source(patient, arch):
    if patient.get("pulp_instance") and arch in ("upper", "lower"):
        pts = nifti_surface_points(patient["pulp_instance"], labels=arch_labels(arch), max_points=20000)
        if pts.shape[0] > 0:
            return pts
    src = patient.get("pulp_instance") or patient.get("pulp_binary")
    if src is None:
        return np.empty((0, 3))
    return nifti_surface_points(src, max_points=20000)


def _teeth_mesh_source(patient, arch):
    """Return (path, labels) for building the teeth surface mesh."""
    if patient.get("teeth_instance") and arch in ("upper", "lower"):
        return patient["teeth_instance"], arch_labels(arch)
    return patient.get("teeth_instance") or patient.get("teeth_binary"), None


def _pulp_mesh_source(patient, arch):
    if patient.get("pulp_instance") and arch in ("upper", "lower"):
        return patient["pulp_instance"], arch_labels(arch)
    src = patient.get("pulp_instance") or patient.get("pulp_binary")
    return src, None


def register_patient(patient, ios_path, arch="auto"):
    """Full registration for one patient.

    arch: 'upper', 'lower', 'both', or 'auto' (try upper & lower, keep the one
    with the higher ICP fitness). Returns a dict with the transform, quality
    metrics, the arch actually used, and downsampled point clouds (for display)
    already expressed in CBCT world space.
    """
    ios = load_mesh_points(ios_path)

    if arch == "auto":
        # If the filename names the arch (…_l_… / …_u_…), trust it and skip the
        # other arch entirely. Otherwise try both (or whole-mouth if no split).
        named = arch_from_filename(ios_path)
        if named:
            candidates = [named]
        elif patient.get("has_arch_split"):
            candidates = ["upper", "lower"]
        else:
            candidates = ["both"]
    else:
        candidates = [arch]

    best = None
    for a in candidates:
        teeth = _teeth_source(patient, a)
        if teeth.shape[0] < 100:
            continue
        res = register(ios, teeth)
        res["arch"] = a
        # Pick the arch by strict overlap (robust to flips / wrong arch).
        if best is None or res["strict_fitness"] > best["strict_fitness"]:
            best = res
    if best is None:
        raise RuntimeError("Registration failed: no usable teeth surface found.")

    arch_used = best["arch"]
    T = best["T_ios_to_cbct"]

    # --- Display meshes (everything in CBCT world space) ---
    teeth_path, teeth_labels = _teeth_mesh_source(patient, arch_used)
    teeth_v, teeth_f = nifti_to_mesh(teeth_path, labels=teeth_labels)

    pulp_path, pulp_labels = _pulp_mesh_source(patient, arch_used)
    if pulp_path:
        pulp_v, pulp_f = nifti_to_mesh(pulp_path, labels=pulp_labels)
    else:
        pulp_v, pulp_f = np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)

    ios_v, ios_f = load_ios_mesh(ios_path)
    ios_v = apply_transform(ios_v, T)  # move IOS onto the CBCT

    return {
        "arch": arch_used,
        "fitness": best["fitness"],
        "rmse": best["rmse"],
        "strict_fitness": best["strict_fitness"],
        "mirrored": best.get("mirrored", False),
        "T_ios_to_cbct": T,
        "T_cbct_to_ios": best["T_cbct_to_ios"],
        "meshes": {
            "teeth": {"vertices": teeth_v, "faces": teeth_f},
            "pulp": {"vertices": pulp_v, "faces": pulp_f},
            "ios": {"vertices": ios_v, "faces": ios_f},
        },
    }


def downsample_for_display(points, max_points=8000):
    """Thin a point cloud for browser rendering."""
    pts = np.asarray(points)
    if pts.shape[0] <= max_points:
        return pts
    idx = np.random.choice(pts.shape[0], max_points, replace=False)
    return pts[idx]


# ----------------------------------------------------------------------------
# File-based entry point (no folder discovery): the caller supplies the CBCT
# segmentation and the IOS directly. Used by the web app and the CLI.
# ----------------------------------------------------------------------------
def _nifti_nonzero_labels(path):
    data = np.asanyarray(nib.load(path).dataobj)
    v = np.unique(data)
    return set(int(x) for x in v if x > 0)


def register_files(teeth_path, ios_path, arch="auto", pulp_path=None):
    """Register an IOS mesh onto a CBCT tooth segmentation, given as file paths.

    Parameters
    ----------
    teeth_path : CBCT tooth segmentation (NIfTI). Per-tooth FDI instance labels
                 enable upper/lower arch splitting; a binary mask is treated as
                 the whole mouth.
    ios_path   : intraoral scan surface mesh (.ply or .stl).
    pulp_path  : optional CBCT pulp segmentation (NIfTI), carried into the IOS
                 frame for visualisation.
    arch       : 'auto' | 'upper' | 'lower' | 'both'.

    Returns the transform, its inverse, quality metrics, whether a mirror was
    applied, and display meshes (teeth, pulp, registered IOS) in CBCT space.
    """
    ios = load_mesh_points(ios_path)
    labels = _nifti_nonzero_labels(teeth_path)
    is_instance = len(labels - {1}) > 0        # more than a single foreground label

    if arch == "auto":
        named = arch_from_filename(ios_path)
        if named and is_instance:
            candidates = [named]
        elif is_instance:
            candidates = ["upper", "lower"]
        else:
            candidates = ["both"]
    else:
        candidates = [arch]

    def labels_for(a):
        return arch_labels(a) if (is_instance and a in ("upper", "lower")) else None

    best = None
    for a in candidates:
        teeth = nifti_surface_points(teeth_path, labels=labels_for(a))
        if teeth.shape[0] < 100:
            continue
        res = register(ios, teeth)
        res["arch"] = a
        if best is None or res["strict_fitness"] > best["strict_fitness"]:
            best = res
    if best is None:
        raise RuntimeError("Registration failed: no usable teeth surface found.")

    arch_used = best["arch"]
    T = best["T_ios_to_cbct"]
    lab = labels_for(arch_used)

    teeth_v, teeth_f = nifti_to_mesh(teeth_path, labels=lab)
    if pulp_path:
        pulp_v, pulp_f = nifti_to_mesh(pulp_path, labels=lab)
        if pulp_f.shape[0] == 0:               # pulp labelled differently; keep all
            pulp_v, pulp_f = nifti_to_mesh(pulp_path, labels=None)
    else:
        pulp_v, pulp_f = np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)

    ios_v, ios_f = load_ios_mesh(ios_path)
    ios_v = apply_transform(ios_v, T)

    return {
        "arch": arch_used,
        "fitness": best["fitness"],
        "rmse": best["rmse"],
        "strict_fitness": best["strict_fitness"],
        "mirrored": best.get("mirrored", False),
        "T_ios_to_cbct": T,
        "T_cbct_to_ios": best["T_cbct_to_ios"],
        "meshes": {
            "teeth": {"vertices": teeth_v, "faces": teeth_f},
            "pulp": {"vertices": pulp_v, "faces": pulp_f},
            "ios": {"vertices": ios_v, "faces": ios_f},
        },
    }
