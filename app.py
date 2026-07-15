"""
Flask web app: provide a CBCT teeth + pulp segmentation and an intraoral scan,
register them, and view the fused result in 3D. Nothing is read from any fixed
data folder; every input is provided through the browser.

If you only have a raw CBCT, the "Segment with TIPs" button runs the TIPs
pipeline (see tips_segment.py) to produce the teeth and pulp segmentations first.

Run:  python app.py     then open http://127.0.0.1:5001
"""

import os
import json
import uuid
import shutil
import tempfile
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request, render_template

import registration as R
import tips_segment

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024   # 2 GB (raw CBCT can be large)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
JOBS_DIR = os.path.join(OUTPUT_DIR, "tips_jobs")   # cached TIPs segmentations
os.makedirs(JOBS_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


def _mesh_json(mesh):
    v = np.asarray(mesh["vertices"], dtype=np.float32).reshape(-1).tolist()
    f = np.asarray(mesh["faces"], dtype=np.int32).reshape(-1).tolist()
    return {"vertices": v, "faces": f}


def _save_upload(file_storage, tmpdir):
    """Save an uploaded file to tmpdir, preserving a safe name/extension."""
    name = secure_filename(file_storage.filename) or "upload"
    path = os.path.join(tmpdir, name)
    file_storage.save(path)
    return path


@app.route("/api/tips_status")
def api_tips_status():
    """Tell the UI whether the TIPs segmentation button can be used here."""
    ok, reason = tips_segment.tips_available()
    return jsonify({"available": ok, "reason": reason})


@app.route("/api/segment", methods=["POST"])
def api_segment():
    """Run TIPs on an uploaded raw CBCT and cache the teeth + pulp segmentations
    under a job id the registration step can reuse (so the user does not have to
    re-upload them)."""
    if "cbct" not in request.files or not request.files["cbct"].filename:
        return jsonify({"error": "Upload a raw CBCT volume (.nii / .nii.gz)."}), 400

    job = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job)
    os.makedirs(job_dir, exist_ok=True)
    cbct_path = _save_upload(request.files["cbct"], job_dir)
    try:
        seg = tips_segment.segment_cbct(cbct_path, os.path.join(job_dir, "tips"))
        # Keep the two segmentations next to the job for the register step.
        teeth_dst = os.path.join(job_dir, "cbct_teeth.nii.gz")
        pulp_dst = os.path.join(job_dir, "cbct_pulp.nii.gz")
        shutil.copy(seg["teeth"], teeth_dst)
        shutil.copy(seg["pulp"], pulp_dst)
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify({"error": str(e)}), 500

    return jsonify({"job": job,
                    "teeth_file": os.path.basename(teeth_dst),
                    "pulp_file": os.path.basename(pulp_dst)})


def _job_segmentations(job):
    """Return (teeth_path, pulp_path) for a completed TIPs job, or (None, None)."""
    job_dir = os.path.join(JOBS_DIR, secure_filename(job or ""))
    teeth = os.path.join(job_dir, "cbct_teeth.nii.gz")
    pulp = os.path.join(job_dir, "cbct_pulp.nii.gz")
    if os.path.isfile(teeth) and os.path.isfile(pulp):
        return teeth, pulp
    return None, None


@app.route("/api/register", methods=["POST"])
def api_register():
    # An IOS mesh is always required. The CBCT teeth + pulp segmentations come
    # either from two uploads, or from a prior TIPs job (form field "job").
    ios_file = request.files.get("ios")
    if not ios_file or not ios_file.filename:
        return jsonify({"error": "An intraoral scan (IOS) file is required."}), 400
    arch = request.form.get("arch", "auto")
    job = request.form.get("job")
    job_teeth, job_pulp = _job_segmentations(job) if job else (None, None)

    have_uploads = (request.files.get("teeth") and request.files["teeth"].filename
                    and request.files.get("pulp") and request.files["pulp"].filename)
    if not job_teeth and not have_uploads:
        return jsonify({"error": "Provide the CBCT teeth + pulp segmentations "
                                 "(upload both, or segment a CBCT with TIPs first)."}), 400

    with tempfile.TemporaryDirectory() as tmp:
        ios_path = _save_upload(ios_file, tmp)
        if job_teeth:                       # reuse the cached TIPs segmentations
            teeth_path, pulp_path = job_teeth, job_pulp
        else:                               # two direct uploads
            teeth_path = _save_upload(request.files["teeth"], tmp)
            pulp_path = _save_upload(request.files["pulp"], tmp)

        try:
            res = R.register_files(teeth_path, ios_path, arch=arch, pulp_path=pulp_path)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        # Persist the transform + summary under outputs/<ios name>/.
        case = os.path.splitext(os.path.splitext(os.path.basename(ios_path))[0])[0]
        case_dir = os.path.join(OUTPUT_DIR, secure_filename(case) or "case")
        os.makedirs(case_dir, exist_ok=True)
        np.save(os.path.join(case_dir, "T_ios_to_cbct.npy"), res["T_ios_to_cbct"])
        np.save(os.path.join(case_dir, "T_cbct_to_ios.npy"), res["T_cbct_to_ios"])
        with open(os.path.join(case_dir, "summary.json"), "w") as f:
            json.dump({
                "ios_file": os.path.basename(ios_path),
                "teeth_file": os.path.basename(teeth_path),
                "arch": res["arch"], "mirrored": res["mirrored"],
                "fitness": res["fitness"], "rmse_mm": res["rmse"],
                "strict_overlap": res["strict_fitness"],
                "T_ios_to_cbct": res["T_ios_to_cbct"].tolist(),
            }, f, indent=2)

        return jsonify({
            "arch": res["arch"],
            "mirrored": res["mirrored"],
            "fitness": round(res["fitness"], 3),
            "rmse_mm": round(res["rmse"], 3),
            "strict_overlap": round(res["strict_fitness"], 3),
            "transform": res["T_ios_to_cbct"].tolist(),
            "output_dir": case_dir,
            "meshes": {
                "teeth": _mesh_json(res["meshes"]["teeth"]),
                "pulp": _mesh_json(res["meshes"]["pulp"]),
                "ios": _mesh_json(res["meshes"]["ios"]),
            },
        })


if __name__ == "__main__":
    print("Open http://127.0.0.1:5001 in your browser.")
    app.run(host="127.0.0.1", port=5001, debug=False)
