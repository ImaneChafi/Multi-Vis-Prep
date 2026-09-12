---
title: Multi-Vis-Prep CBCT-IOS Registration
emoji: 🦷
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Multi-Vis-Prep

Interactive CBCT to intraoral-scan registration. Upload a CBCT tooth
segmentation, a CBCT pulp segmentation, and an intraoral scan (`.ply`/`.stl`);
the app registers them with a fully automated surface-based pipeline and shows
the fused result in 3D, with per-case quality metrics and transparency controls.

The "Segment with TIPs" button needs a GPU and the TIPs model weights, so it is
disabled on this CPU Space; provide the two CBCT segmentations directly.

Source code: https://github.com/ImaneChafi/Multi-Vis-Prep
