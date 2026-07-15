"""
Stack two figures into one 'contributions' pipeline:
  (a) overview  (top)    -> assets/overview.png   [your concept / segmentation
                                                    + registration render]
  (b) pipeline  (bottom) -> assets/pipeline.png    [the method block diagram]

Usage:
  python combined_pipeline.py                 # uses assets/overview.png on top
  python combined_pipeline.py path/to/top.png # override the top image
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
top_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ASSETS, "overview.png")
bot_path = os.path.join(ASSETS, "pipeline.png")

top = mpimg.imread(top_path)
bot = mpimg.imread(bot_path)

def ar(im):                      # aspect ratio (height / width)
    return im.shape[0] / im.shape[1]

W = 11.0                          # figure width (inches); both panels use full width
title_h = 0.45                    # inches reserved for each panel label
h_top = W * ar(top) + title_h
h_bot = W * ar(bot) + title_h

fig = plt.figure(figsize=(W, h_top + h_bot), dpi=200)
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(2, 1, height_ratios=[h_top, h_bot], hspace=0.04)

ax1 = fig.add_subplot(gs[0]); ax1.imshow(top); ax1.axis("off")
ax2 = fig.add_subplot(gs[1]); ax2.imshow(bot); ax2.axis("off")

ax1.set_title("(a)  CBCT and IOS segmentation and registration",
              loc="left", fontsize=13, fontweight="bold", color="#1a2230", pad=8)
ax2.set_title("(b)  Automated registration pipeline",
              loc="left", fontsize=13, fontweight="bold", color="#1a2230", pad=8)

for ext in ("png", "pdf"):
    out = os.path.join(ASSETS, f"contributions.{ext}")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("Saved", out)
