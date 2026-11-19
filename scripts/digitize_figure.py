"""Recover data points from the published inactivation figure.

Used because the raw lab measurements are not available. Every constant below
was derived from the image itself (tick-mark detection), not guessed.

Validation that the calibration is right: the recovered x values land within
0.3 min of 15, 30, 45, 60, 120, 180, 240, 300, 400.

Usage:
    python scripts/digitize_figure.py <figure.png> > data/raw/inactivation_50C_digitized.csv
"""
import sys
import numpy as np
from PIL import Image

# Pixel calibration (from detected axis ticks in the source figure)
X0_PX, X_PER_30_PX = 310.0, 93.0     # t = 0 min at px 310; 30 min every 93 px
Y0_PX, Y120_PX = 903.0, 142.0        # A = 0 % at px 903; A = 120 % at px 142
PLOT_BOX = dict(top=137, bottom=909, left=305, right=1588)

SERIES_RGB = {
    "WT":     (122, 128, 255),
    "DOUBLE": (255, 148, 0),
    "D314A":  (148, 22, 81),
    "S126R":  (255, 212, 121),
}
TIMES = [0, 15, 30, 45, 60, 120, 180, 240, 300, 400]
COLOR_TOL = 90        # L1 distance in RGB
BAND_HALFWIDTH = 5    # px either side of the nominal time
MIN_MARKER_HEIGHT = 13  # markers are ~20 px tall; the fitted line is ~6 px


def px_of_t(t):
    return X0_PX + t * (X_PER_30_PX / 30.0)


def activity_of_py(py):
    return (Y0_PX - py) * 120.0 / (Y0_PX - Y120_PX)


def _runs(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def digitize(path):
    img = np.array(Image.open(path).convert("RGB")).astype(int)
    rows = []
    for name, rgb in SERIES_RGB.items():
        mask = np.abs(img - np.array(rgb)).sum(2) < COLOR_TOL
        mask[:PLOT_BOX["top"]] = False
        mask[PLOT_BOX["bottom"]:] = False
        mask[:, :PLOT_BOX["left"]] = False
        mask[:, PLOT_BOX["right"]:] = False
        for t in TIMES:
            xc = int(round(px_of_t(t)))
            band = mask[:, max(0, xc - BAND_HALFWIDTH): xc + BAND_HALFWIDTH + 1]
            present = band.sum(1) >= 7
            for s, e in _runs(present):
                if e - s + 1 >= MIN_MARKER_HEIGHT:
                    rows.append((name, t, round(activity_of_py((s + e) / 2.0), 1)))
    return rows


if __name__ == "__main__":
    print("variant,time_min,activity_pct")
    for name, t, a in sorted(digitize(sys.argv[1])):
        print(f"{name},{t},{a}")
