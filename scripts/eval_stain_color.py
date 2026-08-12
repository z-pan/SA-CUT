#!/usr/bin/env python3
"""Per-region stain-colour fidelity of virtual H&E against a real H&E reference.

Virtual staining is usually judged by eye ("nuclei not purple enough", "cytoplasm
looks violet rather than pink"). This turns that into a number, separately for
nuclei and cytoplasm, so different pipelines/epochs can be compared and tuned.

The diagnostic axis is **R - B** per region:

* ``R - B > 0`` → pink/magenta (eosin-dominated)
* ``R - B < 0`` → blue/violet (haematoxylin-dominated)

In real H&E, cytoplasm sits near ``R - B ≈ 0`` (balanced magenta) while nuclei are
strongly negative. A virtual result whose *cytoplasm* is negative is the
"cytoplasm looks purple" complaint, quantified.

Regions are defined without needing registration between the virtual and real
sets (they are unpaired):

* virtual: nuclei = the pre-computed TPAF nuclear mask, matched by filename stem
* real:    nuclei = top ``--real-nuc-pct`` percentile of the haematoxylin channel
           (falls back to a luminance-based estimate when no H channel is given)

Usage
-----
    python scripts/eval_stain_color.py \\
        --virtual  results/vhe_patches --virtual-suffix _fake_B \\
        --nuc-mask data/patches/masks \\
        --real     data/raw/he --real-h data/raw/he_deconv_H

Compare several pipelines by running it once per ``--virtual`` directory with the
same ``--real`` reference; the "gap vs real" column is the thing to minimise.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def _rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def _gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def _index(directory: Path) -> dict[str, Path]:
    """Map filename stem → path for every image in *directory*."""
    return {p.stem: p for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix.lower() in _EXT}


def _summarise(nuc: list[np.ndarray], cyt: list[np.ndarray], label: str,
               green: list[float] | None = None) -> dict:
    out = {"set": label, "n": len(cyt)}
    for name, vals in (("nuc", nuc), ("cyt", cyt)):
        if vals:
            m = np.asarray(vals).mean(0)
            out[f"{name}_R"], out[f"{name}_G"], out[f"{name}_B"] = m
            out[f"{name}_RmB"] = m[0] - m[2]
        else:
            for k in ("R", "G", "B", "RmB"):
                out[f"{name}_{k}"] = float("nan")
    out["green_pct"] = float(np.mean(green)) if green else float("nan")
    return out


def _green_fraction(img: np.ndarray, tissue: np.ndarray) -> float:
    """Percentage of tissue pixels where green dominates both other channels.

    H&E has no green ink, so a real section scores ~0. A non-trivial value means
    the generator is inventing colours that cannot occur in the target domain —
    an artifact the per-region R-B statistic cannot see, because a green pixel
    has low R *and* low B and so leaves R-B near zero.
    """
    if tissue.sum() == 0:
        return float("nan")
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    return 100.0 * float(((G > R) & (G > B) & tissue).sum()) / float(tissue.sum())


def measure_virtual(args) -> dict:
    """Region colours of the virtual set, using the TPAF nuclear masks."""
    masks = _index(Path(args.nuc_mask))
    nuc, cyt, green = [], [], []
    missing = 0
    for p in sorted(Path(args.virtual).iterdir()):
        if not (p.is_file() and p.suffix.lower() in _EXT):
            continue
        stem = p.stem
        if args.virtual_suffix:
            stem = stem.replace(args.virtual_suffix, "")
        mpath = masks.get(stem)
        if mpath is None:
            missing += 1
            continue
        img = _rgb(p)
        m = _gray(mpath) > 127
        tissue = img.mean(2) < args.white_thresh
        n_sel, c_sel = m & tissue, tissue & ~m
        if n_sel.sum() < args.min_px or c_sel.sum() < args.min_px:
            continue
        nuc.append(img[n_sel].mean(0))
        cyt.append(img[c_sel].mean(0))
        green.append(_green_fraction(img, tissue))
    if missing:
        print(f"  [warn] {missing} virtual images had no matching mask "
              f"(check --virtual-suffix)", file=sys.stderr)
    return _summarise(nuc, cyt, "virtual", green)


def measure_real(args) -> dict:
    """Region colours of the real reference, using the H channel when available."""
    hmap = _index(Path(args.real_h)) if args.real_h else {}
    nuc, cyt, green = [], [], []
    for p in sorted(Path(args.real).iterdir()):
        if not (p.is_file() and p.suffix.lower() in _EXT):
            continue
        img = _rgb(p)
        tissue = img.mean(2) < args.white_thresh
        if tissue.sum() < args.min_px * 2:
            continue
        hpath = hmap.get(p.stem)
        # Without an H channel, approximate haematoxylin by "dark and blue".
        score = _gray(hpath) if hpath is not None else (img[..., 2] - img[..., 1])
        thr = np.percentile(score[tissue], args.real_nuc_pct)
        n_sel = (score > thr) & tissue
        c_sel = tissue & ~n_sel
        if n_sel.sum() < args.min_px or c_sel.sum() < args.min_px:
            continue
        nuc.append(img[n_sel].mean(0))
        cyt.append(img[c_sel].mean(0))
        green.append(_green_fraction(img, tissue))
    return _summarise(nuc, cyt, "real", green)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-region stain-colour fidelity of virtual H&E vs real H&E.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--virtual", required=True, help="Directory of virtual H&E images.")
    ap.add_argument("--nuc-mask", required=True,
                    help="Nuclear masks for the virtual set, matched by filename stem.")
    ap.add_argument("--real", required=True, help="Directory of real H&E reference images.")
    ap.add_argument("--real-h", default=None,
                    help="Haematoxylin-channel images for the real set (recommended).")
    ap.add_argument("--virtual-suffix", default="",
                    help="Substring stripped from virtual filenames to match masks "
                         "(e.g. '_fake_B').")
    ap.add_argument("--real-nuc-pct", type=float, default=85.0,
                    help="Percentile of the H channel above which a real pixel is nuclear.")
    ap.add_argument("--white-thresh", type=float, default=235.0,
                    help="Mean-RGB above which a pixel is treated as slide background.")
    ap.add_argument("--min-px", type=int, default=200,
                    help="Minimum pixels a region needs before an image is counted.")
    ap.add_argument("--out", default=None, help="Optional CSV path for the raw numbers.")
    args = ap.parse_args()

    real = measure_real(args)
    virt = measure_virtual(args)

    print(f"\nreal reference: {real['n']} images   |   virtual: {virt['n']} images\n")
    hdr = f"{'':<10} {'R':>7} {'G':>7} {'B':>7} {'R-B':>8}   interpretation"
    print(hdr)
    print("-" * len(hdr))
    for region, zh in (("cyt", "cytoplasm"), ("nuc", "nuclei")):
        print(f"[{zh}]")
        for row, lbl in ((real, "real"), (virt, "virtual")):
            rmb = row[f"{region}_RmB"]
            tone = "pink" if rmb > 5 else ("violet" if rmb < -5 else "balanced")
            print(f"  {lbl:<8} {row[f'{region}_R']:>7.1f} {row[f'{region}_G']:>7.1f} "
                  f"{row[f'{region}_B']:>7.1f} {rmb:>8.1f}   {tone}")
        gap = virt[f"{region}_RmB"] - real[f"{region}_RmB"]
        print(f"  {'gap':<8} {'':>7} {'':>7} {'':>7} {gap:>8.1f}   "
              f"{'<- minimise (negative = too violet)' if abs(gap) > 3 else '<- close'}\n")

    print(f"[impossible colours]  green-dominant tissue pixels "
          f"(H&E has no green ink, real ~0)")
    print(f"  {'real':<8} {real['green_pct']:>7.2f} %")
    flag = "  <- ARTIFACT" if virt["green_pct"] > 1.0 else "  <- clean"
    print(f"  {'virtual':<8} {virt['green_pct']:>7.2f} % {flag}\n")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(real.keys()))
            w.writeheader()
            w.writerow(real)
            w.writerow(virt)
        print(f"CSV written → {args.out}")


if __name__ == "__main__":
    main()
