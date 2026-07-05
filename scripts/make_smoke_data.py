"""Generate tiny synthetic patches for the ``--fast_dev_run`` smoke test.

The smoke test only exercises the pipeline wiring (data → model → losses →
optimizer step → checkpoint), so it does not need real data. This writes
correctly-shaped ``float32`` ``.npy`` patches to ``data/smoke/{tpaf,he,masks}``
with matching TPAF/mask filename stems, so it runs with zero Drive access.

Usage::

    python scripts/make_smoke_data.py            # 4 patches, 128x128
    python scripts/make_smoke_data.py --n 6 --size 256
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic smoke-test patches.")
    ap.add_argument("--out", default="data/smoke", help="Output root directory.")
    ap.add_argument("--n", type=int, default=4, help="Number of patches per domain.")
    ap.add_argument("--size", type=int, default=128, help="Patch height/width in pixels.")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    root = Path(args.out)
    for sub in ("tpaf", "he", "masks"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    S = args.size
    for i in range(args.n):
        stem = f"smoke_{i:03d}"                       # TPAF & mask share the stem
        np.save(root / "tpaf" / f"{stem}.npy", rng.random((S, S)).astype(np.float32))
        np.save(root / "masks" / f"{stem}.npy", (rng.random((S, S)) > 0.5).astype(np.float32))
        # H&E is unpaired — a different stem is fine.
        np.save(root / "he" / f"he_{i:03d}.npy", rng.random((S, S, 3)).astype(np.float32))

    print(f"Wrote {args.n} synthetic patches ({S}x{S}) to {root}/ (tpaf, he, masks)")


if __name__ == "__main__":
    main()
