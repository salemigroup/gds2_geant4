#!/usr/bin/env python3
"""
plot_entries.py — one plot per entry, for a given chip and layer.

Reads either an .npz written by chip_geometry.py or a GDS file directly, and
saves a separate image for every entry on the chosen layer. Each title carries
the chip, the layer and the entry name, and each file is named the same way, so
the plots sort into the order the entries were generated.

    python plot_entries.py chips/C1R1.npz --layer 2 --out plots/
    python plot_entries.py chips/all.npz --chip C1R1 --layer 2 --out plots/
    python plot_entries.py wafer.gds --chip C1R1 --layer 2 --out plots/

One plot per object. Each object is drawn with its inserts cut out of it and
its hole centres marked in red. Holes never get a plot of their own, whether
they are recorded as centre points or drawn as separate small polygons; see
--hole-max-size.

Requires: numpy, matplotlib, and chip_geometry.py alongside this file
(the latter only when reading a .gds or resolving a packed file).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------- loading
def load_chips(path, chip=None, layers=None):
    """Return ``{chip_id: {key: array}}`` from a .npz or a .gds."""
    path = Path(path)
    if path.suffix.lower() == ".gds":
        from chip_geometry import build_chip_geometry
        return build_chip_geometry(path, layers=layers)

    z = np.load(path, allow_pickle=True)
    names = list(z.files)
    if any(n.endswith("__xy") for n in names):
        from chip_geometry import load_packed
        return load_packed(path)
    if any("__" in n for n in names):
        from chip_geometry import load_single_npz
        return load_single_npz(path)
    # a single-chip file: take the id from the filename
    return {chip or path.stem: {k: z[k] for k in names}}


def entries_for_layer(chip_dict, layer):
    """Entries on one layer, in generation order, metadata excluded."""
    prefix = f"L{layer}_"
    keys = [k for k in chip_dict if k.startswith(prefix)]

    def order(k):
        rest = k[len(prefix):]
        head = rest.split("_")[0]
        return (0, int(head)) if head.isdigit() else (1, rest)

    return sorted(keys, key=order)


# ------------------------------------------------------------------ plotting
def plot_entry(chip_dict, chip_id, layer, key, out_dir, dpi=130, fmt="png",
               hole_marker=18):
    """Save one entry as its own figure. Returns the path written."""
    arr = np.asarray(chip_dict[key], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    ax.fill(arr[:, 0], arr[:, 1], facecolor="#8fb6e8",
            edgecolor="#22456f", lw=1.0)
    inserts = [k for k in chip_dict if k.startswith(key + "_insert")]
    for r in sorted(inserts, key=lambda k: int(k.rsplit("insert", 1)[1])):
        h = np.asarray(chip_dict[r], dtype=float)
        ax.fill(h[:, 0], h[:, 1], facecolor="white",
                edgecolor="#22456f", lw=0.8)
    centres = chip_dict.get(key + "_holes")
    subtitle = f"{len(arr) - 1} vertices"
    if inserts:
        subtitle += f", {len(inserts)} inserts"
    if centres is not None and len(centres):
        c = np.asarray(centres, dtype=float)
        ax.scatter(c[:, 0], c[:, 1], s=hole_marker, c="#d1372e", lw=0, zorder=3)
        subtitle += f", {len(centres)} holes"

    ax.set_aspect("equal")
    ax.set_title(f"{chip_id}  |  layer {layer}  |  {key}\n{subtitle}",
                 fontsize=10)
    ax.set_xlabel("x (um, relative to chip centre)", fontsize=9)
    ax.set_ylabel("y (um, relative to chip centre)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.3, color="0.9")
    fig.tight_layout()

    out = Path(out_dir) / f"{chip_id}_{key}.{fmt}"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def object_size(arr) -> float:
    """Largest bounding-box dimension of a closed perimeter, in um."""
    arr = np.asarray(arr, dtype=float)
    return max(arr[:, 0].max() - arr[:, 0].min(),
               arr[:, 1].max() - arr[:, 1].min())


def plot_chip_layer(chips, chip_id, layer, out_dir, limit=None,
                    hole_max_size=20.0, **kw):
    """Plot every object of one chip and layer. Returns the paths written.

    Objects no larger than ``hole_max_size`` um across are skipped: on a layer
    where perforations are drawn as separate polygons they would otherwise
    produce tens of thousands of near-identical little squares. Pass 0 to plot
    everything.
    """
    if chip_id not in chips:
        raise KeyError(f"no chip {chip_id!r}; available: {sorted(chips)}")
    chip_dict = chips[chip_id]
    keys = entries_for_layer(chip_dict, layer)
    if not keys:
        raise KeyError(f"chip {chip_id} has nothing on layer {layer}")

    # only outer perimeters get a plot of their own: inserts and hole centres
    # are drawn on top of the object they belong to
    keys = [k for k in keys if k.count("_") == 1 and not k.endswith("_holes")]

    if hole_max_size:
        kept = [k for k in keys if object_size(chip_dict[k]) > hole_max_size]
        skipped = len(keys) - len(kept)
        if skipped:
            print(f"skipping {skipped} objects <= {hole_max_size} um across "
                  f"(perforations)")
        keys = kept
        if not keys:
            raise KeyError(f"chip {chip_id} layer {layer}: every object is "
                           f"<= {hole_max_size} um across; lower "
                           f"--hole-max-size to plot them")
    if limit is not None:
        keys = keys[:limit]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [plot_entry(chip_dict, chip_id, layer, k, out_dir, **kw)
            for k in keys]


# ---------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Plot every entry of one chip and layer to its own file.")
    ap.add_argument("source", help=".npz from chip_geometry.py, or a .gds file")
    ap.add_argument("--chip", help="chip id, e.g. C1R1. Required unless the "
                                   "source is a single-chip .npz")
    ap.add_argument("--layer", type=int,
                    help="layer number; required unless --list is given")
    ap.add_argument("--out", default="plots", help="output directory "
                                                   "(default: plots/)")
    ap.add_argument("--limit", type=int,
                    help="plot only the first N objects, useful on dense layers")
    ap.add_argument("--hole-max-size", type=float, default=20.0,
                    help="skip objects this size or smaller, in um across, so "
                         "perforations drawn as separate polygons do not each "
                         "get a plot. 0 plots everything (default: 20)")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--format", default="png",
                    help="png, pdf, svg ... (default: png)")
    ap.add_argument("--list", action="store_true",
                    help="list the chips and layers available, then exit")
    args = ap.parse_args(argv)

    src = Path(args.source)
    if not src.exists():
        sys.exit(f"no such file: {src}")

    chips = load_chips(src, chip=args.chip,
                       layers=None if args.list else [args.layer])
    if args.list:
        for chip_id in sorted(chips):
            layers = sorted({int(k.split("_")[0][1:]) for k in chips[chip_id]
                             if not k.startswith("_")})
            print(f"{chip_id}: layers {layers}")
        return

    if args.layer is None:
        sys.exit("--layer is required (or use --list to see what is available)")

    chip_id = args.chip
    if chip_id is None:
        if len(chips) != 1:
            sys.exit(f"--chip is required; this file holds {sorted(chips)}")
        chip_id = next(iter(chips))

    try:
        paths = plot_chip_layer(chips, chip_id, args.layer, args.out,
                                limit=args.limit, dpi=args.dpi, fmt=args.format,
                                hole_max_size=args.hole_max_size)
    except KeyError as exc:
        sys.exit(str(exc).strip('"\''))
    print(f"wrote {len(paths)} plots -> {Path(args.out)}/")
    for p in paths[:3]:
        print(f"   {p.name}")
    if len(paths) > 3:
        print(f"   ... and {len(paths) - 3} more")


if __name__ == "__main__":
    main()
