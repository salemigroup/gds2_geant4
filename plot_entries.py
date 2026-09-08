#!/usr/bin/env python3
"""
plot_entries.py — one plot per entry, for a given chip and layer.

Reads an .npz written by chip_geometry.py and saves a separate image for every
entry on the chosen layer. Each title carries
the chip, the layer and the entry name, and each file is named the same way, so
the plots sort into the order the entries were generated.

    python plot_entries.py chips/C1R1.npz --layers 2 --out plots/
    python plot_entries.py chips/all.npz --chip C1R1 --layers 2,3 --out plots/
    python plot_entries.py chips/C1R1.npz --overview --out plots/

One plot per object by default. Each object is drawn with its inserts cut out
of it and its hole centres marked in red. Holes never get a plot of their own,
whether they are recorded as centre points or drawn as separate small polygons;
see --hole-max-size.

With --overview you instead get a single figure of the whole chip with every
layer drawn together, colour-coded, which is the quickest way to see what a
chip actually contains.

--layers takes the same comma-separated form as chip_geometry.py, so
"--layers 2,3" plots both layers in one run.

Requires: numpy, matplotlib, and chip_geometry.py alongside this file
(the latter only for its .npz loaders).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import numpy as np


# ------------------------------------------------------------------- loading
def load_chips(path, chip=None, layers=None):
    """Return ``{chip_id: {key: array}}`` from an .npz written by chip_geometry.

    GDS is not read here. Doing so would use that script's defaults, which for
    a layer needing --invert-layers or --hole-layers draws something that is
    not the layout.
    """
    path = Path(path)
    if path.suffix.lower() == ".gds":
        sys.exit(f"'{path.name}' is a GDS file. Only chip_geometry.py reads "
                 f"GDS; run it first and plot its .npz:\n"
                 f"    python chip_geometry.py {path.name} --layers 2 "
                 f"--out chips/")

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

    inserts = [k for k in chip_dict if k.startswith(key + "_insert")]
    rings = [chip_dict[r] for r in
             sorted(inserts, key=lambda k: int(k.rsplit("insert", 1)[1]))]
    ax.add_patch(polygon_patch(arr, rings, facecolor="#8fb6e8",
                               edgecolor="#22456f", lw=1.0))
    ax.autoscale_view()
    centres = chip_dict.get(key + "_holes")
    subtitle = f"{len(arr) - 1} vertices"
    if inserts:
        subtitle += f", {len(inserts)} inserts"
    if centres is not None and len(centres):
        c = np.asarray(centres, dtype=float)
        ax.scatter(c[:, 0], c[:, 1], s=hole_marker, c="#d1372e", lw=0, zorder=3)
        subtitle += f", {len(centres)} holes"

    # A very long thin object -- a waveguide centre conductor is 200 x 8550 um
    # -- squeezes one axis to a few pixels under equal aspect, and its labels
    # pile up. Pad the short axis so the view is at most 4:1. Equal aspect is
    # kept, so the shape is still true; there is just room around it.
    x0, x1 = arr[:, 0].min(), arr[:, 0].max()
    y0, y1 = arr[:, 1].min(), arr[:, 1].max()
    w, h = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    span = max(w, h) / 3.0
    if w < span:
        pad = (span - w) / 2
        x0, x1 = x0 - pad, x1 + pad
    if h < span:
        pad = (span - h) / 2
        y0, y1 = y0 - pad, y1 + pad
    m = 0.05 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - m, x1 + m)
    ax.set_ylim(y0 - m, y1 + m)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
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


def _ccw(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > 0


def polygon_patch(outer, holes=(), **kw):
    """A filled patch whose holes are genuinely transparent.

    Painting a hole white would hide anything already drawn underneath, which
    on a ground plane means the isolated conductors sitting inside its
    clearances disappear. A compound path leaves them showing through.
    """
    verts, codes = [], []
    for i, ring in enumerate([np.asarray(outer, dtype=float)] +
                             [np.asarray(h, dtype=float) for h in holes]):
        r = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
        # exterior anticlockwise, holes clockwise, so the winding rule cuts
        if _ccw(r) != (i == 0):
            r = r[::-1]
        verts.extend(r)
        verts.append(r[0])
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(r) - 1)
                     + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(verts, codes), **kw)


def object_size(arr) -> float:
    """Largest bounding-box dimension of a closed perimeter, in um."""
    arr = np.asarray(arr, dtype=float)
    return max(arr[:, 0].max() - arr[:, 0].min(),
               arr[:, 1].max() - arr[:, 1].min())


def plot_chip_layer(chips, chip_id, layer, out_dir, limit=None,
                    hole_max_size=10.0, **kw):
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


LAYER_COLOURS = ["#3b7dd8", "#d1372e", "#f0a202", "#2a9d5c", "#8e44ad",
                 "#16a085", "#c0392b", "#7f8c8d"]


def plot_overview(chip_dict, chip_id, out_dir, layers=None, dpi=150, fmt="png",
                  max_polygons=2000, max_markers=4000):
    """One figure of the whole chip, every layer drawn together.

    Every object is filled, with its inserts cut out. The cut-off here is a
    count, not a size: a layer with more objects than ``max_polygons`` keeps
    its largest that many and draws the rest as points, which is what makes a
    perforated ground plane drawable at all. Sizing the cut-off instead would
    turn genuinely small features, such as junction leads, into dots.
    """
    present = sorted({int(k.split("_")[0][1:]) for k in chip_dict
                      if not k.startswith("_")})
    layers = [l for l in (layers or present) if l in present]
    if not layers:
        raise KeyError(f"chip {chip_id} holds nothing on layers {layers}")

    fig, ax = plt.subplots(figsize=(9, 9))
    handles = []
    for i, layer in enumerate(layers):
        colour = LAYER_COLOURS[i % len(LAYER_COLOURS)]
        centres = []
        objects = []
        for key in entries_for_layer(chip_dict, layer):
            if key.endswith("_holes"):
                centres.extend(np.asarray(chip_dict[key], dtype=float))
            elif key.count("_") == 1:          # an outer perimeter
                objects.append((key, np.asarray(chip_dict[key], dtype=float)))

        # too many to draw individually: keep the largest, dot the rest
        small = []
        if len(objects) > max_polygons:
            objects.sort(key=lambda kv: -object_size(kv[1]))
            small = [a.mean(axis=0) for _, a in objects[max_polygons:]]
            objects = objects[:max_polygons]

        for key, arr in objects:
            rings = [chip_dict[k] for k in sorted(chip_dict)
                     if k.startswith(key + "_insert")]
            ax.add_patch(polygon_patch(arr, rings, facecolor=colour,
                                       edgecolor=colour, lw=0.3, alpha=0.75,
                                       zorder=2 + i))
        big = len(objects)

        pts = np.array(small + centres) if (small or centres) else None
        if pts is not None and len(pts):
            step = max(1, len(pts) // max_markers)
            ax.scatter(pts[::step, 0], pts[::step, 1], s=1.0, c=colour, lw=0,
                       alpha=0.6, zorder=2 + i)
        label = f"layer {layer}: {big} objects"
        if len(small):
            label += f" + {len(small)} too small to draw"
        if centres:
            label += f", {len(centres)} holes"
        if pts is not None and len(pts) and step > 1:
            label += f" (every {step}th marker)"
        handles.append(plt.Line2D([], [], color=colour, lw=6, label=label))

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x (um, relative to chip centre)")
    ax.set_ylabel("y (um, relative to chip centre)")
    ax.set_title(f"{chip_id}  |  layers {', '.join(str(l) for l in layers)}",
                 fontsize=12)
    ax.legend(handles=handles, fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, lw=0.3, color="0.92")
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{chip_id}_overview.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Plot every entry of one chip and layer to its own file.")
    ap.add_argument("source", help=".npz written by chip_geometry.py")
    ap.add_argument("--chip", help="chip id, e.g. C1R1. Required unless the "
                                   "source is a single-chip .npz")
    ap.add_argument("--layers",
                    help="comma-separated layer numbers, e.g. '2' or '2,3'. "
                         "Required unless --list or --overview is given")
    ap.add_argument("--out", default="plots", help="output directory "
                                                   "(default: plots/)")
    ap.add_argument("--limit", type=int,
                    help="plot only the first N objects, useful on dense layers")
    ap.add_argument("--hole-max-size", type=float, default=10.0,
                    help="skip objects this size or smaller, in um across, so "
                         "perforations drawn as separate polygons do not each "
                         "get a plot. 0 plots everything (default: 10)")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--format", default="png",
                    help="png, pdf, svg ... (default: png)")
    ap.add_argument("--overview", action="store_true",
                    help="draw one figure of the whole chip with every layer "
                         "together, instead of one image per object. Use "
                         "--layers to restrict it to some of them")
    ap.add_argument("--list", action="store_true",
                    help="list the chips and layers available, then exit")
    args = ap.parse_args(argv)

    src = Path(args.source)
    if not src.exists():
        sys.exit(f"no such file: {src}")

    want = ([int(v) for v in args.layers.split(",")] if args.layers else None)
    chips = load_chips(src, chip=args.chip,
                       layers=None if (args.list or args.overview) else want)
    if args.list:
        for chip_id in sorted(chips):
            layers = sorted({int(k.split("_")[0][1:]) for k in chips[chip_id]
                             if not k.startswith("_")})
            print(f"{chip_id}: layers {layers}")
        return

    if want is None and not args.overview:
        sys.exit("--layers is required (or use --list to see what is available, "
                 "or --overview for the whole chip)")

    chip_id = args.chip
    if chip_id is None:
        if len(chips) != 1:
            sys.exit(f"--chip is required; this file holds {sorted(chips)}")
        chip_id = next(iter(chips))

    if args.overview:
        try:
            p = plot_overview(chips[chip_id], chip_id, args.out,
                              layers=want, dpi=args.dpi, fmt=args.format)
        except KeyError as exc:
            sys.exit(str(exc).strip('"\''))
        print(f"wrote {p}")
        return

    paths = []
    for layer in want:
        try:
            paths += plot_chip_layer(chips, chip_id, layer, args.out,
                                     limit=args.limit, dpi=args.dpi,
                                     fmt=args.format,
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
