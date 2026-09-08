#!/usr/bin/env python3
"""
chip_geometry.py — per-chip dictionaries of perimeter coordinates as numpy arrays.

Builds ``{chip_id: {key: ndarray}}`` from a GDS file. Every value is an (N, 2)
float array of closed perimeter coordinates in micrometres, expressed relative
to the centre of that chip. Keys carry the layer number:

    "L1_00000"          outer perimeter of the largest object on layer 1
    "L1_00000_insert1"  closed perimeter of something cut into it, largest first
    "L1_00000_holes"    (M, 2) centre coordinates of that object's holes
    "L1_holes"          centres of holes on layer 1 that belong to no object
    "_chip_center_um"  that chip's centre in absolute layout coordinates
    "_layers"         which layers the chip holds

Objects are numbered largest first, and each carries the closed perimeter of
every insert cut into it, since on a ground plane the outer boundary is only a
rectangle and all the structure is in the inserts.

Fabrication holes are reported as centre points rather than drawn geometry, on
the assumption that the consuming application places them itself. Their size is
found from the file, not supplied, so a design with slightly different holes
needs no change here. Anything too large to be a hole stays as an explicit
interior ring; see ``hole_max_size``.

Overlapping and abutting source polygons are merged into single objects. The
chip lattice is inferred from the outlines in the file, and objects that
cross a chip boundary (a ground plane, a wafer outline) are clipped to the chip,
so each chip's dictionary is self-contained.

    from chip_geometry import build_chip_geometry, inserts_of, hole_centers

    chips = build_chip_geometry("wafer.gds", layers=[2, 3])
    outer = chips["C1R1"]["L2_00000"]               # outer perimeter, largest first
    cuts = inserts_of(chips["C1R1"], "L2_00000")    # perimeters cut into it
    ctr = hole_centers(chips["C1R1"], "L2_00000")   # its hole centres
    save_npz(chips, "chips/")

    python chip_geometry.py wafer.gds --layers 2,3 --out chips/

A dict of string keys to numpy arrays is exactly what an ``.npz`` file is, so
``save_npz`` round-trips with no schema of your own. For layers with tens of
thousands of small objects use ``save_packed`` instead — see its docstring.

Requires: gdstk, shapely, numpy.
    pip install gdstk shapely numpy
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import gdstk
    from shapely.geometry import MultiPolygon, Point, Polygon, box
    from shapely.ops import unary_union
except ImportError as exc:  # pragma: no cover
    sys.exit(f"missing dependency: {exc}. Install with: pip install gdstk shapely")


META_CENTER = "_chip_center_um"
META_LAYERS = "_layers"

# Redundant collinear vertices are always dropped. The tolerance sits below the
# usual 1 nm database grid, so nothing moves on-grid and the removal is lossless.
COLLINEAR_TOL = 1e-4

# Thresholds for reporting a layer as a likely perforation array. These are
# deliberately only a diagnosis: a dense layer of small real features looks
# identical by these measures, so acting on it is opt-in via auto_holes.
AUTO_HOLE_MIN_COUNT = 100
AUTO_HOLE_MIN_FRACTION = 0.9


# --------------------------------------------------------------- reading a gds
def um_per_user_unit(lib) -> float:
    """Micrometres per GDS user unit.

    A GDS header carries two numbers: the user unit in metres and the database
    precision in metres. gdstk hands back coordinates already divided by the
    user unit, so a file written with unit=1e-6 gives micrometres and one
    written with unit=1e-3 gives millimetres. Everything downstream works in
    micrometres, so coordinates are multiplied by this factor on the way in.
    """
    return lib.unit / 1e-6


def load_polygons(path: Path, layer: int, datatype: int | None):
    """Return (polygons on the layer, um-per-user-unit, gds precision)."""
    lib = gdstk.read_gds(str(path))
    cells = lib.top_level() or lib.cells
    polys = []
    for cell in cells:
        # get_polygons flattens any cell references into absolute coordinates
        for p in cell.get_polygons():
            if p.layer != layer:
                continue
            if datatype is not None and p.datatype != datatype:
                continue
            polys.append(p)
    return polys, um_per_user_unit(lib), lib.precision


def merge(polys, simplify: float, scale: float = 1.0):
    """Union the source polygons into shapes with proper interior rings."""
    geoms = []
    for p in polys:
        g = Polygon(p.points * scale if scale != 1.0 else p.points)
        if not g.is_valid:
            g = g.buffer(0)
        geoms.append(g)
    u = unary_union(geoms)
    geoms = list(u.geoms) if u.geom_type == "MultiPolygon" else [u]
    if simplify > 0:
        geoms = [g.simplify(simplify, preserve_topology=True) for g in geoms]
    return [g for g in geoms if not g.is_empty and g.area > 0]


# ------------------------------------------------------------------ chip layout
def outline_rects(path: Path, chip_layer: int, min_size: float):
    """Chip-outline rectangles on the reference layer, as (cx, cy, w, h) in um."""
    lib = gdstk.read_gds(str(path))
    scale = um_per_user_unit(lib)
    out = []
    for cell in (lib.top_level() or lib.cells):
        for p in cell.get_polygons():
            if p.layer != chip_layer or len(p.points) != 4:
                continue
            (x0, y0), (x1, y1) = p.bounding_box()
            x0, y0, x1, y1 = (v * scale for v in (x0, y0, x1, y1))
            w, h = x1 - x0, y1 - y0
            if min(w, h) < min_size:
                continue
            out.append(((x0 + x1) / 2, (y0 + y1) / 2, w, h))
    return out


def chip_outlines(path: Path, chip_layer: int, min_size: float, tol: float = 0.5,
                  verbose: bool = False):
    """Outline rectangles, with anything that is not a chip outline discarded.

    The reference layer often carries more than chip outlines: process-control
    squares, frames, test structures. Those are the wrong size, and because the
    lattice is inferred from rectangle sizes and spacings, a single stray one
    silently shifts both the pitch and the chip window.

    Chip outlines are the repeated, consistent ones, so the size family with the
    most members sets the scale and anything outside half to twice that size is
    dropped. A layout may legitimately draw two families -- the full cell and
    the same cell inset by the dicing street -- and both survive this test.
    """
    rects = outline_rects(path, chip_layer, min_size)
    if not rects:
        return []
    families = defaultdict(list)
    for r in rects:
        families[(round(r[2] / tol) * tol, round(r[3] / tol) * tol)].append(r)
    ref = max(families, key=lambda k: (len(families[k]), k[0] * k[1]))
    scale = max(ref)
    keep, drop = [], []
    for key, group in families.items():
        (keep if 0.5 * scale <= max(key) <= 2.0 * scale else drop).extend(group)
    if drop and verbose:
        sizes = sorted({(round(r[2]), round(r[3])) for r in drop})
        print(f"note: ignoring {len(drop)} rectangle(s) on layer {chip_layer} "
              f"too far from the {ref[0]:.0f} x {ref[1]:.0f} um chip outline "
              f"to be one: {sizes[:3]}")
    return keep


def detect_chip_size(path: Path, chip_layer: int, min_size: float):
    """The chip proper, as opposed to the lattice cell, in um.

    A layout often carries two kinds of outline: one drawn on the full cell and
    one inset by the dicing street. The smallest outline on each axis is the
    chip itself, so that is what gets returned. When only one size is drawn the
    chip and the cell are the same and no street can be inferred.
    """
    rects = chip_outlines(path, chip_layer, min_size)
    if not rects:
        return None
    return min(r[2] for r in rects), min(r[3] for r in rects)


def detect_chip_grid(path: Path, chip_layer: int, min_size: float, tol: float):
    """Infer the chip lattice from square outlines on a reference layer.

    Returns ``(x0, y0, pitch_x, pitch_y)`` where (x0, y0) is the lower-left
    corner of the lattice, or None if nothing usable is found. The two pitches
    are independent, so rectangular chips work as well as square ones.
    Rectangle *centres* are used
    rather than corners, so an outline drawn as the full cell and one drawn
    inset by the dicing street land on the same lattice point.

    The pitch is measured from the spacing between outlines. A file holding a
    single chip has no spacing to measure, so the chip's own side is used and
    the lattice is one cell wide; the dicing street is then not accounted for,
    which only matters if you expected the cell to be wider than the chip.
    """
    rects = chip_outlines(path, chip_layer, min_size, tol, verbose=True)
    if not rects:
        return None
    centres = [(cx, cy) for cx, cy, _, _ in rects]
    sides = [(w, h) for _, _, w, h in rects]

    def lattice(vals):
        uniq = sorted({round(v / tol) * tol for v in vals})
        if len(uniq) < 2:
            return uniq[0], None
        diffs = [b - a for a, b in zip(uniq, uniq[1:])]
        return uniq[0], min(diffs)

    cx0, px = lattice([c[0] for c in centres])
    cy0, py = lattice([c[1] for c in centres])
    # a single row, column or chip leaves one axis with no spacing to measure;
    # fall back to the chip's own size on that axis
    if px is None:
        px = float(np.median([w for w, _ in sides]))
    if py is None:
        py = float(np.median([h for _, h in sides]))
    return cx0 - px / 2, cy0 - py / 2, px, py


def normalise_grid(grid):
    """Accept ``(x0, y0, pitch)`` or ``(x0, y0, pitch_x, pitch_y)``."""
    if len(grid) == 3:
        x0, y0, p = grid
        return x0, y0, float(p), float(p)
    x0, y0, px, py = grid
    return x0, y0, float(px), float(py)


def outlined_cells(gds, chip_layer, grid, min_size=1000.0, tol=0.5):
    """Cell indices covered by chip-outline rectangles, as a rectangular hull.

    The outlines mark where chips are; the hull fills in any cell inside that
    block whose outline was never drawn. Without this, a layer that runs past
    the device area (a wafer outline, an edge-bead ring) would invent extra
    chips out in empty space.
    """
    x0, y0, px, py = normalise_grid(grid)
    idx = set()
    for cx, cy, _, _ in chip_outlines(Path(gds), chip_layer, min_size):
        idx.add((int(np.floor((cx - x0) / px)),
                 int(np.floor((cy - y0) / py))))
    if not idx:
        return None
    cs = [c for c, _ in idx]
    rs = [r for _, r in idx]
    return {(c, r) for c in range(min(cs), max(cs) + 1)
            for r in range(min(rs), max(rs) + 1)}


# ----------------------------------------------------------------- perimeters
def strip_collinear(ring, tol):
    """Drop vertices lying on the straight line between their neighbours.

    Works cyclically, so a redundant vertex sitting at the ring's start/end
    seam is removed too (Douglas-Peucker always pins the endpoints and cannot).
    A straight edge is left as just its two endpoints. Returns a closed ring.
    """
    pts = list(ring[:-1])  # drop the duplicated closing vertex
    n = len(pts)
    if n < 4:
        return list(ring)
    keep = []
    for i in range(n):
        (ax, ay), (bx, by), (cx, cy) = pts[i - 1], pts[i], pts[(i + 1) % n]
        span = math.hypot(cx - ax, cy - ay)
        if span == 0.0:
            dist = math.hypot(bx - ax, by - ay)
        else:
            # perpendicular distance from b to the line through a and c
            dist = abs((cx - ax) * (ay - by) - (ax - bx) * (cy - ay)) / span
        if dist > tol:
            keep.append(pts[i])
    if len(keep) < 3:
        return list(ring)
    return keep + [keep[0]]


def rings_to_array(geom):
    """Closed rings of a polygon: outer CCW first, then holes CW."""
    out = []
    ext = list(geom.exterior.coords)
    if not geom.exterior.is_ccw:
        ext = ext[::-1]
    out.append(ext)
    for interior in geom.interiors:
        pts = list(interior.coords)
        if interior.is_ccw:
            pts = pts[::-1]
        out.append(pts)
    out = [strip_collinear(r, COLLINEAR_TOL) for r in out]
    return [np.asarray(r, dtype=float) for r in out]


def explode(geom):
    """Flatten a clip result into a list of simple polygons."""
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def layer_shapes(path, layer, datatype, simplify):
    """Merged shapely polygons for one layer."""
    polys, scale, precision = load_polygons(Path(path), layer, datatype)
    if not polys:
        return [], scale, precision
    return merge(polys, simplify, scale), scale, precision


def order_key(arr):
    """Sort key putting the largest perimeter first, ties broken by position.

    Area alone is not enough. A design full of repeated identical objects
    produces areas that agree to about eleven significant figures but differ in
    the last bits, and which way a pair falls then depends on the summation
    order inside numpy, so two machines can number the same layout differently.
    The area is rounded to eight significant figures before comparison, which is
    far finer than any real distinction between two objects, and the centroid
    breaks what is left. Index 00000 then means the same object everywhere.
    """
    c = ring_centroid(arr)
    area = float(f"{ring_area(arr):.8g}")
    return (-area, round(float(c[0]), 6), round(float(c[1]), 6))


def hole_size(arr) -> float:
    """Largest bounding-box dimension of a closed ring, in um.

    Used to decide whether a void is small enough to be a fabrication hole.
    Taking the larger of width and height means a slightly rectangular hole is
    judged on its long side, so the size cap is never accidentally generous.
    """
    return max(arr[:, 0].max() - arr[:, 0].min(),
               arr[:, 1].max() - arr[:, 1].min())


def ring_area(arr) -> float:
    """Unsigned area of a closed ring given as an (N, 2) array."""
    x, y = arr[:-1, 0], arr[:-1, 1]
    return abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def ring_centroid(arr) -> np.ndarray:
    """Area centroid of a closed ring. Falls back to the vertex mean if the
    ring is degenerate (zero area)."""
    x, y = arr[:-1, 0], arr[:-1, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = 0.5 * cross.sum()
    if a == 0:
        return np.array([x.mean(), y.mean()])
    return np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6 * a)


# ------------------------------------------------------------------- main call
def build_chip_geometry(gds, layers=None, datatype=0, grid=None, chip_layer=0,
                       chip_size=None, chip_min_size=1000.0,
                       simplify=0.0, min_area=0.0, all_cells=False,
                       holes="centers", hole_max_size=10.0, hole_layers=None,
                       invert_layers=None, auto_holes=False, verbose=True):
    """Return ``{chip_id: {key: (N, 2) ndarray}}``, coordinates chip-relative.

    Parameters
    ----------
    layers      : layer numbers to include; None means every layer in the file.
    datatype    : datatype filter, or None for any.
    grid        : ``(x0, y0, pitch)`` or ``(x0, y0, pitch_x, pitch_y)`` in um,
                  overriding detection. Use
                  this when the chips are not marked by square outlines, or
                  when the outlines are there but you want a different cell.
    chip_min_size : ignore outline rectangles smaller than this in um when
                  detecting the lattice (default 1000, i.e. 1 mm). Lower it for
                  chips under a millimetre.
    chip_size   : side of the window kept per chip, in um; a pair for a
                  rectangular window. Defaults to the chip itself as drawn on
                  ``chip_layer``, which excludes the dicing street. Pass the
                  full
                  pitch to keep the street as well.
    simplify    : Douglas-Peucker tolerance in um (0 keeps every vertex).
    min_area    : skip objects below this area in um^2. Either a single number
                  for every layer, or ``{layer: value}`` to target one layer,
                  e.g. ``{1: 30}`` to drop 5x5 um fill without touching the
                  junction-scale features on other layers.
    all_cells   : keep every lattice cell any geometry touches, instead of only
                  the block of cells the chip outlines mark out.
    holes       : how to represent fabrication holes. "centers" (default)
                  records one centre point per hole and leaves every outline
                  hole-free; "rings" keeps them as explicit interior contours;
                  "drop" discards them entirely.
    hole_max_size : the largest a hole may be, in um across (default 10). This
                  is only a cap that separates holes from real geometry, not the
                  hole size itself -- whatever size the file actually uses is
                  found and reported. Enclosed voids larger than this stay as
                  interior rings. The cap matters because "enclosed void" alone
                  does not identify a hole: layer 1 of the sample wafer has 162
                  voids that are 165 x 323 um device clearances, an order of
                  magnitude clear of the 5 um holes on either side of the cap.
    invert_layers : layers drawn with negative tone, where what is drawn is what
                  gets etched away. For each such layer the metal is rebuilt as
                  the chip minus the drawn clearances, giving a few large solid
                  objects instead of thousands of fragments. Drawn shapes small
                  enough to be perforations are held out of that subtraction and
                  recorded as hole centres instead, so the plane comes out
                  solid. GDS records nothing about polarity, so this has to be
                  stated rather than detected.
    auto_holes  : act on the perforation-array diagnosis below instead of only
                  reporting it. Off by default, because "many identical small
                  polygons" cannot be told apart from a dense layer of small
                  real features by geometry alone, and acting on it wrongly
                  deletes those features silently. The diagnosis is always
                  printed, so the usual answer is to read it and then pass
                  ``hole_layers`` for the layer it names.
    hole_layers : layers on which small *standalone* objects should also be
                  treated as holes, e.g. ``[1]`` for a fill array drawn as
                  separate positive squares rather than as voids. Their centres
                  land in ``L1_holes`` and no outline is written for them. This
                  is opt-in per layer because, unlike an enclosed void, a small
                  free-standing polygon is usually a real feature.
    """
    gds = Path(gds)
    if grid is None:
        grid = detect_chip_grid(gds, chip_layer, chip_min_size, 0.5)
        if grid is None:
            probe, _, _ = load_polygons(Path(gds), chip_layer, None)
            if not probe:
                raise ValueError(
                    f"layer {chip_layer} holds no polygons, so there is nothing "
                    f"to infer a chip lattice from. Point --chip-layer at the "
                    f"layer carrying the chip outlines, or pass grid=(x0, y0, "
                    f"pitch) explicitly.")
        if grid is None:
            raise ValueError(
                f"layer {chip_layer} has polygons but none usable as a chip "
                f"outline: looking for rectangles at least {chip_min_size} um "
                f"across. Lower chip_min_size, or pass grid=(x0, y0, pitch) "
                f"explicitly.")
    x0, y0, px, py = normalise_grid(grid)
    allowed = (None if all_cells
               else outlined_cells(gds, chip_layer, grid, chip_min_size))
    if chip_size is None:
        detected = detect_chip_size(Path(gds), chip_layer, chip_min_size)
        win_x, win_y = detected if detected else (px, py)
        if verbose:
            sx, sy = (px - win_x) / 2.0, (py - win_y) / 2.0
            print(f"chip grid: {px:.1f} x {py:.1f} um pitch; keeping "
                  f"{win_x:.1f} x {win_y:.1f} um per chip"
                  + (f", dropping a {sx:.1f} x {sy:.1f} um street"
                     if max(sx, sy) > 0 else " (no street drawn)"))
    elif isinstance(chip_size, (list, tuple)):
        win_x, win_y = (float(v) for v in chip_size)
    else:
        win_x = win_y = float(chip_size)

    if layers is None:
        import gdstk
        lib = gdstk.read_gds(str(gds))
        cells = lib.top_level() or lib.cells
        layers = sorted({p.layer for c in cells for p in c.get_polygons()})

    # lattice extent: enough cells to cover everything on the requested layers
    cells = {}
    chips = defaultdict(dict)
    counters = defaultdict(lambda: defaultdict(int))

    area_floor = (min_area if isinstance(min_area, dict)
                  else defaultdict(lambda: min_area))
    if holes not in ("centers", "rings", "drop"):
        raise ValueError("holes must be 'centers', 'rings' or 'drop'")
    hole_layers = set(hole_layers or ())
    invert_layers = set(invert_layers or ())
    if chip_layer in invert_layers and verbose:
        print(f"warning: layer {chip_layer} is both the chip-outline layer and "
              f"listed in invert_layers. Its drawn shapes are the chip "
              f"outlines, so inverting it leaves nothing.")
    outside = defaultdict(int)          # lattice cells dropped for being unmarked
    hole_sizes = defaultdict(list)      # layer -> measured bounding-box sides
    loose_holes = defaultdict(list)     # (chip_id, layer) -> centres
    pending = defaultdict(list)         # (chip_id, layer) -> object records

    def cell_of(cx, cy):
        return (int(np.floor((cx - x0) / px)), int(np.floor((cy - y0) / py)))

    def window_for(key):
        if key not in cells:
            c, r = key
            cx = x0 + (c + 0.5) * px
            cy = y0 + (r + 0.5) * py
            cells[key] = (cx, cy, box(cx - win_x / 2, cy - win_y / 2,
                                      cx + win_x / 2, cy + win_y / 2))
        return cells[key]

    def emit(chip_id, layer, geom, origin):
        """Buffer one polygon as (outer perimeter, inserts, hole centres).

        Nothing is keyed yet: objects are named only once the whole layer is
        known, so that index 00000 is the largest object on the chip.
        """
        outer, inserts, centres = None, [], []
        for i, arr in enumerate(rings_to_array(geom)):
            arr = arr - origin
            if i == 0:
                outer = arr
                continue
            if holes != "rings" and hole_size(arr) <= hole_max_size:
                if holes == "centers":
                    centres.append(ring_centroid(arr))
                    hole_sizes[layer].append(hole_size(arr))
                continue
            inserts.append(arr)
        inserts.sort(key=order_key)
        pending[(chip_id, layer)].append(
            (outer, inserts, np.array(centres) if centres else None))

    for layer in layers:
        t0 = time.time()
        floor = area_floor.get(layer, 0.0) if isinstance(min_area, dict) else min_area

        # ---- negative tone: the metal is whatever was NOT drawn
        if layer in invert_layers:
            polys, scale, _ = load_polygons(Path(gds), layer, datatype)
            if not polys:
                if verbose:
                    print(f"layer {layer}: nothing to export")
                continue
            cleared, hole_pts = [], []
            for q in polys:
                pts = np.asarray(q.points, dtype=float) * scale
                if hole_size(pts) <= hole_max_size:
                    hole_pts.append(pts)       # a perforation, not a clearance
                else:
                    g = Polygon(pts)
                    cleared.append(g if g.is_valid else g.buffer(0))
            cleared = unary_union(cleared) if cleared else None

            keys = sorted(allowed) if allowed is not None else sorted(
                {cell_of(*np.asarray(q.points).mean(axis=0) * scale)
                 for q in polys})
            n_parts = 0
            for key in keys:
                cx, cy, window = window_for(key)
                chip_id = f"C{key[0]}R{key[1]}"
                plane = window.difference(cleared) if cleared is not None else window
                if simplify > 0:
                    plane = plane.simplify(simplify, preserve_topology=True)
                for part in explode(plane):
                    if part.area <= 0 or (floor and part.area < floor):
                        continue
                    emit(chip_id, layer, part, np.array([cx, cy]))
                    n_parts += 1
            if holes != "drop":
                for pts in hole_pts:
                    ctr = ring_centroid(np.vstack([pts, pts[:1]]))
                    key = cell_of(*ctr)
                    if allowed is not None and key not in allowed:
                        outside[key] += 1
                        continue
                    cx, cy, window = window_for(key)
                    if not window.contains(Point(ctr)):
                        continue               # in the street, not on the chip
                    loose_holes[(f"C{key[0]}R{key[1]}", layer)].append(
                        ctr - np.array([cx, cy]))
                    hole_sizes[layer].append(hole_size(pts))
            if verbose:
                print(f"layer {layer}: inverted -> {n_parts} metal objects, "
                      f"{len(hole_pts)} perforations, {time.time() - t0:.1f}s")
                if n_parts == 0:
                    print(f"   layer {layer} inverted to nothing: the drawn "
                          f"shapes cover every chip, so there is no metal "
                          f"left. Is this layer really negative tone?")
            continue

        shapes, _, _ = layer_shapes(gds, layer, datatype, simplify)

        # A layer that is overwhelmingly made of identical small polygons is a
        # fill array drawn as separate shapes, not thousands of components.
        # Emitting them as objects is the difference between 49 volumes and
        # 29,870, so it is worth noticing without being asked.
        if layer not in hole_layers and shapes:
            small = [g for g in shapes
                     if hole_size(np.asarray(g.exterior.coords)) <= hole_max_size]
            frac = len(small) / len(shapes)
            if len(small) >= AUTO_HOLE_MIN_COUNT and frac >= AUTO_HOLE_MIN_FRACTION:
                if auto_holes:
                    hole_layers.add(layer)
                    if verbose:
                        print(f"layer {layer}: {len(small)} of {len(shapes)} "
                              f"objects are <= {hole_max_size} um across "
                              f"({100*frac:.1f}%); treating them as a "
                              f"perforation array (auto_holes is on).")
                elif verbose:
                    print(f"layer {layer}: {len(small)} of {len(shapes)} objects "
                          f"are <= {hole_max_size} um across ({100*frac:.1f}%). "
                          f"That looks like a perforation array drawn as "
                          f"separate polygons, and emitting it as components "
                          f"gives {len(shapes)} volumes instead of "
                          f"{len(shapes) - len(small)}. If so, re-run with "
                          f"--hole-layers {layer} (or --invert-layers {layer} "
                          f"if the layer is also negative tone).")

        if not shapes:
            if verbose:
                print(f"layer {layer}: nothing to export")
            continue

        for g in shapes:
            if floor and g.area < floor:
                continue
            gx0, gy0, gx1, gy1 = g.bounds
            c_lo = int(np.floor((gx0 - x0) / px))
            c_hi = int(np.floor((gx1 - x0) / px))
            r_lo = int(np.floor((gy0 - y0) / py))
            r_hi = int(np.floor((gy1 - y0) / py))

            for c in range(c_lo, c_hi + 1):
                for r in range(r_lo, r_hi + 1):
                    key = (c, r)
                    if allowed is not None and key not in allowed:
                        outside[key] += 1
                        continue
                    if key not in cells:
                        cx = x0 + (c + 0.5) * px
                        cy = y0 + (r + 0.5) * py
                        cells[key] = (cx, cy,
                                      box(cx - win_x / 2, cy - win_y / 2,
                                          cx + win_x / 2, cy + win_y / 2))
                    cx, cy, window = cells[key]
                    parts = explode(g.intersection(window))
                    chip_id = f"C{c}R{r}"
                    for part in parts:
                        if part.area <= 0 or (floor and part.area < floor):
                            continue
                        origin = np.array([cx, cy])

                        # a small standalone object on a hole layer is a hole,
                        # not an object: record its centre and move on
                        if (layer in hole_layers and
                                hole_size(np.asarray(part.exterior.coords))
                                <= hole_max_size):
                            arr = np.asarray(part.exterior.coords) - origin
                            if holes != "drop":
                                loose_holes[(chip_id, layer)].append(
                                    ring_centroid(arr))
                                hole_sizes[layer].append(hole_size(arr))
                            continue

                        emit(chip_id, layer, part, origin)
        if verbose:
            print(f"layer {layer}: {len(shapes)} objects merged, "
                  f"placed in {time.time() - t0:.1f}s")

    if outside and verbose:
        n = sum(outside.values())
        print(f"note: {n} objects sit in {len(outside)} lattice cells that no "
              f"chip outline marks, and were dropped. Pass all_cells=True to "
              f"keep them: {sorted(outside)[:4]}")

    # name the objects: largest outer perimeter first, then its inserts
    for (chip_id, layer), records in pending.items():
        records.sort(key=lambda rec: order_key(rec[0]))
        for idx, (outer, inserts, centres) in enumerate(records):
            base = f"L{layer}_{idx:05d}"
            chips[chip_id][base] = outer
            for n, arr in enumerate(inserts, start=1):
                chips[chip_id][f"{base}_insert{n}"] = arr
            if centres is not None:
                chips[chip_id][f"{base}_holes"] = centres
        counters[chip_id][layer] = len(records)

    for (chip_id, layer), pts in loose_holes.items():
        chips[chip_id][f"L{layer}_holes"] = np.array(pts)

    # renumber chips so C0R0 is the lower-left populated cell, and attach metadata
    if not chips:
        return {}

    if verbose:
        for layer in sorted(hole_sizes):
            sizes = np.asarray(hole_sizes[layer])
            print(f"layer {layer}: {len(sizes)} holes -> centres, "
                  f"size {sizes.min():.3f} to {sizes.max():.3f} um "
                  f"(cap {hole_max_size} um)")

    used = sorted((int(k.split("R")[0][1:]), int(k.split("R")[1])) for k in chips)
    cmin = min(c for c, _ in used)
    rmin = min(r for _, r in used)
    out = {}
    for (c, r) in used:
        old = f"C{c}R{r}"
        new = f"C{c - cmin}R{r - rmin}"
        cx, cy, _ = cells[(c, r)]
        d = chips[old]
        d[META_CENTER] = np.array([cx, cy], dtype=float)
        d[META_LAYERS] = np.array(sorted(counters[old]), dtype=int)
        out[new] = d
    return out


# --------------------------------------------------------------------- helpers
def objects_on_layer(chip: dict, layer: int) -> dict:
    """``{key: array}`` for one layer of one chip, outer perimeters only.

    Ordered largest first, so the first entry on a ground-plane layer is the
    plane itself. Inserts and hole centres are excluded; fetch those with
    inserts_of() and hole_centers().
    """
    pre = f"L{layer}_"
    return {k: v for k, v in chip.items()
            if k.startswith(pre) and k.count("_") == 1
            and not k.endswith("_holes")}


def inserts_of(chip: dict, key: str) -> list:
    """Closed perimeters of everything cut into an object, largest first.

    These are the real interior boundaries -- device clearances, waveguide
    gaps, cut-outs. Small perforations are not among them; those are centre
    points, see hole_centers().
    """
    names = [k for k in chip if k.startswith(key + "_insert")]
    return [chip[k] for k in sorted(names,
                                    key=lambda k: int(k.rsplit("insert", 1)[1]))]


def hole_centers(chip: dict, key=None, layer=None) -> np.ndarray:
    """Hole centres: for one object (``key``), or every hole on a ``layer``.

    Returns an (M, 2) array, empty if there are none.
    """
    if key is not None:
        return chip.get(f"{key}_holes", np.empty((0, 2)))
    pre = f"L{layer}_"
    parts = [v for k, v in chip.items()
             if k.startswith(pre) and k.endswith("_holes")]
    return np.concatenate(parts) if parts else np.empty((0, 2))


def chip_summary(chips: dict):
    """Rows of ``(chip_id, layer, objects, inserts, hole_centres, vertices)``."""
    rows = []
    for chip_id, d in chips.items():
        per_layer = defaultdict(lambda: [0, 0, 0, 0])
        for k, v in d.items():
            if k.startswith("_"):
                continue
            layer = int(k.split("_")[0][1:])
            stat = per_layer[layer]
            if k.endswith("_holes"):
                stat[2] += len(v)          # one row per hole centre
            elif "_insert" in k:
                stat[1] += 1               # closed insert
                stat[3] += len(v)
            else:
                stat[0] += 1               # outer perimeter
                stat[3] += len(v)
        for layer, (n, nr, nc, nv) in sorted(per_layer.items()):
            rows.append((chip_id, layer, n, nr, nc, nv))
    return rows


# --------------------------------------------------------------------- storage
def save_npz(chips: dict, out_dir, compress=True):
    """One .npz per chip. Keys survive exactly; load with np.load."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fn = np.savez_compressed if compress else np.savez
    for chip_id, d in chips.items():
        p = out_dir / f"{chip_id}.npz"
        fn(p, **d)
        paths.append(p)
    return paths


def save_single_npz(chips: dict, path, compress=True):
    """All chips in one .npz, keys prefixed ``C0R0__L2_00007``."""
    flat = {f"{chip_id}__{k}": v for chip_id, d in chips.items() for k, v in d.items()}
    fn = np.savez_compressed if compress else np.savez
    fn(Path(path), **flat)
    return Path(path)


def load_npz(path, into_memory=True):
    """Load a per-chip .npz back into a plain dict of arrays."""
    z = np.load(Path(path))
    return {k: z[k] for k in z.files} if into_memory else z


def load_single_npz(path):
    """Load a combined .npz back into ``{chip_id: {key: array}}``."""
    z = np.load(Path(path))
    out = defaultdict(dict)
    for k in z.files:
        chip_id, _, name = k.partition("__")
        out[chip_id][name] = z[k]
    return dict(out)


def save_packed(chips: dict, path):
    """Compact alternative: concatenated vertices plus offsets, per chip.

    A .npz holds one zip member per key, so a chip with tens of thousands of
    small objects writes slowly and stores badly. This packs each chip into
    three arrays instead, which is far faster and smaller; load_packed()
    gives the same dictionary back.
    """
    flat = {}
    for chip_id, d in chips.items():
        keys = [k for k in sorted(d) if not k.startswith("_")]
        if keys:
            xy = np.concatenate([d[k] for k in keys])
            lens = np.array([len(d[k]) for k in keys], dtype=np.int64)
        else:
            xy = np.zeros((0, 2))
            lens = np.zeros(0, dtype=np.int64)
        flat[f"{chip_id}__xy"] = xy
        flat[f"{chip_id}__offsets"] = np.concatenate([[0], np.cumsum(lens)])
        flat[f"{chip_id}__keys"] = np.array(keys, dtype=object)
        for m in (k for k in d if k.startswith("_")):
            flat[f"{chip_id}__{m}"] = d[m]
    np.savez_compressed(Path(path), **flat, allow_pickle=True)
    return Path(path)


def load_packed(path):
    """Inverse of save_packed."""
    z = np.load(Path(path), allow_pickle=True)
    chips = defaultdict(dict)
    for name in z.files:
        chip_id, _, part = name.partition("__")
        chips[chip_id][part] = z[name]
    out = {}
    for chip_id, parts in chips.items():
        xy, off = parts["xy"], parts["offsets"]
        d = {str(k): xy[off[i]:off[i + 1]]
             for i, k in enumerate(parts["keys"])}
        for m in (k for k in parts if k.startswith("_")):
            d[m] = parts[m]
        out[chip_id] = d
    return out

# ------------------------------------------------------------------------- cli
def parse_min_area(spec):
    """'30' -> 30.0;  '1:30,4:2' -> {1: 30.0, 4: 2.0}."""
    if ":" not in str(spec):
        return float(spec)
    out = {}
    for part in str(spec).split(","):
        layer, _, val = part.partition(":")
        out[int(layer)] = float(val)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build per-chip dictionaries of perimeter arrays from a GDS file.")
    ap.add_argument("gds")
    ap.add_argument("--layers", help="comma-separated layer numbers "
                                     "(default: every layer in the file)")
    ap.add_argument("--datatype", type=int, default=0,
                    help="datatype filter; -1 for any (default: 0)")
    ap.add_argument("--out", default="chips", help="output directory (default: chips/)")
    ap.add_argument("--chip-layer", type=int, default=0,
                    help="layer holding chip outlines (default: 0)")
    ap.add_argument("--chip-pitch",
                    help="override the detected chip pitch in um; one number, "
                         "or 'X,Y' for rectangular chips")
    ap.add_argument("--chip-origin",
                    help="override the lattice lower-left corner as 'X,Y' in um; "
                         "needs --chip-pitch too")
    ap.add_argument("--chip-min-size", type=float, default=1000.0,
                    help="ignore outline rectangles smaller than this in um when "
                         "detecting the lattice (default: 1000)")
    ap.add_argument("--chip-size", type=float,
                    help="window kept per chip in um (default: the chip as "
                         "drawn, excluding the dicing street; pass the full "
                         "pitch to keep the street)")
    ap.add_argument("--all-cells", action="store_true",
                    help="keep every lattice cell touched by geometry, not just "
                         "the block marked out by the chip outlines")
    ap.add_argument("--simplify", type=float, default=0.0,
                    help="Douglas-Peucker tolerance in um (default: 0)")
    ap.add_argument("--min-area", default="0",
                    help="skip objects below this area in um^2. Either one number "
                         "for all layers, or per-layer pairs such as '1:30' to "
                         "drop 5x5 um fill on layer 1 only (default: 0)")
    ap.add_argument("--holes", choices=("centers", "rings", "drop"),
                    default="centers",
                    help="how to represent small enclosed voids: record a centre "
                         "point per hole and keep outlines hole-free (default), "
                         "keep them as explicit interior rings, or discard them")
    ap.add_argument("--hole-max-size", type=float, default=10.0,
                    help="largest a hole may be, in um across; enclosed voids "
                         "bigger than this are treated as real geometry. A cap, "
                         "not the hole size, which is found from the file "
                         "(default: 10)")
    ap.add_argument("--invert-layers",
                    help="comma-separated layers drawn with negative tone, e.g. "
                         "'1'. The metal is rebuilt as the chip minus the drawn "
                         "clearances and perforations become hole centres")
    ap.add_argument("--auto-holes", action="store_true",
                    help="act on the perforation-array diagnosis instead of "
                         "only reporting it. Off by default: a dense layer of "
                         "small real features looks the same and would be "
                         "silently reduced to points")
    ap.add_argument("--hole-layers",
                    help="comma-separated layers where small standalone objects "
                         "are holes too, e.g. '1' for a fill array drawn as "
                         "separate squares")
    ap.add_argument("--single-file", action="store_true",
                    help="one combined .npz instead of one file per chip")
    ap.add_argument("--packed", action="store_true",
                    help="write the compact packed format (much faster for "
                         "layers with very many objects)")
    args = ap.parse_args(argv)

    src = Path(args.gds)
    if not src.exists():
        sys.exit(f"no such file: {src}")

    layers = ([int(v) for v in args.layers.split(",")] if args.layers else None)
    chips = build_chip_geometry(
        src, layers=layers,
        datatype=None if args.datatype < 0 else args.datatype,
        chip_layer=args.chip_layer, chip_size=args.chip_size,
        chip_min_size=args.chip_min_size,
        grid=((tuple(float(v) for v in args.chip_origin.split(","))
               + tuple(float(v) for v in args.chip_pitch.split(",")))
              if args.chip_pitch and args.chip_origin else None),
        all_cells=args.all_cells,
        holes=args.holes, hole_max_size=args.hole_max_size,
        hole_layers=([int(v) for v in args.hole_layers.split(",")]
                     if args.hole_layers else None),
        invert_layers=([int(v) for v in args.invert_layers.split(",")]
                       if args.invert_layers else None),
        auto_holes=args.auto_holes,
        simplify=args.simplify,
        min_area=parse_min_area(args.min_area))
    if not chips:
        sys.exit("no geometry found")

    print(f"\n{len(chips)} chips")
    print(f"{'chip':>6s} {'layer':>6s} {'objects':>9s} {'inserts':>8s} "
          f"{'holes':>7s} {'vertices':>10s}")
    for chip_id, layer, n, nr, nc, nv in chip_summary(chips):
        print(f"{chip_id:>6s} {layer:>6d} {n:>9d} {nr:>8d} {nc:>7d} {nv:>10d}")

    out = Path(args.out)
    (out.parent if out.suffix else out).mkdir(parents=True, exist_ok=True)
    if args.packed:
        p = save_packed(chips, out.with_suffix(".npz") if out.suffix else
                        out / "chips_packed.npz")
        print(f"\nwrote packed -> {p}")
    elif args.single_file:
        p = save_single_npz(chips, out.with_suffix(".npz") if out.suffix else
                            out / "chips.npz")
        print(f"\nwrote -> {p}")
    else:
        paths = save_npz(chips, out)
        print(f"\nwrote {len(paths)} files -> {out}/")


if __name__ == "__main__":
    main()
