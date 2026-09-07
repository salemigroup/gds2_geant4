#!/usr/bin/env python3
"""
make_geant4.py — write Geant4 geometry classes from chip perimeters.

Reads a config file naming the chip, the layers, their materials and
thicknesses, and which entries to emit. For each selected entry it writes a
matching pair of .cc/.hh files built around a G4ExtrudedSolid, following the
style of geometry_island.

  * one class per object; inserts are subtracted inside that same class
  * holes get a single shared class, geometry_hole, subtracted by the others
  * where two layers overlap in plan view the lower one wins, resolved in
    shapely before any C++ is written, so no runtime booleans are needed
  * near-coincident edges are snapped so that sub-nanometre gaps close exactly

    python make_geant4.py config.yaml
    python make_geant4.py config.yaml --dry-run

Requires: numpy, shapely, pyyaml (JSON configs work without pyyaml), and
chip_geometry.py alongside this file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import snap, unary_union
from shapely.strtree import STRtree

# coordinates out of chip_geometry are micrometres
UM = 1.0


# ------------------------------------------------------------------- config
DEFAULTS = {
    "chip": None,
    "layer_thickness_um": 0.1,
    "output_dir": "g4_geometry",
    "touch_tolerance_nm": 1.0,
    "hole": {"shape": "square", "size_um": 5.0, "class_name": "geometry_hole"},
    "class_prefix": "geometry",
    "warn_vertices": 2000,
    "warn_holes": 200,
}


def load_config(path):
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            sys.exit("this config is YAML; install pyyaml or use JSON")
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)

    out = dict(DEFAULTS)
    out.update(cfg)
    out["hole"] = {**DEFAULTS["hole"], **cfg.get("hole", {})}
    for key in ("source", "layers", "items"):
        if not out.get(key):
            sys.exit(f"config is missing '{key}'")
    return out


# ---------------------------------------------------------------- geometry
def rebuild_polygon(chip, key):
    """Shapely polygon for one entry: outer perimeter minus its inserts."""
    from chip_geometry import inserts_of
    outer = np.asarray(chip[key], dtype=float)
    holes = [np.asarray(a, dtype=float)[:-1] for a in inserts_of(chip, key)]
    poly = Polygon(outer[:-1], holes)
    return poly if poly.is_valid else poly.buffer(0)


def explode(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def resolve_overlaps(items, skip_layers=(), stack_order=None, verbose=True):
    """Lower layers win: subtract everything below from each layer in turn.

    Every non-substrate layer sits on the substrate at the same z, so two of
    them cannot share a footprint. Where they do, the lower one keeps the
    material and the upper one is trimmed. For a junction lead drawn on top of
    a pad this removes the part lying over the pad and leaves the pad whole,
    which is what a 2D transport model wants: the two end up sharing an edge,
    so a border surface is still generated between them.

    ``stack_order`` lists layers bottom to top. It defaults to ascending layer
    number, which matches most layouts but is worth stating explicitly when the
    fab's numbering does not follow deposition order.

    Doing this here rather than with G4SubtractionSolid keeps the emitted C++ to
    plain extruded solids.

    Only selected objects take part: a layer that is never emitted cannot
    shadow one that is. The substrate is skipped as well -- it sits below the
    others in z rather than beside them, so a shared footprint is expected and
    is not an overlap.
    """
    placed = None
    n_cut = 0
    by_layer = defaultdict(list)
    for it in items:
        if it["layer"] in skip_layers:
            continue
        by_layer[it["layer"]].append(it)

    if stack_order:
        ranked = [l for l in stack_order if l in by_layer]
        ranked += [l for l in sorted(by_layer) if l not in ranked]
    else:
        ranked = sorted(by_layer)                     # ascending = bottom up

    for layer in ranked:
        layer_items = by_layer[layer]
        if placed is not None:
            for it in layer_items:
                before = it["geom"].area
                cut = it["geom"].difference(placed)
                if cut.is_empty:
                    it["geom"] = cut
                    n_cut += 1
                    continue
                if abs(before - cut.area) > 1e-9:
                    n_cut += 1
                it["geom"] = cut
        here = unary_union([it["geom"] for it in layer_items if not it["geom"].is_empty])
        placed = here if placed is None else unary_union([placed, here])

    if verbose and n_cut:
        print(f"overlap: {n_cut} objects trimmed where a lower layer already "
              f"occupied the same footprint")
    return items


def close_small_gaps(items, tol_um, verbose=True):
    """Snap edges that are within ``tol_um`` of each other into contact.

    Anything closer than the tolerance is a rounding artefact rather than a
    designed gap, and Geant4 handles an exact shared face far better than a
    picometre sliver of vacuum. Afterwards every coordinate is put on the
    tolerance grid so the contact is exact in the emitted numbers too.
    """
    live = [it for it in items if not it["geom"].is_empty]
    if len(live) > 1:
        geoms = [it["geom"] for it in live]
        tree = STRtree(geoms)
        n_snapped = 0
        for i, it in enumerate(live):
            for j in tree.query(it["geom"].buffer(tol_um)):
                if j <= i:
                    continue
                d = geoms[i].distance(geoms[j])
                if 0 < d <= tol_um:
                    it["geom"] = snap(it["geom"], geoms[j], tol_um)
                    n_snapped += 1
        if verbose:
            print(f"gaps: {n_snapped} pairs closer than {tol_um * 1000:.1f} nm "
                  f"snapped into contact")

    for it in items:
        if it["geom"].is_empty:
            continue
        g = shapely.set_precision(it["geom"], tol_um)
        if not g.is_valid:
            g = g.buffer(0)
        it["geom"] = g
    return items


def as_ccw(coords):
    """G4ExtrudedSolid wants its perimeter anticlockwise.

    Boolean operations leave rings wound either way, and an interior ring comes
    back clockwise by convention, so every ring is re-oriented on the way out
    rather than trusting whatever shapely produced.
    """
    pts = list(coords)
    area = sum(x0 * y1 - x1 * y0
               for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    return pts if area > 0 else pts[::-1]


# ------------------------------------------------------------ C++ emission
def cpp_ident(text):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in text)


def fmt_points(coords, indent="        ", per_line=1):
    """G4TwoVector push_back lines for a ring, in micrometres."""
    out = []
    for x, y in coords:
        out.append(f"{indent}p.push_back(G4TwoVector({x:.4f}*um, {y:.4f}*um));")
    return "\n".join(out)


HEADER_TEMPLATE = '''#ifndef {guard}
#define {guard}

#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "G4ExtrudedSolid.hh"
#include "G4TwoVector.hh"
#include "G4SystemOfUnits.hh"
#include <vector>
#include <cmath>
#include <cstddef>

class G4LogicalVolume;
class G4VPhysicalVolume;

// {comment}
// Generated by make_geant4.py -- do not edit by hand.
// Coordinates are micrometres, measured from the centre of chip {chip}.
class {cls}
{{
    public:
        {cls}();
        ~{cls}();

        // alas...I am forced into camel_case by G4 conventions
        G4VPhysicalVolume* Construct(G4LogicalVolume* parent_log_volume,
                                     const G4ThreeVector& position,
                                     G4RotationMatrix* rotation = nullptr);

        G4LogicalVolume* GetLogicalVolume() const {{return logic_{name};}}
        G4VPhysicalVolume* GetPhysicalVolume() const {{return phys_{name};}}

        // half-thickness of this layer, for the detector construction to
        // stack against
        static G4double GetZHalfLength() {{return {z_half:.6f}*um;}}
        // cutting solids run past both faces so the subtraction is clean
        static G4double GetZCutLength() {{return {z_cut:.6f}*um;}}

{ring_functions}{hole_function}
    private:
        G4LogicalVolume* logic_{name} = nullptr;
        G4VPhysicalVolume* phys_{name} = nullptr;

}};

#endif
'''

SOURCE_TEMPLATE = '''#include "{header}"
#include "G4NistManager.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
{extra_includes}
// {comment}
// Generated by make_geant4.py -- do not edit by hand.
// Perimeter points live in the header; material is {material}.

G4VPhysicalVolume* {cls}::Construct(G4LogicalVolume* parent_log_volume,
{indent}const G4ThreeVector& position,
{indent}G4RotationMatrix* rotation){{

    G4NistManager* nist = G4NistManager::Instance();
    G4Material* material = nist->FindOrBuildMaterial("{material}");

    G4VSolid* solid_{name} = new G4ExtrudedSolid(
        "{solid_name}",{solid_pad}// Name
        OuterPolygon(),         // Perimeter
        GetZHalfLength(),       // Z half-length
        G4TwoVector(0,0), 1.0,  // Bottom face offset/scale
        G4TwoVector(0,0), 1.0   // Top face offset/scale
    );

{insert_block}{hole_block}
    logic_{name} = new G4LogicalVolume(solid_{name}, material, "logic_{name}");
    phys_{name} = new G4PVPlacement(rotation, position, logic_{name}, "phys_{name}", parent_log_volume, false, 0, true);

    return phys_{name};
}}

{cls}::{cls}() {{
}}

{cls}::~{cls}() {{
}}
'''

HOLE_HEADER = '''#ifndef {guard}
#define {guard}

#include "G4ExtrudedSolid.hh"
#include "G4TwoVector.hh"
#include "G4SystemOfUnits.hh"
#include <vector>
#include <cmath>
#include <cstddef>

class G4VSolid;

// A single fabrication hole, shared by every component that is perforated.
// The components subtract this solid at each recorded centre, so the hole
// pattern lives in one place.
// Generated by make_geant4.py -- do not edit by hand.
class {cls}
{{
    public:
        // z_half_length should run past both faces of the layer being cut
        explicit {cls}(G4double z_half_length);
        ~{cls}();

        G4VSolid* GetSolid() const {{return solid_hole;}}

        static G4double GetSize() {{return {size:.6f}*um;}}   // {shape}, across flats

        static std::vector<G4TwoVector> HolePolygon(){{
            std::vector<G4TwoVector> p;
            p.reserve({n});
{hole_points}
            return p;
        }}

    private:
        G4VSolid* solid_hole = nullptr;

}};

#endif
'''

HOLE_SOURCE = '''#include "{header}"
#include "G4SystemOfUnits.hh"
#include "G4VSolid.hh"

// Generated by make_geant4.py -- do not edit by hand.
// {shape} hole, {size:.4f} um across. Perimeter points live in the header.

{cls}::{cls}(G4double z_half_length) {{
    solid_hole = new G4ExtrudedSolid(
        "hole",                 // Name
        HolePolygon(),          // Perimeter
        z_half_length,          // Z half-length
        G4TwoVector(0,0), 1.0,  // Bottom face offset/scale
        G4TwoVector(0,0), 1.0   // Top face offset/scale
    );
}}

{cls}::~{cls}() {{
}}
'''


def ring_function(fname, coords):
    """An inline static member returning one ring; no free functions, no
    anonymous namespace, and defining it in the class body makes it implicitly
    inline so several translation units can include it safely."""
    return (f"        static std::vector<G4TwoVector> {fname}(){{\n"
            f"            std::vector<G4TwoVector> p;\n"
            f"            p.reserve({len(coords)});\n"
            f"{fmt_points(coords, indent='            ')}\n"
            f"            return p;\n"
            f"        }}\n")


def hole_function(coords):
    body = "\n".join(f"            p.push_back(G4TwoVector({x:.4f}*um, "
                     f"{y:.4f}*um));" for x, y in coords)
    return (f"\n        // centres of the perforations cut from this object\n"
            f"        static std::vector<G4TwoVector> HoleCenters(){{\n"
            f"            std::vector<G4TwoVector> p;\n"
            f"            p.reserve({len(coords)});\n"
            f"{body}\n"
            f"            return p;\n"
            f"        }}\n")


def hole_polygon_points(shape, size, segments=24):
    r = size / 2.0
    if shape == "square":
        pts = [(-r, -r), (r, -r), (r, r), (-r, r)]
    elif shape == "circle":
        t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        pts = [(r * np.cos(a), r * np.sin(a)) for a in t]
    else:
        sys.exit(f"hole shape must be 'square' or 'circle', got {shape!r}")
    return pts


def write_hole_class(cfg, out_dir):
    hole = cfg["hole"]
    cls = hole["class_name"]
    pts = hole_polygon_points(hole["shape"], hole["size_um"])
    hdr = out_dir / f"{cls}.hh"
    src = out_dir / f"{cls}.cc"
    hdr.write_text(HOLE_HEADER.format(guard=cls.upper() + "_HH", cls=cls,
                                      shape=hole["shape"], size=hole["size_um"],
                                      n=len(pts),
                                      hole_points=fmt_points(pts, indent="            ")))
    src.write_text(HOLE_SOURCE.format(header=hdr.name, cls=cls,
                                      shape=hole["shape"],
                                      size=hole["size_um"]))
    return [hdr, src]


def write_component(item, cfg, out_dir):
    cls = item["class"]
    name = item["name"]
    geom = orient(item["geom"], sign=1.0)   # CCW exterior, CW interiors
    hole_cls = cfg["hole"]["class_name"]

    outer = as_ccw(list(geom.exterior.coords)[:-1])
    inserts = [as_ccw(list(r.coords)[:-1]) for r in geom.interiors]
    centres = item["holes"]

    rings = [ring_function("OuterPolygon", outer)]
    for i, ring in enumerate(inserts, start=1):
        rings.append("\n" + ring_function(f"Insert{i}", ring))

    extra = ""
    if inserts or centres:
        extra += '#include "G4SubtractionSolid.hh"\n'
    if centres:
        extra += f'#include "{hole_cls}.hh"\n'

    insert_block = ""
    for i in range(1, len(inserts) + 1):
        insert_block += f'''
    G4VSolid* insert_solid_{i} = new G4ExtrudedSolid(
        "{name}_insert_{i}", Insert{i}(), GetZCutLength(),
        G4TwoVector(0,0), 1.0, G4TwoVector(0,0), 1.0);
    solid_{name} = new G4SubtractionSolid(
        "{name}_minus_insert_{i}", solid_{name}, insert_solid_{i});
'''

    hole_block = ""
    if centres:
        hole_block = f'''
    // perforations, subtracted from the shared {hole_cls} solid
    {hole_cls} hole(GetZCutLength());
    const std::vector<G4TwoVector> centers = HoleCenters();
    for (std::size_t i = 0; i < centers.size(); ++i) {{
        solid_{name} = new G4SubtractionSolid(
            "{name}_minus_hole", solid_{name}, hole.GetSolid(), nullptr,
            G4ThreeVector(centers[i].x(), centers[i].y(), 0));
    }}
'''

    comment = (f"chip {item['chip']}, layer {item['layer']}, entry "
               f"{item['entry']}")
    solid_name = f"{name}_outer"
    hdr = out_dir / f"{cls}.hh"
    src = out_dir / f"{cls}.cc"
    hdr.write_text(HEADER_TEMPLATE.format(
        guard=cls.upper() + "_HH", cls=cls, name=name, comment=comment,
        chip=item["chip"], z_half=item["z_half"], z_cut=item["z_half"] * 2.0,
        ring_functions="".join(rings),
        hole_function=hole_function(centres) if centres else ""))
    src.write_text(SOURCE_TEMPLATE.format(
        header=hdr.name, cls=cls, name=name, comment=comment,
        material=item["material"], solid_name=solid_name,
        solid_pad=" " * max(1, 24 - len(solid_name) - 3),
        insert_block=insert_block, hole_block=hole_block,
        extra_includes=extra,
        indent=" " * (len("G4VPhysicalVolume* ") + len(cls) + len("::Construct("))))
    return [hdr, src]


# --------------------------------------------------------------------- main
def collect_items(chips, cfg):
    """Turn the config's 'items' selection into a list of work records."""
    from chip_geometry import hole_centers, objects_on_layer

    chip_id = cfg["chip"] or next(iter(chips))
    if chip_id not in chips:
        sys.exit(f"no chip {chip_id!r}; available: {sorted(chips)}")
    chip = chips[chip_id]

    materials = {lay["layer"]: lay["material"] for lay in cfg["layers"]}
    thickness = {lay["layer"]: lay.get("thickness_um",
                                       cfg["layer_thickness_um"])
                 for lay in cfg["layers"]}
    sub = cfg.get("substrate")
    if sub:
        materials[sub["layer"]] = sub["material"]
        thickness[sub["layer"]] = sub["thickness_um"]

    items, seen, duplicates = [], set(), []
    for spec in cfg["items"]:
        layer = spec["layer"]
        if layer not in materials:
            sys.exit(f"layer {layer} is selected in 'items' but has no "
                     f"material in 'layers' or 'substrate'")
        available = objects_on_layer(chip, layer)
        wanted = spec.get("entries", "all")
        if wanted == "all":
            keys = list(available)
        elif isinstance(wanted, int):
            keys = list(available)[:wanted]
        else:
            keys = [k for k in wanted if k in available]
            missing = [k for k in wanted if k not in available]
            if missing:
                sys.exit(f"layer {layer} has no entries {missing}")
        if thickness[layer] <= 0:
            sys.exit(f"layer {layer} has thickness {thickness[layer]}; an "
                     f"extruded solid needs a positive thickness")
        for key in keys:
            if (layer, key) in seen:
                duplicates.append(key)
                continue
            seen.add((layer, key))
            items.append(dict(
                chip=chip_id, layer=layer, entry=key,
                material=spec.get("material", materials[layer]),
                z_half=thickness[layer] / 2.0,
                geom=rebuild_polygon(chip, key),
                holes=[tuple(c) for c in hole_centers(chip, key)],
            ))
    if duplicates:
        print(f"  {len(duplicates)} entries selected more than once, "
              f"kept once each: {sorted(set(duplicates))[:4]}")
    return chip_id, items


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate Geant4 geometry classes from chip perimeters.")
    ap.add_argument("config")
    ap.add_argument("--no-detector", action="store_true",
                    help="skip the detector construction even if the config "
                         "has a 'detector' block")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written without writing it")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    src = Path(cfg["source"])
    if not src.exists():
        sys.exit(f"no such source: {src}")

    from chip_geometry import build_chip_geometry, load_npz, load_packed

    if src.suffix.lower() == ".gds":
        # only the layers actually selected are built; a layer defined in
        # 'layers' but never used in 'items' costs nothing
        layers = sorted({spec["layer"] for spec in cfg["items"]})
        chips = build_chip_geometry(src, layers=layers,
                                    **cfg.get("build_options", {}))
    else:
        z = np.load(src, allow_pickle=True)
        chips = (load_packed(src) if any(n.endswith("__xy") for n in z.files)
                 else {cfg["chip"] or src.stem: load_npz(src)})

    chip_id, items = collect_items(chips, cfg)
    print(f"chip {chip_id}: {len(items)} objects selected")

    substrate_layer = (cfg["substrate"]["layer"] if cfg.get("substrate")
                       else None)
    items = resolve_overlaps(
        items, skip_layers=() if substrate_layer is None else {substrate_layer},
        stack_order=cfg.get("stack_order"))
    items = close_small_gaps(items, cfg["touch_tolerance_nm"] / 1000.0)

    # a subtraction can split an object or empty it entirely
    expanded = []
    for it in items:
        parts = explode(it["geom"])
        if not parts:
            print(f"  {it['entry']}: fully covered by a lower layer, skipped")
            continue
        for n, part in enumerate(parts, start=1):
            rec = dict(it, geom=part)
            suffix = "" if len(parts) == 1 else f"_part{n}"
            rec["holes"] = [c for c in it["holes"] if part.contains(Point(c))]
            rec["name"] = cpp_ident(f"{it['chip']}_{it['entry']}{suffix}")
            rec["class"] = f"{cfg['class_prefix']}_{rec['name']}"
            expanded.append(rec)
        if len(parts) > 1:
            print(f"  {it['entry']}: split into {len(parts)} parts by the "
                  f"overlap cut")

    for it in expanded:
        nv = len(it["geom"].exterior.coords) - 1
        if nv > cfg["warn_vertices"]:
            print(f"  {it['class']}: {nv} vertices in one extruded solid")
        if len(it["holes"]) > cfg["warn_holes"]:
            print(f"  {it['class']}: {len(it['holes'])} hole subtractions, "
                  f"this will be slow to navigate in Geant4")

    if args.dry_run:
        print(f"\nwould write {2 * (len(expanded) + 1)} files:")
        for it in expanded[:10]:
            print(f"   {it['class']}.hh / .cc  "
                  f"({len(it['geom'].exterior.coords) - 1} verts, "
                  f"{len(it['geom'].interiors)} inserts, "
                  f"{len(it['holes'])} holes)")
        if len(expanded) > 10:
            print(f"   ... and {len(expanded) - 10} more")
        return

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    written = write_hole_class(cfg, out_dir)
    for it in expanded:
        written += write_component(it, cfg, out_dir)

    if cfg.get("detector") and not args.no_detector:
        from detector_builder import write_detector
        written += write_detector(expanded, cfg, out_dir, substrate_layer)

    print(f"\nwrote {len(written)} files -> {out_dir}/")


if __name__ == "__main__":
    main()
