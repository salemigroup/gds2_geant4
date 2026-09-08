# gds2_geant4

Author: Ryan Gibbons, rmg at lbl dot gov

**In active development! Send me your requests/bugs/complaints.**

## Description
* Converts GDS files into Geant4 geometries for use with G4CMP. 
* Designed for superconducting devices, ~200 nm films on ~0.5 mm substrate.
    - For other applications, noteably semiconductor devices, this will likely not work properly.
* This is not a fully automatic process, users are expected to still use their brains.


### AI Disclaimer
* This project contains code generated with Claude.
* Human review of the outputs is part of how this utility works. The user is responsible for checking outputs are accurate.


### Change log
* 2025-09-08: First complete iteration.

## Dependencies 

### Running python scripts:
* Python ≥ 3.8
    - shapely ≥ 2.0
    - gdstk
    - pyyaml 
    - numpy
    - matplotlib

Install using pip:
`
pip install shapely>=2.0, gdstk, pyyaml, numpy, matplotlib
`

### Geant4 output:
* The generated Geant4 code is intended for Geant4 v11.4.1, which is required for quasiparticle dynamics in G4CMP.
* If you only want phonon physics, the code should work in Geant4 v10.4 - v10.7. 
* C++ code follows C++11 standard, and was tested with the GCC compiler. 


## Quick start
```bash
# 1. Export GDS to npz
python chip_geometry.py wafer.gds --layers 0,1,2,3,4 --invert-layers 1 --out chips/
 
# 2. Verify each component
python plot_entries.py chips/C1R1.npz --layers 2 --overview --out plots/
 
# 3. Generate Geant4
python make_geant4.py config.yaml
```



## More details
### 1. Export GDS to npz files
* `python chip_geometry.py file_name.gds`
* Specify which superconducting layers you want to include. 
    - If a layer is inverted (used for etching) then use `--invert-layer`
* Each diced chip in the wafer has a npz output, ordered by column and row in the wafer. E.g., `C1R2.npz`
* You should not need to open the npz files, but they may be useful for other applications.
* The npz structure is explained further below.


### 2. Generate plots of each component
* `python plot_entries.py CXRY.npz`
* This will create and save plots of each component in the dictionary.
* If you are very confident in this code, you can skip directly to step 3.


### 3. Write the config file
* Inspect the plots you created in Step 2. This is critical for having accurate output. 
* If any component doesn't look quite right, you will need to manually insert it later. 
* Create a yaml config file based on the plots. See the example config file.


### 4. Create Geant4 files
* `python make_geant4.py config.yml`
* This creates the geometry cc/hh files. Each component recieves a cc/hh file, which are included in a detector construction cc/hh file. 
* By default, every component is a child volume of the vacuum world volume.
* Repeated components are not checked. If really want to G4 your x100 junction test chip, this will be terribly inefficient.


### 5. Add to your project
* This can be as simple as copy paste all of the files created in Step 4, and making sure your existing project knows about these files. 
* The user is assumed to have basic knowledge of Geant4 code structure.


## Important notes
* Your GDS file is assumed to have die trenches, or only be a single chip.
* Complicated features, such as logos and debug/test features, might not turn out correct. It it important you manually verify each feature in Steps 2 and 3 above.
* If you have a ground plane with lots of flux trapping holes, it might take several minutes to compile your project.
* Objects are created using G4ExtrudedSolid of a 2D polygon. This minimizes artificial surfaces which should speed up performance.
* Currently, only NIST materials can be specified e.g., "G4_Si". Custom materials must be manually written in your project.
* In G4CMP, quasiparticle dynamics are only done in 2D. For 2D film overlaps, only the bottom layer is used for that region. Similarly, all films have exactly the same thickness, specified in the config file.
    * Refer to the quasiparticle example in G4CMP for more details on how to design your project.
    * Remember: your substrate contains >99.99% of the total mass and dicing introduces a larger uncertainty than the film mass.
* Any overlap/gap between features that are < 1 nm are automatically rounded to be touching by default. This can be changed in the config file.
* Viewing the Geant4 geometry can be tricky since the meshing here is complicated. Inserting a bunch of geantinos that immediately die or having a very low energy electrons is one way around this.


## Reference of arguments and syntax

### chip_geometry.py
 
```
python chip_geometry.py gds_file.gds [options]
```
 
| Argument | Default | Description |
|---|---|---|
| `gds_file.gds` | required | Input GDS file |
| `--layers` | every layer | Comma-separated layer numbers, e.g. `2,3` |
| `--invert-layers` | none | Comma-separated layer number to invert, e.g. `2,3` |
| `--datatype` | `0` | Datatype filter; `-1` for any |
| `--out` | `chips/` | Output directory, or a `.npz` path with `--packed`/`--single-file` |
| `--chip-layer` | `0` | Layer holding the substrate/die trenches |
| `--chip-pitch` | detected | Override the lattice pitch in um; one number, or `X,Y` for rectangular chips. Needs `--chip-origin` |
| `--chip-origin` | detected | Lattice lower-left corner as `X,Y` in um |
| `--chip-min-size` | `1000` | Minimum chip size, in um |
| `--chip-size` | chip with no trenches | Window kept per chip in um, excluding the dicing trenches. Pass the full pitch to keep the trenches|
| `--all-cells` | off | Keep every lattice cell any geometry touches, not just the block the outlines mark out |
| `--simplify` | `0` | Douglas-Peucker tolerance in um. `0.005` roughly halves vertex counts for a 5 nm deviation |
| `--min-area` | `0` | Skip objects below this area in um². One number, or per-layer pairs like `1:30` |
| `--holes` | `centers` | `centers` records a point per hole and leaves outlines hole-free; `rings` keeps them as inserts; `drop` discards them |
| `--hole-max-size` | `20` | Largest a hole may be, in um across. A **cap** separating holes from real geometry, not the hole size |
| `--hole-layers` | none | Layers where small *standalone* objects are holes too, e.g. `1` for a fill array drawn as separate squares |
| `--single-file` | off | One combined `.npz` instead of one per chip |
| `--packed` | off | Compact format: concatenated vertices plus offsets. Much faster for layers with tens of thousands of objects |
 

&nbsp;
&nbsp;
&nbsp;
&nbsp;

 
### plot_entries.py
 
```
python plot_entries.py SOURCE --layer N [options]
```
 
One image per object, drawn with its inserts cut out and its hole centres
marked. Holes never get an image of their own.
 
| Argument | Default | Description |
|---|---|---|
| `source` | required | A `.npz` from `chip_geometry.py`, or a `.gds` directly |
| `--chip` | — | Chip id, e.g., `C1R1`. Required unless the source is a single-chip `.npz` |
| `--layers` | required | Layer number |
| `--overview` | off | Creates a plot of the entire chip |
| `--out` | `plots/` | Output directory, created if needed |
| `--limit` | all | Plot only the first N objects |
| `--hole-max-size` | `20` | Skip objects this size or smaller in um across, so perforations don't each get an image. `0` plots everything |
| `--dpi` | `130` | Raster resolution |
| `--format` | `png` | Any matplotlib format: `png`, `pdf`, `svg` |
| `--list` | off | List the chips and layers available, then exit |
 
Files are named `<chip>_<entry>.<format>`, e.g. `C1R1_L2_00000.png`.
 
&nbsp;
&nbsp;
&nbsp;
&nbsp;
 
### make_geant4.py
 
```
python make_geant4.py CONFIG [--dry-run] [--no-detector]
```
 
| Argument | Default | Description |
|---|---|---|
| `config` | required | YAML or JSON. Format is decided by the file extension |
| `--dry-run` | off | Report what would be written without writing it |
| `--no-detector` | off | Skip `construction.hh/.cc` even if the config has a `detector` block |
 

&nbsp;
&nbsp;
&nbsp;
&nbsp;

### Config file

#### Top level
 
| Key | Required | Default | Meaning |
|---|---|---|---|
| `source` | yes | — | An `.npz` file generated from `chip_geometry.py` |
| `chip` | if the source holds several | — | chip id, e.g. `C1R1` |
| `output_dir` | no | `g4_geometry` | where the `.cc`/`.hh` go |
| `layers` | yes | — | list of `{layer, material, thickness_um?}` |
| `layer_thickness_um` | no | `0.1` | thickness for any layer not overriding it |
| `stack_order` | no | Ascending layer number | Deposition order, bottom to top. The lower layer overrides any shared footprint |
| `substrate` | no | — | `{layer, material, thickness_um}`. Excluded from overlap resolution |
| `items` | yes | — | which objects to emit, see below |
| `hole` | no | square, 5 um | `{include, shape, size_um, class_name}`. `shape` is `square` or `circle` |
| `touch_tolerance_nm` | no | `1.0` | anything closer than this is snapped into exact contact |
| `class_prefix` | no | `geometry` | prefix for generated class names |
| `warn_vertices` | no | `2000` | warn above this many vertices in one solid |
| `warn_holes` | no | `200` | warn above this many hole subtractions on one object |
| `build_options` | no | — | passed straight to `build_chip_geometry` when the source is a `.gds` |
| `detector` | no | — | detector construction settings, see below |
| `lattices` | if `detector` is set | — | per-material lattice settings, see below |
| `boundaries` | if `detector` is set | — | surface properties and border rules, see below |
 

&nbsp;
&nbsp;

#### `items`
 
Each block picks a layer and says which entries. Objects are numbered largest
first, so `L2_00000` is the biggest on that layer.
 
```yaml
items:
  - layer: 2
    entries: all                            # every object
  - layer: 3
    entries: 6                              # the 6 largest
  - layer: 2
    entries: [L2_00015, L2_00016]           # by name
  - layer: 2
    entries: [L2_00000]
    material: G4_Nb                         # override this layer's material
```
 
Inserts and holes come with the object automatically and must not be listed.
A name that doesn't exist stops the run. Duplicates across blocks are dropped
with a warning.

&nbsp;
&nbsp;

#### `detector`
 
Omit the block, or pass `--no-detector`, to emit only the component files.
 
| key | default | meaning |
|---|---|---|
| `class_name` | `MyDetectorConstruction` | generated class |
| `file_stem` | `construction` | filenames `construction.hh` / `.cc` |
| `world_material` | `G4_Galactic` | |
| `world_margin_um` | `500` | world half-size = geometry extent + margin |
| `world_half_size_um` | — | `[x, y, z]`, overrides the margin |
 
The substrate is centred on the origin and every other layer sits on its top
face.
 
&nbsp;
&nbsp;

#### `lattices`
 
Keyed by G4 material name. One entry per material used.
 
| key | required | meaning |
|---|---|---|
| `name` | yes | lattice folder passed to `LoadLattice`, e.g. `Si` |
| `miller` | no, `[1,0,0]` | passed to `SetMillerOrientation` |
| `superconductor` | no | if present, the six-argument `G4LatticePhysical` is used |
 
`superconductor` takes `elScatMFP_nm`, `delta0_eV`, `Teff_K`,
`Dn_um2_per_ns`, `tauQPTrap_ms`.
 
&nbsp;
&nbsp;

#### `boundaries`
 
```yaml
boundaries:
  scattering:                    # shared by every surface property
    anh_cutoff: 520.0
    refl_cutoff: 350.0
    anh_coeffs: [0, 0, 0, 0, 0, 1.51e-14]
    diff_coeffs: []
    spec_coeffs: []
 
  defaults:                      # matched on a material pair
    - materials: [G4_Si, G4_Al]
      name: silicon_aluminum_interface
      params: [0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,  0.0, 1.0]
 
  specific:                      # matched on a volume pair, wins over defaults
    - volumes: [C1R1_L2_00002, C1R1_L3_00017]
      name: qubit_junction_interface
      params: [...]
```
 
`params` is the ten `G4CMPSurfaceProperty` constructor arguments after the
name, passed through in order and not validated.
 
Volume names are `<chip>_<entry>`, the same names used for the generated
classes. Use `world` as one of the two names for a volume-to-world boundary.
 
Borders are worked out from the geometry, not listed: every volume borders the
world, every stacked volume borders the substrate it sits on, and two stacked
volumes border each other when their footprints touch. Pairs with no matching
rule are reported and skipped.


&nbsp;
&nbsp;
&nbsp;
&nbsp;


### npz file structure
* Each chip has one npz file, named by the column/row in your wafer.
* Each npz file contains a dictionary.
* Each entry in the dictionary is an `[N,2]` array of perimeter coordiantes of each object, in um.
* Coordinates are wrt each chip, NOT the wafer.
* Etched inserts and flux trapping holes are labeled to the object in which it is contained.
* For flux trapping holes, only the center coordiantes are saved.
* Objects are ordered by size.

| Key | Descirption |
|---|---|
| `L2_00000` | outer perimeter of the largest object on layer 2 |
| `L2_00000_insert1` | a closed inner perimeter of the object |
| `L2_00000_holes` | `(M, 2)` center coordinates of that object's flux trapping holes |
| `L1_holes` | centers of flux trapping holes on layer 1 belonging to no object |
| `_chip_center_um` | the chip center coordinates wrt the original wafer |
| `_layers` | which layers this chip holds |




