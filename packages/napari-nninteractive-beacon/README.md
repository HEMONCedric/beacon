# napari-nninteractive-beacon

BEACON wrapper around `napari-nninteractive==2.5.3` adding a one-way final manual-refinement phase before **Next Object**.

The package contributes two widgets: **nnInteractive (BEACON Classic)** and
**nnInteractive (BEACON 3-plane Lasso)**.

## Three-plane lasso workflow

1. Browse to a 3D medical image (a `.dcm` loads its series), enter a file path,
   or select an Image layer and click **Use active image layer**.
2. The plugin creates a working volume at **1 × 1 × 1 mm** using linear
   interpolation, preserving the acquisition's origin and direction. No canonical
   reorientation or de-obliquing is applied. The source array is retained.
   A partial final millimetre may be padded; anatomy is never stretched to fit
   rounded array dimensions. Active DICOM layers use their scanner metadata;
   unmodified NIfTI/MetaImage/NRRD layers can recover geometry from their source
   file. Other scalar 3D layers use the layer transform, interpreted in mm.
   The working image remains aligned with the active source layer's display
   frame, including its translation and axis flips. Scanner coordinates are
   stored separately for export. Plane button names follow the acquisition's
   orientation; oblique acquisitions retain their native planes.
3. Configure **Local / Remote** in the **Settings** tab and click
   **Initialize**.
4. Draw one closed positive lasso around the same lesion in **Axial**,
   **Coronal** and **Sagittal**. The plugin advances to the next missing plane.
   Redrawing a plane replaces its mask; the other two remain available.
   Previously drawn masks appear in bright red with thin, single-slice traces
   in the other planes. Display does not expand the masks into neighbouring
   slices; inference receives the original masks with unchanged footprints.
5. Click **Generate segmentation**. All three cropped 2D masks are accumulated
   before the single prediction. The controls are locked while it runs.
6. **Undo** returns from the prediction to the three lassos. Before prediction,
   Undo removes the most recently drawn mask. **Refine Manually** enables
   Paint/Erase; Undo then uses the label layer's manual edit history.
7. **Export mask (original grid)** resamples labels with nearest-neighbour
   interpolation to the exact original size, spacing, origin and direction.
   Reimport through BEACON's medical-volume reader restores spacing and aligns
   the mask with the source/working image. Exported masks reopen as Labels.

This widget has no Next Object, negative/correction prompts, point, bounding-box,
scribble, mask-initialization or object-aggregation workflow. It retains Classic's
local and remote inference configuration, including optional SSH forwarding.

Remote mode also offers an optional **SSH target** field. For a localhost server
URL, Connect or Initialize opens a loopback-only SSH forward automatically;
leave the field blank for an existing local server or use a direct server URL
without a tunnel. SSH uses your existing non-interactive SSH authentication.
The nnInteractive API-key field is separate and may be left empty when the
server does not require one. No Auto-Prompt package is needed.
