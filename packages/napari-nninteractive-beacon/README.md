# BEACON

BEACON wrapper around `napari-nninteractive==2.5.3` adding a one-way final manual-refinement phase before **Next Object**.

Open **Plugins → BEACON** and choose **BEACON Classic** or **BEACON 3 Lasso**.
Select **nnInteractive** or **IMPACT-OncoRep** under **Settings → Model**.
Both run locally. nnInteractive also supports the existing remote server and
SSH forwarding. See the [installation commands](../../README.md#local-installation).

OncoRep applies its released body crop, resampling (1.5 × 1.5 × 2 mm in xyz),
and modality-specific normalization to the native image. The same image is
preprocessed once per initialization. The model reuses the encoded patch for
corrections within that patch. **Auto** uses available DICOM modality metadata
or recognizes typical CT in HU. For a scan with ambiguous metadata, select
**CT**, **MR** or **PET (SUV)**. Classic uses positive and negative points and
scribbles, positive boxes and lassos, and a previous mask as a refinement seed.

Classic exports 3D object, instance and semantic masks as NIfTI on the source
image grid. The 3 Lasso export includes the displayed mask with any manual edits
and dilation.

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
   before one prediction. OncoRep receives the triplet on its own model planes.
   For oblique acquisitions, each outline is projected onto the closest
   corresponding model plane. The model runs in a worker thread.
6. After prediction, choose **Add Lasso**, **Dilate**, **Refine Manually** or
   export. **Add Lasso** enables the plane buttons: choose any plane and slice,
   then draw another positive lasso. Prediction runs automatically when drawing
   finishes. Additional lassos accumulate, even in the same plane; they do not
   replace the original three. Click **Add Lasso** again (or Undo) to cancel an
   unfinished addition without changing the mask.
   **Dilate** expands the binary mask by a spherical **2 mm radius per click**,
   measured using the physical working spacing, not the viewer's display units.
   Dilation runs locally, including when inference uses a remote server.
   **Undo** first restores the mask before the last dilation/additional lasso;
   after those edits are undone, it returns to the original three-lasso stage.
   Before prediction, Undo removes the most recently drawn mask. Snapshots are
   compressed and bounded to 20 entries / 128 MiB (at least the newest retained).
   **Refine Manually** remains the final Paint/Erase phase: Add Lasso is then
   disabled, but Dilate remains available and shares the native paint Undo history.
7. **Export mask (original grid)** resamples labels with nearest-neighbour
   interpolation to the exact original size, spacing, origin and direction.
   Reimport through BEACON's medical-volume reader restores spacing and aligns
   the mask with the source/working image. Exported masks reopen as Labels.

This widget has no Next Object, negative/correction prompts, point, bounding-box,
scribble, mask-initialization or object-aggregation workflow. It retains Classic's
local and remote inference configuration, including optional SSH forwarding.

Additional AI refinement seeds nnInteractive from the currently displayed mask,
then replays the initial and additional lassos before a single prediction. This
retains any prior dilation and makes refinement safe after client-side Undo or
reconnect, without uploading the image again. It requires a checkpoint supporting
initial-label refinement; the Add Lasso button is disabled otherwise. Only binary
masks (labels 0/1) are supported by these two actions. Failed refinements leave
the displayed mask/history untouched. Dilation and refinement preserve the
existing export-to-original-grid behavior.

Remote mode also offers an optional **SSH target** field. For a localhost server
URL, Connect or Initialize opens a loopback-only SSH forward automatically;
leave the field blank for an existing local server or use a direct server URL
without a tunnel. SSH uses your existing non-interactive SSH authentication.
The nnInteractive API-key field is separate and may be left empty when the
server does not require one. No Auto-Prompt package is needed.

## Native-volume remote upload (three-plane widget)

When the server supports the BEACON extension, the widget sends a losslessly
compressed snapshot of the selected **native image**, retaining its dtype
(e.g. int16 CT), rather than its interpolated float32 working volume. The source
array order and patient-space geometry are sent separately from the **exact**
1 mm target grid. The server uses the same SimpleITK linear interpolation,
nearest extrapolation and float32 output as the workstation, without recalculating
target dimensions. Display remains resampled locally; inference, lasso coordinates,
refinement and export all keep their existing grids.

The response grid, shape and sampled intensities are checked before prompts are
enabled. Mismatches stop initialization rather than accepting shifted masks.
Standard servers remain compatible: a missing extension falls back to uploading
the working image. If the working image is already smaller in raw bytes (e.g.
a high-resolution source downsampled to 1 mm), it is sent directly instead.
The native snapshot consumes additional workstation RAM while the image is open.

The upload dialog identifies native compression and server reconstruction; the
status shows compressed size, compression time, request time and server resampling
time. Request time includes network, decompression and server initialization: it
is **not** a measurement of network-only throughput. No image values are logged.

### Extended server container

From this package directory, build and launch a separate instance:

```bash
docker build -f Dockerfile.server -t beacon-nninteractive-native .
docker run -d --name nninteractive-beacon-native --gpus 'device=0' \
  -p 127.0.0.1:1528:1527 beacon-nninteractive-native
```

This reuses the official server/model image and adds only the GUI-free extension.
The original endpoints, API-key authentication (optional), session leases,
heartbeats and per-session locks are retained. Set `NN_INTERACTIVE_API_KEY` on the
container if desired. Keep the published port loopback-only and access it by SSH.
No modifications to installed nnInteractive source files are required.

In the BEACON environment, with an SSH tunnel to the server:

```bash
python scripts/check_native_upload.py --url http://127.0.0.1:1528 --predict
python scripts/check_native_upload.py --url http://127.0.0.1:1528 --large
```

These checks use synthetic images only. They compare full working-volume SHA-256
digests across workstation/server, optionally test three real GPU lassos and
verify export geometry. Test leases are always released. The larger check also
compares compressed native and interpolated sizes; actual savings depend on the
image, original spacing, dtype and network connection.
