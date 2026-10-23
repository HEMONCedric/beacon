# napari-beacon-dicom

Geometry-aware DICOM image-series loading for BEACON.

## Usage

- Open **Plugins > BEACON DICOM > Open DICOM series** to scan a folder and
  select a series.
- A folder containing a single series can also be opened through napari's
  **File > Open Folder...** action with the **BEACON DICOM** reader.
- Individual `.dcm`, `.dicom`, or `.ima` files select their complete sibling
  series. Files without an extension remain available through the folder
  selector.
- The **Image file** selectors in the ARTIST and RECIST standalone apps use
  the same behavior: selecting any one slice loads the complete 3D series.

The reader groups instances by `SeriesInstanceUID`, lets GDCM sort slices in
physical order, preserves spacing/origin/direction, and uses DICOM Window
Center/Width when available. Only non-identifying acquisition metadata is
attached to the layer. DICOM SEG and RTSTRUCT are not image series and are
intentionally outside this plugin's current scope.

The layer is displayed in the acquisition's native stack orientation. This
avoids napari 0.6 treating sagittal or coronal acquisitions as thin oblique
reformats. Full patient-space spacing, origin, and direction remain attached as
DICOM metadata for downstream mask export or analysis.

## NIfTI and exported masks

The same reader also opens scalar 3D `.nii`, `.nii.gz`, `.mha`, `.mhd`, and
`.nrrd` files. It restores voxel spacing instead of relying on a generic image
reader returning only a pixel array. These volumes use the same native-axis,
local-origin display convention as DICOM; complete patient-space geometry is
retained in `beacon_volume` metadata.

Matching masks inherit the display transform of an open DICOM or medical image.
The three-plane widget also provides its original-grid display transform, so
reimport aligns with the 1 mm working image even when the source used different
viewer units or axis flips. Conflicting reference placements are not guessed.

New BEACON exports carry a segmentation marker and reopen as Labels. Older
unmarked binary files named `segmentation`, `mask`, `seg`, or `labels` also open
as Labels. Other unmarked files remain images; File > Open Labels can be used
to request a labels layer explicitly.

BEACON startup configures this reader as the default for these medical formats.
Its format-specific rules precede legacy generic rules such as `*.gz`, which
napari 0.6 otherwise treats as equally specific to `*.nii.gz`. Other compressed
formats retain their existing reader preference.
The NIfTI itself retains its original patient-space header for use outside napari.
