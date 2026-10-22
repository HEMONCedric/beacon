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
