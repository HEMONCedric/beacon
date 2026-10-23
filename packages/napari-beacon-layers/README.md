# napari-beacon-layers

A collection of custom napari layers for specialized visualization and interaction within the napari viewer.

Includes:
- `ManualPointsLayer`: editable points layer with undo/redo keyboard shortcuts (`Ctrl+Z` undo, `Ctrl+Shift+Z` redo).

## Image window presets

All Image layers in BEACON (including DICOM, imported volumes and the 1 mm working
image) have a **window preset** dropdown immediately below **contrast limits**.
The same control replaces the old CT-only dropdown on `FixedImageLayer`.
Native contrast, interpolation, gamma and colormap controls remain available.

- **CT**: abdomen, bone, brain, lung, mediastinum, soft tissue, liver and extended
  dental/inner-ear/larynx/pelvis/spine/vertebrae windows. CT input must already be
  in HU, as with BEACON's SimpleITK DICOM reader. The core abdomen/bone/brain/lung
  values follow [3D Slicer's slice-window presets](https://github.com/Slicer/Slicer/blob/main/Modules/Loadable/Volumes/Resources/VolumeDisplayPresets.json);
  the liver window follows [OHIF](https://docs.ohif.org/coverage/extensions/cornerstone/src/components/windowlevelactionmenu/defaultwindowlevelpresets.ts).
- **MR**: standard (1–99%), high contrast (5–95%) and wide (0.1–99.9%) windows
  computed from finite, non-zero values in a bounded spatial sample. These are
  relative display aids, not fixed or sequence-specific T1/T2 presets. Intensities
  are neither normalized nor modified.
- **Auto**: 0.5–99.5% of a bounded sample, including zero background.
- **Original** restores the window present when the layer controls were created;
  **Manual** keeps the current window for normal slider/mouse adjustment. Manual
  adjustments automatically change the selected preset to Manual. Each layer
  keeps its own selection while switching between layers.

Presets only change display contrast limits. They do not alter voxel values,
spacing, orientation, prompts or exported masks. Applying a preset stops
continuous auto-contrast so scrolling does not overwrite the selected window.
Large/lazy volumes are sampled before conversion (at most 262,144 voxels), and
RGB image controls disable the scalar-window dropdown.
