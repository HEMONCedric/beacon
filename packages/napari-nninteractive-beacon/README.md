# napari-nninteractive-beacon

BEACON wrapper around `napari-nninteractive==2.5.3` adding a one-way final manual-refinement phase before **Next Object**.

Remote mode also offers an optional **SSH target** field. For a localhost server
URL, Connect or Initialize opens a loopback-only SSH forward automatically;
leave the field blank for an existing local server or use a direct server URL
without a tunnel. SSH uses your existing non-interactive SSH authentication.
The nnInteractive API-key field is separate and may be left empty when the
server does not require one. No Auto-Prompt package is needed.
