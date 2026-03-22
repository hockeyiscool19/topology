> Begin script
> Prompt enter state name or abbreviation
> Prompt for dimensions of material (e.g., 12 inches wide)
> Get shape of a state through geopandas
> Save to SVG file with topology as color compatible with easel

Path Closing: Ensure the state border is a single, continuous closed loop. If there are gaps, the CNC won't know where the "inside" and "outside" are for tool compensation.

Scaling: GIS data uses latitude/longitude or meters. You will need to scale the SVG to your actual material size (e.g., 12 inches wide).

Smoothing: Apply convolution to smooth out any jagged edges or sharp corners.
