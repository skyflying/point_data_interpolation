# Point Data Interpolation
GUI Python for point cloud gridding and GeoTIFF generation, with optional boundary shapefile export.

Supports .xyz, .csv, .txt, and .pts point files containing X/Y/Z data.

---

## ✨ Features

- **Multiple input formats**: `.xyz`, `.csv`, `.txt`, `.pts` (auto delimiter detection).
- **Flexible CRS handling**:
  - Specify input EPSG code.
  - Specify output EPSG (defaults to input).
  - Automatically converts 1 m base resolution to degrees for geographic CRS (e.g., EPSG:4326).
- **Interpolation algorithms**:
  - **Nearest neighbor** — configurable *k*, search radius, and aggregation (`mean`, `nearest`, `median`).
  - **Inverse Distance Weighting (IDW)** — configurable *k*, search radius, and power.
  - **TIN** — triangulated irregular network via `scipy.LinearNDInterpolator`.
  - **Ordinary Kriging** (optional, requires `pykrige`).
- **Parameter auto-enable** — only relevant parameters are active for the chosen algorithm.
- **Outputs**:
  - GeoTIFF raster.
  - Optional polygon boundary as Shapefile (`footprint` or `convex hull`), with buffer/simplify options.
- **Batch processing**:
  - Add multiple files at once or scan entire folders.
  - Output file names are based on input names.
- **Preview resolution in degrees** when output CRS is geographic.
- **User-friendly PyQt6 GUI** — no coding required for basic use.

---

## 📦 Dependencies

Install Python 3.9+ and required packages:

```bash
pip install numpy pandas scipy rasterio shapely fiona pyproj PyQt6
# Optional: for Kriging
pip install pykrige
```

---

## 🚀 Usage

Run the application:
```bash
python pointgrid_gui_pyqt6.py
```
- **Add one or more point files (.xyz, .csv, .txt, .pts).**

- **Set Input EPSG and optionally Output EPSG.**

- **Choose grid resolution (meters; converted to degrees automatically if needed).**

- **Select interpolation algorithm and configure parameters.**

- **Choose whether to export GeoTIFF, boundary Shapefile, or both.**

- **Start processing.**

---

## 🗺 Output Example

- **data.tif — interpolated raster in the selected output CRS.**

- **data_boundary.shp — optional polygon boundary.**
