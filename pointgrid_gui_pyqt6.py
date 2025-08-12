#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point Grid Interpolation GUI (PyQt6)
- Inputs: .xyz/.csv/.txt/.pts containing x y z (with or without header)
- CRS: choose input EPSG and (optional) output EPSG (defaults to input)
- Resolution: base = 1 meter; if output CRS is geographic (degrees), auto-convert to Δlon/Δlat (degrees) at dataset center via pyproj.Geod
- Algorithms:
    * Nearest: k / radius / aggregation(mean|nearest|median)
    * IDW: k / radius / power
    * TIN: scipy LinearNDInterpolator
    * Kriging: pykrige (optional)
- Outputs: GeoTIFF (optional), boundary Shapefile (optional)
- Batch: multiple files or folder scan; outputs named <stem>.tif and <stem>_boundary.shp

Author: 2025
License: MIT
"""

from __future__ import annotations

import sys
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Literal, Callable, Union

import numpy as np
import pandas as pd

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QThread, pyqtSignal

# Geo & raster deps
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS as RioCRS
from rasterio.features import shapes as rio_shapes

from shapely.geometry import shape as shp_shape, mapping as shp_mapping
from shapely.geometry import MultiPolygon, Polygon, MultiPoint
from shapely.ops import unary_union

import fiona
from pyproj import CRS, Transformer, Geod

from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator

# Optional Kriging
try:
    from pykrige.ok import OrdinaryKriging
    KRIGING_AVAILABLE = True
except Exception:
    KRIGING_AVAILABLE = False

# -----------------------------
# Configuration dataclasses
# -----------------------------
Algorithm = Literal["nearest", "idw", "tin", "kriging"]
BoundaryMode = Literal["footprint", "convex"]
NearestAgg = Literal["mean", "nearest", "median"]

@dataclass
class InterpParams:
    algorithm: Algorithm = "nearest"
    grid_resolution_base_m: float = 1.0
    search_k: int = 12
    search_radius: Optional[float] = None
    idw_power: float = 1.0
    tin_max_triangle_area: Optional[float] = None
    kriging_variogram: Literal["spherical", "exponential", "gaussian"] = "spherical"
    kriging_n_lags: int = 6
    nearest_agg: NearestAgg = "mean"  # 新增：Nearest 的彙整策略

@dataclass
class CRSParams:
    input_epsg: Optional[int] = None
    output_epsg: Optional[int] = None  # if None -> same as input

@dataclass
class OutputParams:
    write_geotiff: bool = True
    export_boundary: bool = False
    boundary_mode: BoundaryMode = "footprint"
    boundary_simplify: float = 0.0
    boundary_buffer: float = 0.0
    nodata: float = -9999.0
    dtype: str = "float32"
    out_dir: Optional[Path] = None
    overwrite: bool = False

@dataclass
class JobConfig:
    files: List[Path]
    interp: InterpParams
    crs: CRSParams
    out: OutputParams
    stop_on_error: bool = False


# -----------------------------
# Utility: reading points
# -----------------------------
def read_xyz_like(path: Path) -> np.ndarray:
    """
    Load x y z from .xyz/.csv/.txt/.pts.
    - Detect delimiter among [space, comma, tab]
    - Header optional; keep first 3 numeric columns as x,y,z
    Returns: np.ndarray shape (N,3)
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    # Guess delimiter
    sample = path.read_text(errors="ignore").splitlines()[:5]
    text = "\n".join(sample)
    if "," in text:
        sep = ","
    elif "\t" in text:
        sep = "\t"
    else:
        sep = r"\s+"

    df = pd.read_csv(path, sep=sep, engine="python", header=None, comment="#")
    if df.shape[1] < 3:
        df2 = pd.read_csv(path, sep=sep, engine="python", header=0, comment="#")
        if df2.shape[1] < 3:
            raise ValueError(f"{path.name}: need at least 3 columns (x y z).")
        df = df2

    # take first 3 numeric columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 3:
        df = df.apply(pd.to_numeric, errors="coerce")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) < 3:
            raise ValueError(f"{path.name}: cannot parse numeric x y z.")

    arr = df[num_cols[:3]].to_numpy(dtype=float)
    arr = arr[~np.isnan(arr).any(axis=1)]
    if arr.size == 0:
        raise ValueError(f"{path.name}: empty or invalid data.")
    return arr  # (N,3)


# -----------------------------
# CRS & resolution helpers
# -----------------------------
def is_geographic(epsg: int) -> bool:
    try:
        return CRS.from_epsg(epsg).is_geographic
    except Exception:
        return False

def auto_center_lonlat(points_xy: np.ndarray, input_epsg: int, out_epsg: int) -> Tuple[float, float]:
    """Return center in lon/lat for resolution conversion (always in EPSG:4326)."""
    x = points_xy[:, 0]; y = points_xy[:, 1]
    cx = float((x.min() + x.max()) / 2.0)
    cy = float((y.min() + y.max()) / 2.0)
    if input_epsg == 4326 and out_epsg == 4326:
        lon, lat = cx, cy
    else:
        t = Transformer.from_crs(CRS.from_epsg(input_epsg), CRS.from_epsg(4326), always_xy=True)
        lon, lat = t.transform(cx, cy)
    return lon, lat

def degree_resolution_for_1m(center_lon: float, center_lat: float) -> Tuple[float, float]:
    """Compute Δlon, Δlat in degrees that approximately represent 1 meter at center point."""
    geod = Geod(ellps="WGS84")
    lon_e, lat_e, _ = geod.fwd(center_lon, center_lat, 90, 1.0)  # East 1 m
    lon_n, lat_n, _ = geod.fwd(center_lon, center_lat, 0, 1.0)   # North 1 m
    dlon = abs(lon_e - center_lon)
    dlat = abs(lat_n - center_lat)
    return dlon, dlat

def transform_points(points: np.ndarray, epsg_in: int, epsg_out: int) -> np.ndarray:
    if epsg_in == epsg_out:
        return points[:, :2].copy()
    t = Transformer.from_crs(CRS.from_epsg(epsg_in), CRS.from_epsg(epsg_out), always_xy=True)
    x, y = t.transform(points[:, 0], points[:, 1])
    return np.column_stack([x, y])

# -----------------------------
# Grid builder
# -----------------------------
def build_grid(points_xy_out: np.ndarray,
               out_epsg: int,
               base_res_m: float,
               nodata: float,
               degree_hint_center_lonlat: Optional[Tuple[float, float]] = None
               ) -> Tuple[np.ndarray, np.ndarray, Affine, Tuple[float,float,float,float], float, float]:
    """
    Construct regular grid covering points.
    - If output CRS is geographic, convert base_res_m to degrees at center
    - Returns grid_x, grid_y, affine, bounds (minx,miny,maxx,maxy), res_x, res_y
    """
    minx = float(points_xy_out[:, 0].min())
    maxx = float(points_xy_out[:, 0].max())
    miny = float(points_xy_out[:, 1].min())
    maxy = float(points_xy_out[:, 1].max())

    if is_geographic(out_epsg):
        if degree_hint_center_lonlat is None:
            raise ValueError("Geographic CRS requires center lon/lat for degree resolution.")
        dlon, dlat = degree_resolution_for_1m(*degree_hint_center_lonlat)
        res_x, res_y = dlon * (base_res_m / 1.0), dlat * (base_res_m / 1.0)
    else:
        res_x = res_y = base_res_m

    nx = max(1, int(math.ceil((maxx - minx) / res_x)))
    ny = max(1, int(math.ceil((maxy - miny) / res_y)))

    transform = Affine.translation(minx, maxy) * Affine.scale(res_x, -res_y)
    xs = minx + (np.arange(nx) + 0.5) * res_x
    ys = maxy - (np.arange(ny) + 0.5) * res_y
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x, grid_y, transform, (minx, miny, maxx, maxy), float(res_x), float(res_y)

# -----------------------------
# Interpolators
# -----------------------------
def interpolate_nearest_or_idw(points_xy: np.ndarray, values: np.ndarray,
                               grid_x: np.ndarray, grid_y: np.ndarray,
                               k: int = 12, radius: Optional[float] = None,
                               power: Optional[float] = None,
                               aggregation: NearestAgg = "mean",
                               nodata: float = -9999.0) -> np.ndarray:
    """
    Unified interpolator:
      - power is None -> Nearest with (k, radius, aggregation)
      - power is not None -> IDW with (k, radius, power)
    """
    tree = cKDTree(points_xy)
    q = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    if power is None:
        # ---------- Nearest with k / radius / aggregation ----------
        kk = max(1, k if k is not None else 1)
        out = np.full(q.shape[0], nodata, dtype=float)

        if radius is not None and radius > 0:
            idxs = tree.query_ball_point(q, r=radius)
            for i, neigh in enumerate(idxs):
                if not neigh:
                    continue
                # keep only k nearest inside radius if too many
                if kk and len(neigh) > kk:
                    dists = np.linalg.norm(points_xy[neigh] - q[i], axis=1)
                    sel = np.argsort(dists)[:kk]
                    neigh = [neigh[j] for j in sel]
                    dists = dists[sel]
                else:
                    dists = np.linalg.norm(points_xy[neigh] - q[i], axis=1)

                vals = values[neigh]
                if vals.size == 0:
                    continue

                if aggregation == "nearest":
                    out[i] = float(vals[np.argmin(dists)])
                elif aggregation == "median":
                    out[i] = float(np.median(vals))
                else:  # "mean"
                    out[i] = float(np.mean(vals))
            return out.reshape(grid_x.shape)

        # no radius: pure k-NN
        kk = max(1, kk)
        dist, idx = tree.query(q, k=kk)
        if kk == 1:
            out[:] = values[idx].astype(float)
            return out.reshape(grid_x.shape)
        vals = values[idx]  # (Nq, k)
        if aggregation == "nearest":
            out[:] = vals[:, 0].astype(float)  # the first is the nearest
        elif aggregation == "median":
            out[:] = np.median(vals, axis=1)
        else:
            out[:] = np.mean(vals, axis=1)
        return out.reshape(grid_x.shape)

    # ---------- IDW ----------
    if radius is not None and radius > 0:
        idxs = tree.query_ball_point(q, r=radius)
        out = np.full(q.shape[0], nodata, dtype=float)
        for i, neigh in enumerate(idxs):
            if not neigh:
                continue
            if k and len(neigh) > k:
                dists = np.linalg.norm(points_xy[neigh] - q[i], axis=1)
                sel = np.argsort(dists)[:k]
                neigh = [neigh[j] for j in sel]
                dists = dists[sel]
            else:
                dists = np.linalg.norm(points_xy[neigh] - q[i], axis=1)
            w = 1.0 / np.maximum(dists, 1e-12)**power
            out[i] = float(np.sum(w * values[neigh]) / np.sum(w))
        return out.reshape(grid_x.shape)

    kk = max(1, k)
    dist, idx = tree.query(q, k=kk)
    if kk == 1:
        w = 1.0 / np.maximum(dist, 1e-12)**power
        z = values[idx]
        arr = (w * z) / w
        return arr.reshape(grid_x.shape)
    w = 1.0 / np.maximum(dist, 1e-12)**power
    num = np.sum(w * values[idx], axis=1)
    den = np.sum(w, axis=1)
    out = num / den
    return out.reshape(grid_x.shape)

def interpolate_tin(points_xy: np.ndarray, values: np.ndarray,
                    grid_x: np.ndarray, grid_y: np.ndarray,
                    max_triangle_area: Optional[float],
                    nodata: float) -> np.ndarray:
    try:
        interp = LinearNDInterpolator(points_xy, values, fill_value=nodata, rescale=False)
    except Exception as e:
        raise RuntimeError(f"TIN interpolation failed: {e}") from e
    zi = interp(grid_x, grid_y)
    return zi.astype(float)

def interpolate_kriging(points_xy: np.ndarray, values: np.ndarray,
                        grid_x: np.ndarray, grid_y: np.ndarray,
                        variogram_model: str, n_lags: int,
                        nodata: float) -> np.ndarray:
    if not KRIGING_AVAILABLE:
        raise RuntimeError("pykrige is not installed.")
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    ok = OrdinaryKriging(
        x, y, values,
        variogram_model=variogram_model,
        nlags=n_lags,
        enable_plotting=False,
        verbose=False,
    )
    zi, ss = ok.execute("grid", grid_x[0], grid_y[:, 0])
    out = np.array(zi, dtype=float)
    out[~np.isfinite(out)] = nodata
    return out

# -----------------------------
# Raster writer
# -----------------------------
def write_geotiff(path: Path, array: np.ndarray, transform: Affine, epsg: int,
                  nodata: float, dtype: str = "float32", overwrite: bool = False) -> Path:
    if path.exists() and not overwrite:
        base = path.with_suffix("")
        i = 1
        newp = base.with_name(base.name + f"_{i}").with_suffix(".tif")
        while newp.exists():
            i += 1
            newp = base.with_name(base.name + f"_{i}").with_suffix(".tif")
        path = newp

    dtype_np = np.dtype(dtype)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype_np.name,
        "crs": RioCRS.from_epsg(epsg),
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.asarray(array, dtype=dtype_np), 1)
    return path

# -----------------------------
# Boundary extractor & writer
# -----------------------------
def extract_boundary(array: np.ndarray, transform: Affine, mode: BoundaryMode,
                     nodata: float) -> Union[Polygon, MultiPolygon]:
    valid_mask = np.isfinite(array) & (array != nodata)
    if not valid_mask.any():
        raise ValueError("No valid pixels for boundary extraction.")
    if mode == "footprint":
        geoms = []
        for geom, val in rio_shapes(valid_mask.astype(np.uint8), mask=None, transform=transform):
            if val == 1:
                geoms.append(shp_shape(geom))
        if not geoms:
            raise ValueError("No valid polygons.")
        poly = unary_union(geoms)
        return poly

    # convex
    ys, xs = np.where(valid_mask)
    xs = xs + 0.5
    ys = ys + 0.5
    xw = transform.c + xs * transform.a + ys * transform.b
    yw = transform.f + xs * transform.d + ys * transform.e
    mp = MultiPoint(list(zip(xw, yw)))
    hull = mp.convex_hull
    if isinstance(hull, (Polygon, MultiPolygon)):
        return hull
    raise ValueError("Convex hull failed.")

def write_shapefile(path: Path, geom: Union[Polygon, MultiPolygon], epsg: int,
                    overwrite: bool = False) -> Path:
    if path.exists():
        if overwrite:
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                try:
                    path.with_suffix(ext).unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            base = path.with_suffix("")
            i = 1
            newp = base.with_name(base.name + f"_{i}").with_suffix(".shp")
            while newp.exists():
                i += 1
                newp = base.with_name(base.name + f"_{i}").with_suffix(".shp")
            path = newp

    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    if isinstance(geom, Polygon):
        geoms = [geom]
    elif isinstance(geom, MultiPolygon):
        geoms = list(geom.geoms)
    else:
        raise ValueError("Geometry must be Polygon/MultiPolygon.")

    crs_wkt = CRS.from_epsg(epsg).to_wkt()

    with fiona.open(
        path, "w",
        driver="ESRI Shapefile",
        schema=schema,
        crs_wkt=crs_wkt,      # 使用 crs_wkt（Fiona 1.9+ 推薦）
        encoding="utf-8"
    ) as dst:
        for i, g in enumerate(geoms, start=1):
            dst.write({"geometry": shp_mapping(g), "properties": {"id": i}})

    # 確保 .prj 存在（多數情況 Fiona 已產生）
    prj = path.with_suffix(".prj")
    if not prj.exists():
        prj.write_text(crs_wkt, encoding="utf-8")
    return path

# -----------------------------
# Process one file
# -----------------------------
def process_one_file(
    file_path: Path,
    interp: InterpParams,
    crs_params: CRSParams,
    out_params: OutputParams,
    logger: Callable[[str], None],
) -> Tuple[Optional[Path], Optional[Path]]:
    logger(f"[START] {file_path.name}")
    pts = read_xyz_like(file_path)
    if crs_params.input_epsg is None:
        raise ValueError("Input EPSG is required.")
    epsg_in = int(crs_params.input_epsg)
    epsg_out = int(crs_params.output_epsg or epsg_in)

    # Transform to output CRS for gridding
    pts_xy_out = transform_points(pts, epsg_in, epsg_out)
    z = pts[:, 2].astype(float)

    # Degree resolution conversion if needed
    center_hint = None
    if is_geographic(epsg_out):
        lon, lat = auto_center_lonlat(pts[:, :2], epsg_in, epsg_out)
        center_hint = (lon, lat)

    grid_x, grid_y, transform, bounds, res_x, res_y = build_grid(
        pts_xy_out, epsg_out, interp.grid_resolution_base_m, out_params.nodata, center_hint
    )

    # Interpolate
    algo = interp.algorithm
    logger(f"  - Algorithm: {algo}")
    if algo == "nearest":
        arr = interpolate_nearest_or_idw(
            pts_xy_out, z, grid_x, grid_y,
            k=interp.search_k,
            radius=interp.search_radius,
            power=None,
            aggregation=interp.nearest_agg,
            nodata=out_params.nodata
        )
    elif algo == "idw":
        arr = interpolate_nearest_or_idw(
            pts_xy_out, z, grid_x, grid_y,
            k=interp.search_k, radius=interp.search_radius,
            power=interp.idw_power,
            aggregation="mean",  # not used for IDW
            nodata=out_params.nodata
        )
    elif algo == "tin":
        arr = interpolate_tin(
            pts_xy_out, z, grid_x, grid_y,
            max_triangle_area=interp.tin_max_triangle_area,
            nodata=out_params.nodata
        )
    elif algo == "kriging":
        if not KRIGING_AVAILABLE:
            logger("  ! pykrige not installed, falling back to nearest.")
            arr = interpolate_nearest_or_idw(
                pts_xy_out, z, grid_x, grid_y, k=1, power=None, aggregation="nearest", nodata=out_params.nodata
            )
        else:
            arr = interpolate_kriging(
                pts_xy_out, z, grid_x, grid_y,
                variogram_model=interp.kriging_variogram,
                n_lags=interp.kriging_n_lags, nodata=out_params.nodata
            )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Output directory
    out_dir = out_params.out_dir if out_params.out_dir else file_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = file_path.stem

    tif_path = None
    shp_path = None

    if out_params.write_geotiff:
        tif_path = out_dir / f"{stem}.tif"
        tif_path = write_geotiff(
            tif_path, arr, transform, epsg_out, nodata=out_params.nodata,
            dtype=out_params.dtype, overwrite=out_params.overwrite
        )
        logger(f"  - Wrote GeoTIFF: {tif_path.name}")

    if out_params.export_boundary:
        geom = extract_boundary(arr, transform, mode=out_params.boundary_mode, nodata=out_params.nodata)
        if abs(out_params.boundary_buffer) > 0:
            geom = geom.buffer(out_params.boundary_buffer)
        if out_params.boundary_simplify > 0:
            geom = geom.simplify(out_params.boundary_simplify, preserve_topology=True)
        shp_path = out_dir / f"{stem}_boundary.shp"
        shp_path = write_shapefile(shp_path, geom, epsg_out, overwrite=out_params.overwrite)
        logger(f"  - Wrote Shapefile: {shp_path.name}")

    logger(f"[DONE ] {file_path.name}")
    return tif_path, shp_path


# -----------------------------
# Qt Worker (thread)
# -----------------------------
class InterpWorker(QtCore.QObject):
    progress = pyqtSignal(int, str)          # percent, message
    fileProgress = pyqtSignal(int, int, str) # current_idx, total, filename
    finished = pyqtSignal(int, int)          # success_count, fail_count
    message = pyqtSignal(str)

    def __init__(self, job: JobConfig):
        super().__init__()
        self.job = job
        self._abort = False

    def stop(self):
        self._abort = True

    def log(self, msg: str):
        self.message.emit(msg)

    @QtCore.pyqtSlot()
    def run(self):
        files = self.job.files
        total = len(files)
        succ = 0
        fail = 0
        for i, f in enumerate(files, start=1):
            if self._abort:
                break
            self.fileProgress.emit(i, total, f.name)
            try:
                process_one_file(f, self.job.interp, self.job.crs, self.job.out, self.log)
                succ += 1
            except Exception as e:
                fail += 1
                tb = traceback.format_exc(limit=2)
                self.log(f"[ERROR] {f.name}: {e}\n{tb}")
                if self.job.stop_on_error:
                    break
            pct = int(i / total * 100)
            self.progress.emit(pct, f"{i}/{total}")
        self.finished.emit(succ, fail)


# -----------------------------
# Qt GUI
# -----------------------------
FILE_FILTER = "Point files (*.xyz *.csv *.txt *.pts);;All files (*.*)"

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Grid Interpolation (PyQt6)")
        self.setMinimumSize(1000, 740)
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[InterpWorker] = None

        self.files: List[Path] = []

        self._build_ui()
        self._wire_events()
        self._refresh_kriging_available()
        self._update_degree_preview()
        self._update_param_enablements()

    # ---------- UI ----------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)

        # Top: file inputs
        files_group = QtWidgets.QGroupBox("Input files")
        v.addWidget(files_group)
        fg = QtWidgets.QGridLayout(files_group)

        self.btn_add_files = QtWidgets.QPushButton("Add Files…")
        self.btn_add_folder = QtWidgets.QPushButton("Add Folder…")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.list_files = QtWidgets.QListWidget()
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        fg.addWidget(self.btn_add_files, 0, 0)
        fg.addWidget(self.btn_add_folder, 0, 1)
        fg.addWidget(self.btn_clear, 0, 2)
        fg.addWidget(self.list_files, 1, 0, 1, 3)

        # CRS group
        crs_group = QtWidgets.QGroupBox("CRS / Resolution")
        v.addWidget(crs_group)
        cg = QtWidgets.QGridLayout(crs_group)

        self.input_epsg_edit = QtWidgets.QLineEdit()
        self.input_epsg_edit.setPlaceholderText("e.g., 3826 / 4326")
        self.output_epsg_edit = QtWidgets.QLineEdit()
        self.output_epsg_edit.setPlaceholderText("Optional; default=Input EPSG")

        self.resolution_spin = QtWidgets.QDoubleSpinBox()
        self.resolution_spin.setDecimals(3)
        self.resolution_spin.setRange(0.001, 1_000_000)
        self.resolution_spin.setValue(1.0)  # base meters

        self.lbl_out_unit = QtWidgets.QLabel("Output unit: (auto)")
        self.lbl_deg_preview = QtWidgets.QLabel("~ Δlon=?, Δlat=? (deg)")

        cg.addWidget(QtWidgets.QLabel("Input EPSG:"), 0, 0)
        cg.addWidget(self.input_epsg_edit, 0, 1)
        cg.addWidget(QtWidgets.QLabel("Output EPSG:"), 0, 2)
        cg.addWidget(self.output_epsg_edit, 0, 3)
        cg.addWidget(QtWidgets.QLabel("Resolution base (meters):"), 1, 0)
        cg.addWidget(self.resolution_spin, 1, 1)
        cg.addWidget(self.lbl_out_unit, 1, 2)
        cg.addWidget(self.lbl_deg_preview, 1, 3)

        # Algorithm group
        algo_group = QtWidgets.QGroupBox("Algorithm & Parameters")
        v.addWidget(algo_group)
        ag = QtWidgets.QGridLayout(algo_group)

        self.cb_algorithm = QtWidgets.QComboBox()
        self.cb_algorithm.addItems(["nearest", "idw", "tin", "kriging"])
        self.spin_k = QtWidgets.QSpinBox(); self.spin_k.setRange(1, 1000); self.spin_k.setValue(12)
        self.spin_radius = QtWidgets.QDoubleSpinBox(); self.spin_radius.setRange(0.0, 1e9); self.spin_radius.setDecimals(3)
        self.spin_idw_power = QtWidgets.QDoubleSpinBox(); self.spin_idw_power.setRange(0.01, 10.0); self.spin_idw_power.setDecimals(2); self.spin_idw_power.setValue(1.0)
        self.spin_tin_max_area = QtWidgets.QDoubleSpinBox(); self.spin_tin_max_area.setRange(0.0, 1e12); self.spin_tin_max_area.setDecimals(2)
        self.cb_variogram = QtWidgets.QComboBox(); self.cb_variogram.addItems(["spherical", "exponential", "gaussian"])
        self.spin_nlags = QtWidgets.QSpinBox(); self.spin_nlags.setRange(2, 50); self.spin_nlags.setValue(6)

        # 新增：Nearest aggregation
        self.cb_nearest_agg = QtWidgets.QComboBox()
        self.cb_nearest_agg.addItems(["mean", "nearest", "median"])

        ag.addWidget(QtWidgets.QLabel("Algorithm:"), 0, 0)
        ag.addWidget(self.cb_algorithm, 0, 1)
        ag.addWidget(QtWidgets.QLabel("k:"), 0, 2)
        ag.addWidget(self.spin_k, 0, 3)
        ag.addWidget(QtWidgets.QLabel("radius:"), 0, 4)
        ag.addWidget(self.spin_radius, 0, 5)

        ag.addWidget(QtWidgets.QLabel("IDW power:"), 1, 0)
        ag.addWidget(self.spin_idw_power, 1, 1)
        ag.addWidget(QtWidgets.QLabel("TIN max triangle area:"), 1, 2)
        ag.addWidget(self.spin_tin_max_area, 1, 3)
        ag.addWidget(QtWidgets.QLabel("Kriging variogram:"), 1, 4)
        ag.addWidget(self.cb_variogram, 1, 5)
        ag.addWidget(QtWidgets.QLabel("Kriging nlags:"), 1, 6)
        ag.addWidget(self.spin_nlags, 1, 7)

        ag.addWidget(QtWidgets.QLabel("Nearest aggregation:"), 2, 0)
        ag.addWidget(self.cb_nearest_agg, 2, 1)

        # Output group
        out_group = QtWidgets.QGroupBox("Output")
        v.addWidget(out_group)
        og = QtWidgets.QGridLayout(out_group)

        self.cb_write_tif = QtWidgets.QCheckBox("Write GeoTIFF"); self.cb_write_tif.setChecked(True)
        self.cb_boundary = QtWidgets.QCheckBox("Export boundary Shapefile (.shp)"); self.cb_boundary.setChecked(False)
        self.cb_boundary_mode = QtWidgets.QComboBox(); self.cb_boundary_mode.addItems(["footprint", "convex"])
        self.spin_boundary_simplify = QtWidgets.QDoubleSpinBox(); self.spin_boundary_simplify.setRange(0.0, 1e6); self.spin_boundary_simplify.setDecimals(3)
        self.spin_boundary_buffer = QtWidgets.QDoubleSpinBox(); self.spin_boundary_buffer.setRange(-1e6, 1e6); self.spin_boundary_buffer.setDecimals(3)

        self.btn_choose_outdir = QtWidgets.QPushButton("Choose Output Dir (optional)")
        self.outdir_edit = QtWidgets.QLineEdit(); self.outdir_edit.setPlaceholderText("Default: same as source files")
        self.cb_overwrite = QtWidgets.QCheckBox("Overwrite if exists"); self.cb_overwrite.setChecked(False)
        self.cb_stop_on_error = QtWidgets.QCheckBox("Stop on first error (batch)"); self.cb_stop_on_error.setChecked(False)

        og.addWidget(self.cb_write_tif, 0, 0)
        og.addWidget(self.cb_boundary, 0, 1)
        og.addWidget(QtWidgets.QLabel("Boundary mode:"), 1, 0)
        og.addWidget(self.cb_boundary_mode, 1, 1)
        og.addWidget(QtWidgets.QLabel("Boundary simplify:"), 1, 2)
        og.addWidget(self.spin_boundary_simplify, 1, 3)
        og.addWidget(QtWidgets.QLabel("Boundary buffer:"), 1, 4)
        og.addWidget(self.spin_boundary_buffer, 1, 5)
        og.addWidget(self.btn_choose_outdir, 2, 0)
        og.addWidget(self.outdir_edit, 2, 1, 1, 5)
        og.addWidget(self.cb_overwrite, 3, 0)
        og.addWidget(self.cb_stop_on_error, 3, 1)

        # Run + progress + log
        run_group = QtWidgets.QGroupBox("Run")
        v.addWidget(run_group)
        rg = QtWidgets.QGridLayout(run_group)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.progress_overall = QtWidgets.QProgressBar()
        self.progress_file = QtWidgets.QProgressBar()
        self.lbl_current = QtWidgets.QLabel("Ready.")

        rg.addWidget(self.btn_start, 0, 0)
        rg.addWidget(self.btn_cancel, 0, 1)
        rg.addWidget(QtWidgets.QLabel("Overall:"), 1, 0)
        rg.addWidget(self.progress_overall, 1, 1, 1, 5)
        rg.addWidget(QtWidgets.QLabel("Current file:"), 2, 0)
        rg.addWidget(self.progress_file, 2, 1, 1, 5)
        rg.addWidget(self.lbl_current, 3, 0, 1, 6)

        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        v.addWidget(self.log_edit, 1)

    def _wire_events(self):
        self.btn_add_files.clicked.connect(self.on_add_files)
        self.btn_add_folder.clicked.connect(self.on_add_folder)
        self.btn_clear.clicked.connect(self.on_clear_files)

        self.input_epsg_edit.editingFinished.connect(self._update_degree_preview)
        self.output_epsg_edit.editingFinished.connect(self._update_degree_preview)

        self.btn_choose_outdir.clicked.connect(self.on_choose_outdir)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_cancel.clicked.connect(self.on_cancel)

        self.cb_algorithm.currentTextChanged.connect(self._refresh_kriging_available)
        self.cb_algorithm.currentTextChanged.connect(self._update_param_enablements)

    def _refresh_kriging_available(self):
        if not KRIGING_AVAILABLE:
            idx = self.cb_algorithm.findText("kriging")
            if idx >= 0:
                model = self.cb_algorithm.model()
                item = model.item(idx)
                if item is not None:
                    item.setEnabled(False)
            if self.cb_algorithm.currentText() == "kriging":
                self.cb_algorithm.setCurrentText("nearest")

    def _update_param_enablements(self):
        alg = self.cb_algorithm.currentText()
        # 預設全部先關
        for w in (
            self.spin_k, self.spin_radius, self.spin_idw_power,
            self.spin_tin_max_area, self.cb_variogram, self.spin_nlags,
            self.cb_nearest_agg
        ):
            w.setEnabled(False)

        if alg == "nearest":
            self.spin_k.setEnabled(True)
            self.spin_radius.setEnabled(True)
            self.cb_nearest_agg.setEnabled(True)
        elif alg == "idw":
            self.spin_k.setEnabled(True)
            self.spin_radius.setEnabled(True)
            self.spin_idw_power.setEnabled(True)
        elif alg == "tin":
            self.spin_tin_max_area.setEnabled(True)
        elif alg == "kriging":
            self.cb_variogram.setEnabled(True)
            self.spin_nlags.setEnabled(True)

    def append_log(self, text: str):
        self.log_edit.appendPlainText(text)
        self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())

    def on_add_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select point files", "", FILE_FILTER
        )
        for f in files:
            p = Path(f)
            if p not in self.files:
                self.files.append(p)
                self.list_files.addItem(str(p))
        self._update_degree_preview()

    def on_add_folder(self):
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder")
        if not dir_path:
            return
        p = Path(dir_path)
        exts = {".xyz", ".csv", ".txt", ".pts"}
        for fp in sorted(p.rglob("*")):
            if fp.suffix.lower() in exts and fp.is_file():
                if fp not in self.files:
                    self.files.append(fp)
                    self.list_files.addItem(str(fp))
        self._update_degree_preview()

    def on_clear_files(self):
        self.files.clear()
        self.list_files.clear()
        self._update_degree_preview()

    def on_choose_outdir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output directory")
        if d:
            self.outdir_edit.setText(d)

    def _update_degree_preview(self):
        input_epsg = self._get_int_or_none(self.input_epsg_edit.text())
        output_epsg = self._get_int_or_none(self.output_epsg_edit.text())
        epsg_out = output_epsg if output_epsg else input_epsg
        if epsg_out is None:
            self.lbl_out_unit.setText("Output unit: (unknown)")
            self.lbl_deg_preview.setText("~ Δlon=?, Δlat=? (deg)")
            return
        if is_geographic(epsg_out):
            self.lbl_out_unit.setText("Output unit: degrees")
            try:
                if self.files:
                    pts = read_xyz_like(self.files[0])
                    lon, lat = auto_center_lonlat(pts[:, :2], input_epsg or epsg_out, epsg_out)
                    dlon, dlat = degree_resolution_for_1m(lon, lat)
                    base_m = self.resolution_spin.value()
                    self.lbl_deg_preview.setText(f"~ Δlon={dlon*base_m:.8f}, Δlat={dlat*base_m:.8f} (deg)")
                else:
                    self.lbl_deg_preview.setText("~ Δlon=?, Δlat=? (deg) [no file]")
            except Exception:
                self.lbl_deg_preview.setText("~ Δlon=?, Δlat=? (deg) [calc failed]")
        else:
            self.lbl_out_unit.setText("Output unit: meters")
            self.lbl_deg_preview.setText("~ Δlon=?, Δlat=? (deg)")

    def _gather_job(self) -> JobConfig:
        files = list(self.files)
        if not files:
            raise ValueError("No input files.")
        input_epsg = self._get_int_or_none(self.input_epsg_edit.text())
        if input_epsg is None:
            raise ValueError("Input EPSG is required.")
        output_epsg = self._get_int_or_none(self.output_epsg_edit.text())

        interp = InterpParams(
            algorithm=self.cb_algorithm.currentText(),  # type: ignore
            grid_resolution_base_m=float(self.resolution_spin.value()),
            search_k=int(self.spin_k.value()),
            search_radius=float(self.spin_radius.value()) if self.spin_radius.value() > 0 else None,
            idw_power=float(self.spin_idw_power.value()),
            tin_max_triangle_area=float(self.spin_tin_max_area.value()) if self.spin_tin_max_area.value() > 0 else None,
            kriging_variogram=self.cb_variogram.currentText(),  # type: ignore
            kriging_n_lags=int(self.spin_nlags.value()),
            nearest_agg=self.cb_nearest_agg.currentText(),  # type: ignore
        )
        out_dir = Path(self.outdir_edit.text()).expanduser() if self.outdir_edit.text().strip() else None
        out = OutputParams(
            write_geotiff=self.cb_write_tif.isChecked(),
            export_boundary=self.cb_boundary.isChecked(),
            boundary_mode=self.cb_boundary_mode.currentText(),  # type: ignore
            boundary_simplify=float(self.spin_boundary_simplify.value()),
            boundary_buffer=float(self.spin_boundary_buffer.value()),
            nodata=-9999.0,
            dtype="float32",
            out_dir=out_dir,
            overwrite=self.cb_overwrite.isChecked()
        )
        crs = CRSParams(input_epsg=input_epsg, output_epsg=output_epsg)
        stop_on_error = self.cb_stop_on_error.isChecked()
        return JobConfig(files=files, interp=interp, crs=crs, out=out, stop_on_error=stop_on_error)

    def _get_int_or_none(self, s: str) -> Optional[int]:
        s = (s or "").strip()
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None

    def on_start(self):
        try:
            job = self._gather_job()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Invalid settings", str(e))
            return

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_overall.setValue(0)
        self.progress_file.setValue(0)
        self.lbl_current.setText("Running…")

        self.worker_thread = QThread()
        self.worker = InterpWorker(job)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.fileProgress.connect(self.on_file_progress)
        self.worker.message.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def on_cancel(self):
        if self.worker:
            self.worker.stop()
        self.btn_cancel.setEnabled(False)
        self.lbl_current.setText("Cancelling…")

    def on_progress(self, pct: int, msg: str):
        self.progress_overall.setValue(pct)

    def on_file_progress(self, idx: int, total: int, name: str):
        self.progress_file.setValue(int(idx / max(total, 1) * 100))
        self.lbl_current.setText(f"{idx}/{total}: {name}")

    def on_finished(self, succ: int, fail: int):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_current.setText(f"Done. Success={succ}, Failed={fail}")
        self.append_log(f"Done. Success={succ}, Failed={fail}")

# -----------------------------
# Main
# -----------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
