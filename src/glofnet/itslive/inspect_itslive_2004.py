"""
ITS_LIVE 2004 Observation Inspector
====================================

Purpose
-------
Inspect the successfully downloaded 2004 ITS_LIVE NetCDF observation
for Shishper Glacier.

Checks performed
----------------
1. Total number of pixels
2. Valid v pixels
3. Percentage of valid v pixels
4. Valid vx pixels
5. Valid vy pixels
6. v min/max/mean/median
7. v_error statistics
8. interp_mask values and counts
9. Whether valid velocity pixels fall inside the Shishper glacier polygon
10. Whether velocity magnitudes are physically plausible

This script does NOT modify the downloaded NetCDF.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================================
# CONFIGURATION
# ============================================================================

GLACIER_ID = "RGI2000-v7.0-G-14-08488"

NETCDF_PATH = Path(
    "data/raw/itslive/temporal_v1/"
    f"{GLACIER_ID}_2004-07-20T05-24-06.nc"
)

RGI_PATH = Path(
    "data/reference/RGI/"
    "RGI2000-v7.0-G-14_south_asia_west/"
    "RGI2000-v7.0-G-14_south_asia_west.shp"
)


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title: str) -> None:
    print()
    print(f"----- {title} -----")


# ============================================================================
# LOAD DATA
# ============================================================================

def load_dataset() -> xr.Dataset:
    """Load the downloaded ITS_LIVE NetCDF."""

    print_header("LOADING ITS_LIVE NETCDF")

    if not NETCDF_PATH.exists():
        raise FileNotFoundError(
            f"NetCDF file not found:\n{NETCDF_PATH}"
        )

    print(f"File: {NETCDF_PATH}")
    print(f"File size: {NETCDF_PATH.stat().st_size / (1024 ** 2):.3f} MB")

    ds = xr.open_dataset(NETCDF_PATH)

    print()
    print(ds)

    return ds


def load_glacier() -> gpd.GeoDataFrame:
    """Load the Shishper glacier polygon from RGI."""

    print_header("LOADING SHISHPER GLACIER POLYGON")

    if not RGI_PATH.exists():
        raise FileNotFoundError(
            f"RGI shapefile not found:\n{RGI_PATH}"
        )

    gdf = gpd.read_file(RGI_PATH)

    print(f"RGI polygons loaded: {len(gdf)}")
    print(f"RGI CRS: {gdf.crs}")

    glacier = gdf[gdf["rgi_id"] == GLACIER_ID].copy()

    if glacier.empty:
        raise ValueError(
            f"Glacier {GLACIER_ID} was not found in the RGI shapefile."
        )

    print(f"Target glacier: {GLACIER_ID}")

    geometry = glacier.geometry.iloc[0]

    minx, miny, maxx, maxy = geometry.bounds

    print("Glacier bounds:")
    print(f"  minx = {minx}")
    print(f"  miny = {miny}")
    print(f"  maxx = {maxx}")
    print(f"  maxy = {maxy}")

    return glacier


# ============================================================================
# BASIC DATASET INFORMATION
# ============================================================================

def inspect_dataset_structure(ds: xr.Dataset) -> None:
    """Inspect dimensions, coordinates, variables and metadata."""

    print_header("DATASET STRUCTURE")

    print(f"Dimensions:")
    for name, size in ds.sizes.items():
        print(f"  {name}: {size}")

    print()
    print("Coordinates:")
    for name in ds.coords:
        values = ds[name].values

        if values.size > 0:
            print(
                f"  {name}: "
                f"count={values.size}, "
                f"min={np.nanmin(values)}, "
                f"max={np.nanmax(values)}"
            )

    print()
    print("Data variables:")

    for name in ds.data_vars:
        variable = ds[name]

        print(
            f"  {name}: "
            f"dtype={variable.dtype}, "
            f"shape={variable.shape}"
        )

        if "units" in variable.attrs:
            print(f"      units={variable.attrs['units']}")

    print()
    print("Important dataset attributes:")

    for key in [
        "glacier_id",
        "observation_date",
        "cube_index",
        "interp_mask_status",
        "title",
    ]:
        if key in ds.attrs:
            print(f"  {key}: {ds.attrs[key]}")


# ============================================================================
# 1. TOTAL PIXELS
# ============================================================================

def calculate_total_pixels(ds: xr.Dataset) -> int:
    """Calculate total number of raster pixels."""

    print_header("1. TOTAL NUMBER OF PIXELS")

    ny = ds.sizes["y"]
    nx = ds.sizes["x"]

    total_pixels = ny * nx

    print(f"Y dimension: {ny}")
    print(f"X dimension: {nx}")
    print(f"Total pixels: {total_pixels:,}")

    return total_pixels


# ============================================================================
# VALIDITY HELPERS
# ============================================================================

def valid_mask(data: xr.DataArray) -> np.ndarray:
    """
    Return a boolean mask for finite/non-NaN values.
    """

    return np.isfinite(data.values)


def print_variable_validity(
    name: str,
    data: xr.DataArray,
    total_pixels: int,
) -> tuple[int, float]:
    """Print valid pixel count and percentage."""

    mask = valid_mask(data)

    valid_count = int(mask.sum())

    percentage = (
        valid_count / total_pixels * 100
        if total_pixels > 0
        else 0.0
    )

    print(f"{name}:")
    print(f"  Valid pixels: {valid_count:,}")
    print(f"  Invalid pixels: {total_pixels - valid_count:,}")
    print(f"  Valid percentage: {percentage:.4f}%")

    return valid_count, percentage


# ============================================================================
# 2. VALID V PIXELS
# ============================================================================

def inspect_v_validity(
    ds: xr.Dataset,
    total_pixels: int,
) -> np.ndarray:
    """Inspect valid velocity magnitude pixels."""

    print_header("2. VALID v PIXELS")

    v_mask = valid_mask(ds["v"])

    valid_count = int(v_mask.sum())

    print(f"Valid v pixels: {valid_count:,}")
    print(
        f"Invalid/NaN v pixels: "
        f"{total_pixels - valid_count:,}"
    )

    percentage = valid_count / total_pixels * 100

    print(f"Valid v percentage: {percentage:.4f}%")

    return v_mask


# ============================================================================
# 3. VALID v PERCENTAGE
# ============================================================================

def inspect_v_percentage(
    v_mask: np.ndarray,
    total_pixels: int,
) -> None:
    """Print percentage of valid v pixels."""

    print_header("3. VALID v PERCENTAGE")

    valid_count = int(v_mask.sum())

    percentage = (
        valid_count / total_pixels * 100
        if total_pixels > 0
        else 0.0
    )

    print(f"{percentage:.4f}% of all raster pixels contain finite v values.")


# ============================================================================
# 4. VALID vx PIXELS
# ============================================================================

def inspect_vx_validity(
    ds: xr.Dataset,
    total_pixels: int,
) -> np.ndarray:
    """Inspect valid vx pixels."""

    print_header("4. VALID vx PIXELS")

    vx_mask = valid_mask(ds["vx"])

    valid_count = int(vx_mask.sum())

    print(f"Valid vx pixels: {valid_count:,}")
    print(f"Invalid/NaN vx pixels: {total_pixels - valid_count:,}")
    print(
        f"Valid vx percentage: "
        f"{valid_count / total_pixels * 100:.4f}%"
    )

    return vx_mask


# ============================================================================
# 5. VALID vy PIXELS
# ============================================================================

def inspect_vy_validity(
    ds: xr.Dataset,
    total_pixels: int,
) -> np.ndarray:
    """Inspect valid vy pixels."""

    print_header("5. VALID vy PIXELS")

    vy_mask = valid_mask(ds["vy"])

    valid_count = int(vy_mask.sum())

    print(f"Valid vy pixels: {valid_count:,}")
    print(f"Invalid/NaN vy pixels: {total_pixels - valid_count:,}")
    print(
        f"Valid vy percentage: "
        f"{valid_count / total_pixels * 100:.4f}%"
    )

    return vy_mask


# ============================================================================
# 6. v STATISTICS
# ============================================================================

def inspect_v_statistics(ds: xr.Dataset) -> None:
    """Calculate v min/max/mean/median."""

    print_header("6. v STATISTICS")

    values = ds["v"].values.astype(np.float64)

    valid = values[np.isfinite(values)]

    if valid.size == 0:
        print("No valid v values found.")
        return

    print(f"Valid observations: {valid.size:,}")
    print(f"Minimum: {np.min(valid):.6f}")
    print(f"Maximum: {np.max(valid):.6f}")
    print(f"Mean:    {np.mean(valid):.6f}")
    print(f"Median:  {np.median(valid):.6f}")
    print(f"Std:     {np.std(valid):.6f}")

    print()
    print("Percentiles:")

    for percentile in [1, 5, 25, 50, 75, 95, 99]:
        value = np.percentile(valid, percentile)
        print(f"  P{percentile:02d}: {value:.6f}")


# ============================================================================
# 7. v_error STATISTICS
# ============================================================================

def inspect_v_error_statistics(ds: xr.Dataset) -> None:
    """Calculate statistics for v_error."""

    print_header("7. v_error STATISTICS")

    values = ds["v_error"].values.astype(np.float64)

    valid = values[np.isfinite(values)]

    if valid.size == 0:
        print("No valid v_error values found.")
        return

    print(f"Valid observations: {valid.size:,}")
    print(f"Minimum: {np.min(valid):.6f}")
    print(f"Maximum: {np.max(valid):.6f}")
    print(f"Mean:    {np.mean(valid):.6f}")
    print(f"Median:  {np.median(valid):.6f}")
    print(f"Std:     {np.std(valid):.6f}")

    print()
    print("Percentiles:")

    for percentile in [1, 5, 25, 50, 75, 95, 99]:
        value = np.percentile(valid, percentile)
        print(f"  P{percentile:02d}: {value:.6f}")


# ============================================================================
# 8. interp_mask
# ============================================================================

def inspect_interp_mask(ds: xr.Dataset) -> None:
    """Inspect unique interp_mask values."""

    print_header("8. interp_mask VALUES")

    values = ds["interp_mask"].values

    finite = values[np.isfinite(values)]

    if finite.size == 0:
        print("No finite interp_mask values found.")
        return

    unique, counts = np.unique(
        finite,
        return_counts=True,
    )

    print("Unique values:")

    for value, count in zip(unique, counts):
        percentage = count / finite.size * 100

        print(
            f"  {value}: "
            f"{count:,} pixels "
            f"({percentage:.4f}%)"
        )

    print()
    print(f"Total finite interp_mask pixels: {finite.size:,}")


# ============================================================================
# 9. PIXELS INSIDE GLACIER POLYGON
# ============================================================================

def inspect_spatial_membership(
    ds: xr.Dataset,
    glacier: gpd.GeoDataFrame,
    v_mask: np.ndarray,
) -> None:
    """
    Determine whether valid velocity pixels fall inside the
    Shishper glacier polygon.

    IMPORTANT:
    The NetCDF x/y coordinates are EPSG:32643.
    The RGI polygon is transformed to EPSG:32643 before
    testing pixel-center membership.
    """

    print_header(
        "9. VALID VELOCITY PIXELS INSIDE SHISHPER GLACIER"
    )

    glacier_projected = glacier.to_crs("EPSG:32643")

    geometry = glacier_projected.geometry.iloc[0]

    x = ds["x"].values
    y = ds["y"].values

    # Mesh of pixel-center coordinates.
    xx, yy = np.meshgrid(x, y)

    # Flatten coordinates.
    x_flat = xx.ravel()
    y_flat = yy.ravel()

    # Flatten velocity validity mask.
    valid_flat = v_mask.ravel()

    valid_indices = np.flatnonzero(valid_flat)

    print(f"Total valid v pixels: {len(valid_indices):,}")

    if len(valid_indices) == 0:
        print("No valid velocity pixels to test.")
        return

    valid_x = x_flat[valid_indices]
    valid_y = y_flat[valid_indices]

    # Construct GeoDataFrame containing only valid pixel centers.
    points = gpd.GeoDataFrame(
        {
            "pixel_index": valid_indices,
            "x": valid_x,
            "y": valid_y,
        },
        geometry=gpd.points_from_xy(
            valid_x,
            valid_y,
        ),
        crs="EPSG:32643",
    )

    # Pixel-center membership test.
    inside = points.geometry.within(geometry)

    inside_count = int(inside.sum())
    outside_count = len(points) - inside_count

    inside_percentage = (
        inside_count / len(points) * 100
        if len(points) > 0
        else 0.0
    )

    outside_percentage = (
        outside_count / len(points) * 100
        if len(points) > 0
        else 0.0
    )

    print()
    print(f"Valid velocity pixels inside glacier: {inside_count:,}")
    print(
        f"Valid velocity pixels outside glacier: "
        f"{outside_count:,}"
    )

    print()
    print(
        f"Percentage of valid velocity pixels inside: "
        f"{inside_percentage:.4f}%"
    )

    print(
        f"Percentage outside: "
        f"{outside_percentage:.4f}%"
    )

    print()

    if inside_count > 0:
        print("Spatial membership result: PASS")
        print(
            "At least some valid velocity pixel centers "
            "fall inside the Shishper glacier polygon."
        )
    else:
        print("Spatial membership result: FAIL")
        print(
            "No valid velocity pixel centers fall inside "
            "the Shishper glacier polygon."
        )


# ============================================================================
# 10. PHYSICAL PLAUSIBILITY
# ============================================================================

def inspect_physical_plausibility(ds: xr.Dataset) -> None:
    """
    Evaluate whether velocity magnitudes appear physically plausible.

    This is a screening test, NOT a scientific validation.

    The ITS_LIVE values should be interpreted using their NetCDF
    metadata/units. We therefore print units first and avoid blindly
    applying a unit conversion.
    """

    print_header("10. VELOCITY PHYSICAL PLAUSIBILITY")

    v = ds["v"].values.astype(np.float64)

    valid_v = v[np.isfinite(v)]

    if valid_v.size == 0:
        print("No valid velocity values available.")
        return

    units = ds["v"].attrs.get("units", "unknown")

    print(f"v units: {units}")

    print()
    print("Velocity magnitude summary:")
    print(f"  Minimum: {np.min(valid_v):.6f}")
    print(f"  Maximum: {np.max(valid_v):.6f}")
    print(f"  Mean:    {np.mean(valid_v):.6f}")
    print(f"  Median:  {np.median(valid_v):.6f}")

    # Absolute values are used because an unexpected negative
    # magnitude may indicate a direction/sign convention issue.
    absolute_v = np.abs(valid_v)

    print()
    print("Absolute velocity magnitude:")
    print(f"  Maximum: {np.max(absolute_v):.6f}")
    print(f"  Mean:    {np.mean(absolute_v):.6f}")
    print(f"  Median:  {np.median(absolute_v):.6f}")

    print()
    print("Extreme-value screening:")

    # These are deliberately screening thresholds rather than
    # scientific rejection criteria.
    thresholds = [
        100,
        500,
        1000,
        2000,
        5000,
    ]

    for threshold in thresholds:
        count = int(
            np.sum(absolute_v > threshold)
        )

        percentage = count / valid_v.size * 100

        print(
            f"  |v| > {threshold}: "
            f"{count:,} pixels "
            f"({percentage:.4f}%)"
        )

    print()
    print(
        "IMPORTANT: Extreme values are not automatically invalid. "
        "They must be interpreted using ITS_LIVE units, uncertainty, "
        "glacier dynamics, and the specific observation."
    )


# ============================================================================
# CONSISTENCY CHECKS
# ============================================================================

def inspect_component_consistency(ds: xr.Dataset) -> None:
    """
    Compare v against sqrt(vx^2 + vy^2).

    This is useful because v should generally represent the
    velocity magnitude derived from the velocity components.
    """

    print_header("VELOCITY COMPONENT CONSISTENCY")

    v = ds["v"].values.astype(np.float64)
    vx = ds["vx"].values.astype(np.float64)
    vy = ds["vy"].values.astype(np.float64)

    valid = (
        np.isfinite(v)
        & np.isfinite(vx)
        & np.isfinite(vy)
    )

    if not np.any(valid):
        print("No pixels have valid v, vx and vy simultaneously.")
        return

    expected_v = np.sqrt(
        vx[valid] ** 2 +
        vy[valid] ** 2
    )

    actual_v = np.abs(v[valid])

    difference = actual_v - expected_v

    print(
        f"Pixels with valid v/vx/vy: "
        f"{valid.sum():,}"
    )

    print()
    print("Component-derived magnitude:")
    print(f"  Min:    {np.min(expected_v):.6f}")
    print(f"  Max:    {np.max(expected_v):.6f}")
    print(f"  Mean:   {np.mean(expected_v):.6f}")
    print(f"  Median: {np.median(expected_v):.6f}")

    print()
    print("|v| - sqrt(vx² + vy²):")
    print(f"  Mean difference:   {np.mean(difference):.6f}")
    print(f"  Median difference: {np.median(difference):.6f}")
    print(f"  Max abs difference: {np.max(np.abs(difference)):.6f}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 70)
    print("       ITS_LIVE 2004 OBSERVATION INSPECTOR")
    print("=" * 70)

    ds = None

    try:
        # ------------------------------------------------------------
        # Load
        # ------------------------------------------------------------

        ds = load_dataset()

        glacier = load_glacier()

        # ------------------------------------------------------------
        # Structure
        # ------------------------------------------------------------

        inspect_dataset_structure(ds)

        # ------------------------------------------------------------
        # 1. Total pixels
        # ------------------------------------------------------------

        total_pixels = calculate_total_pixels(ds)

        # ------------------------------------------------------------
        # 2. Valid v
        # ------------------------------------------------------------

        v_mask = inspect_v_validity(
            ds,
            total_pixels,
        )

        # ------------------------------------------------------------
        # 3. Valid v percentage
        # ------------------------------------------------------------

        inspect_v_percentage(
            v_mask,
            total_pixels,
        )

        # ------------------------------------------------------------
        # 4. Valid vx
        # ------------------------------------------------------------

        inspect_vx_validity(
            ds,
            total_pixels,
        )

        # ------------------------------------------------------------
        # 5. Valid vy
        # ------------------------------------------------------------

        inspect_vy_validity(
            ds,
            total_pixels,
        )

        # ------------------------------------------------------------
        # 6. v statistics
        # ------------------------------------------------------------

        inspect_v_statistics(ds)

        # ------------------------------------------------------------
        # 7. v_error
        # ------------------------------------------------------------

        inspect_v_error_statistics(ds)

        # ------------------------------------------------------------
        # 8. interp_mask
        # ------------------------------------------------------------

        inspect_interp_mask(ds)

        # ------------------------------------------------------------
        # 9. Spatial membership
        # ------------------------------------------------------------

        inspect_spatial_membership(
            ds,
            glacier,
            v_mask,
        )

        # ------------------------------------------------------------
        # 10. Physical plausibility
        # ------------------------------------------------------------

        inspect_physical_plausibility(ds)

        # ------------------------------------------------------------
        # Additional consistency test
        # ------------------------------------------------------------

        inspect_component_consistency(ds)

        # ------------------------------------------------------------
        # Complete
        # ------------------------------------------------------------

        print()
        print("=" * 70)
        print("       INSPECTION COMPLETE")
        print("=" * 70)

    finally:
        if ds is not None:
            ds.close()


if __name__ == "__main__":
    main()