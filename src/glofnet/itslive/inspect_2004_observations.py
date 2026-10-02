"""
Inspect all 2004 ITS_LIVE observations over the Shishper glacier.

Purpose
-------
Determine whether the selected 2004 representative observation is uniquely
empty or whether other 2004 observations contain valid velocity measurements.

This is a diagnostic script only.

It:
    1. Loads the representative-date manifest.
    2. Selects all observations from 2004.
    3. Loads the ITS_LIVE Zarr metadata.
    4. Uses the existing Shishper spatial subset.
    5. Extracts ONLY the raw `v` variable for each 2004 observation.
    6. Counts valid/missing pixels.
    7. Calculates basic velocity statistics.
    8. Sorts observations by valid-pixel coverage.

It does NOT download:
    - vx
    - vy
    - v_error
    - interp_mask

Those should only be investigated after we find an observation
with usable velocity data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ======================================================================
# CONFIGURATION
# ======================================================================

YEAR = 2004

MANIFEST_PATH = Path(
    "data/raw/itslive/temporal_v1/"
    "RGI2000-v7.0-G-14-08488_representative_dates_v1.csv"
)

ZARR_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/"
    "N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)

ARRAY_NAME = "v"

GLACIER_ID = "RGI2000-v7.0-G-14-08488"

# Shishper spatial subset already validated in the previous scripts.
Y_START = 514
Y_END = 640

X_START = 486
X_END = 594

# Zarr chunk dimensions from metadata.
TIME_CHUNK_SIZE = 20_000
Y_CHUNK_SIZE = 10
X_CHUNK_SIZE = 10

# ITS_LIVE missing value from .zattrs.
MISSING_VALUE = -32767

# Small delay between S3 requests.
REQUEST_DELAY = 0.15


# ======================================================================
# HTTP SESSION
# ======================================================================

SESSION = requests.Session()

# Cache decompressed chunks during this run.
#
# Important:
# This must be defined OUTSIDE read_zarr_chunk().
#
# If it is created inside the function, it is reset on every call and
# provides no caching benefit.
CHUNK_CACHE: dict[str, np.ndarray] = {}


# ======================================================================
# HTTP HELPERS
# ======================================================================

def http_get(url: str) -> requests.Response:
    """
    Download one HTTP resource using the shared requests session.
    """

    response = SESSION.get(
        url,
        timeout=60,
    )

    response.raise_for_status()

    return response


# ======================================================================
# ZARR METADATA
# ======================================================================

def load_array_metadata() -> tuple[dict, dict]:
    """
    Load .zarray and .zattrs for the v array.
    """

    zarray_url = f"{ZARR_URL}/{ARRAY_NAME}/.zarray"
    zattrs_url = f"{ZARR_URL}/{ARRAY_NAME}/.zattrs"

    print("Loading array metadata:")
    print(zarray_url)

    response = http_get(zarray_url)
    array_meta = response.json()

    time.sleep(REQUEST_DELAY)

    print("Loading array attributes:")
    print(zattrs_url)

    response = http_get(zattrs_url)
    array_attrs = response.json()

    time.sleep(REQUEST_DELAY)

    return array_meta, array_attrs


# ======================================================================
# BLOSC DECODER
# ======================================================================

def decode_blosc(
    compressed: bytes,
    compressor: dict | None,
) -> bytes:
    """
    Decode a Blosc-compressed Zarr chunk.

    Requires python-blosc.

    Install if necessary:

        pip install blosc
    """

    if compressor is None:
        return compressed

    if compressor.get("id") != "blosc":
        raise ValueError(
            f"Unsupported compressor: {compressor}"
        )

    import blosc

    return blosc.decompress(compressed)


# ======================================================================
# RAW ZARR CHUNK READER
# ======================================================================

def read_zarr_chunk(
    array_name: str,
    chunk_key: str,
    array_meta: dict,
) -> np.ndarray:
    """
    Download, decompress and decode one Zarr v2 chunk.

    Returns
    -------
    np.ndarray
        Flat decoded chunk array.
    """

    chunk_url = (
        f"{ZARR_URL}/"
        f"{array_name}/"
        f"{chunk_key}"
    )

    # --------------------------------------------------------------
    # CACHE
    # --------------------------------------------------------------

    if chunk_url in CHUNK_CACHE:
        return CHUNK_CACHE[chunk_url]

    # --------------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------------

    response = http_get(chunk_url)

    time.sleep(REQUEST_DELAY)

    # --------------------------------------------------------------
    # DECODE
    # --------------------------------------------------------------

    raw = decode_blosc(
        response.content,
        array_meta.get("compressor"),
    )

    dtype = np.dtype(
        array_meta["dtype"]
    )

    values = np.frombuffer(
        raw,
        dtype=dtype,
    )

    # --------------------------------------------------------------
    # VALIDATE CHUNK SIZE
    # --------------------------------------------------------------

    expected_size = int(np.prod(array_meta["chunks"]))

    if values.size != expected_size:
        raise ValueError(
            f"Decoded chunk has {values.size} values, "
            f"expected {expected_size}."
        )

    CHUNK_CACHE[chunk_url] = values

    return values


# ======================================================================
# COORDINATE LOADING
# ======================================================================

def load_coordinates() -> tuple[np.ndarray, np.ndarray]:
    """
    Load x and y coordinate arrays from the Zarr store.
    """

    print()
    print("=" * 70)
    print("LOADING COORDINATES")
    print("=" * 70)

    coordinates = {}

    for name in ("x", "y"):

        url = f"{ZARR_URL}/{name}/.zarray"

        response = http_get(url)
        meta = response.json()

        time.sleep(REQUEST_DELAY)

        chunk_key = "0"

        url = f"{ZARR_URL}/{name}/{chunk_key}"

        response = http_get(url)

        time.sleep(REQUEST_DELAY)

        raw = decode_blosc(
            response.content,
            meta.get("compressor"),
        )

        dtype = np.dtype(meta["dtype"])

        values = np.frombuffer(
            raw,
            dtype=dtype,
        )

        expected = meta["shape"][0]

        if values.size != expected:
            raise ValueError(
                f"{name} coordinate size mismatch: "
                f"{values.size} != {expected}"
            )

        coordinates[name] = values

    x = coordinates["x"]
    y = coordinates["y"]

    print(f"x count: {len(x)}")
    print(
        f"x range: {x.min()} -> {x.max()}"
    )

    print(f"y count: {len(y)}")
    print(
        f"y range: {y.min()} -> {y.max()}"
    )

    return x, y


# ======================================================================
# OBSERVATION EXTRACTION
# ======================================================================

def extract_observation(
    cube_index: int,
    array_meta: dict,
) -> np.ndarray:
    """
    Extract the complete Shishper spatial subset for one observation.

    Returns
    -------
    np.ndarray
        2-D raw int16 velocity array with shape:

            (Y_END - Y_START, X_END - X_START)
    """

    time_chunk = cube_index // TIME_CHUNK_SIZE

    time_offset = (
        cube_index % TIME_CHUNK_SIZE
    )

    y_chunk_start = Y_START // Y_CHUNK_SIZE
    y_chunk_end = (
        (Y_END - 1) // Y_CHUNK_SIZE
    )

    x_chunk_start = X_START // X_CHUNK_SIZE
    x_chunk_end = (
        (X_END - 1) // X_CHUNK_SIZE
    )

    output_height = Y_END - Y_START
    output_width = X_END - X_START

    output = np.full(
        (
            output_height,
            output_width,
        ),
        MISSING_VALUE,
        dtype=np.int16,
    )

    spatial_chunks = (
        (y_chunk_end - y_chunk_start + 1)
        *
        (x_chunk_end - x_chunk_start + 1)
    )

    print(
        f"  time chunk: {time_chunk}"
    )

    print(
        f"  time offset: {time_offset}"
    )

    print(
        f"  Y chunks: "
        f"{y_chunk_start} -> {y_chunk_end}"
    )

    print(
        f"  X chunks: "
        f"{x_chunk_start} -> {x_chunk_end}"
    )

    print(
        f"  Spatial chunks required: "
        f"{spatial_chunks}"
    )

    downloaded = 0

    # --------------------------------------------------------------
    # Loop through spatial chunks
    # --------------------------------------------------------------

    for y_chunk in range(
        y_chunk_start,
        y_chunk_end + 1,
    ):

        for x_chunk in range(
            x_chunk_start,
            x_chunk_end + 1,
        ):

            chunk_key = (
                f"{time_chunk}."
                f"{y_chunk}."
                f"{x_chunk}"
            )

            chunk = read_zarr_chunk(
                ARRAY_NAME,
                chunk_key,
                array_meta,
            )

            # ------------------------------------------------------
            # Chunk shape
            # ------------------------------------------------------

            chunk_array = chunk.reshape(
                TIME_CHUNK_SIZE,
                Y_CHUNK_SIZE,
                X_CHUNK_SIZE,
            )

            # ------------------------------------------------------
            # Extract requested time slice
            # ------------------------------------------------------

            observation = chunk_array[
                time_offset,
                :,
                :,
            ]

            # ------------------------------------------------------
            # Global coordinates covered by this chunk
            # ------------------------------------------------------

            chunk_y_start = (
                y_chunk * Y_CHUNK_SIZE
            )

            chunk_x_start = (
                x_chunk * X_CHUNK_SIZE
            )

            chunk_y_end = (
                chunk_y_start + Y_CHUNK_SIZE
            )

            chunk_x_end = (
                chunk_x_start + X_CHUNK_SIZE
            )

            # ------------------------------------------------------
            # Intersection with requested spatial subset
            # ------------------------------------------------------

            iy_start = max(
                Y_START,
                chunk_y_start,
            )

            iy_end = min(
                Y_END,
                chunk_y_end,
            )

            ix_start = max(
                X_START,
                chunk_x_start,
            )

            ix_end = min(
                X_END,
                chunk_x_end,
            )

            if (
                iy_start >= iy_end
                or ix_start >= ix_end
            ):
                continue

            # ------------------------------------------------------
            # Source offsets
            # ------------------------------------------------------

            source_y_start = (
                iy_start - chunk_y_start
            )

            source_y_end = (
                iy_end - chunk_y_start
            )

            source_x_start = (
                ix_start - chunk_x_start
            )

            source_x_end = (
                ix_end - chunk_x_start
            )

            # ------------------------------------------------------
            # Destination offsets
            # ------------------------------------------------------

            destination_y_start = (
                iy_start - Y_START
            )

            destination_y_end = (
                iy_end - Y_START
            )

            destination_x_start = (
                ix_start - X_START
            )

            destination_x_end = (
                ix_end - X_START
            )

            # ------------------------------------------------------
            # Copy into output
            # ------------------------------------------------------

            output[
                destination_y_start:
                destination_y_end,
                destination_x_start:
                destination_x_end,
            ] = observation[
                source_y_start:
                source_y_end,
                source_x_start:
                source_x_end,
            ]

            downloaded += 1

            print(
                f"    {downloaded}/"
                f"{spatial_chunks} chunks downloaded",
                end="\r",
            )

    print()

    return output


# ======================================================================
# STATISTICS
# ======================================================================

def calculate_statistics(
    values: np.ndarray,
) -> dict:
    """
    Calculate velocity statistics while respecting ITS_LIVE
    missing-value encoding.
    """

    values = np.asarray(values)

    missing_mask = (
        values == MISSING_VALUE
    )

    valid_mask = ~missing_mask

    total = values.size

    missing_count = int(
        missing_mask.sum()
    )

    valid_count = int(
        valid_mask.sum()
    )

    missing_percentage = (
        missing_count / total * 100
        if total
        else 0.0
    )

    valid_percentage = (
        valid_count / total * 100
        if total
        else 0.0
    )

    result = {
        "total_pixels": total,
        "missing_pixels": missing_count,
        "valid_pixels": valid_count,
        "missing_percentage": missing_percentage,
        "valid_percentage": valid_percentage,
    }

    if valid_count > 0:

        valid = values[
            valid_mask
        ].astype(np.float64)

        result.update(
            {
                "min": float(
                    np.min(valid)
                ),
                "max": float(
                    np.max(valid)
                ),
                "mean": float(
                    np.mean(valid)
                ),
                "median": float(
                    np.median(valid)
                ),
            }
        )

    else:

        result.update(
            {
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "median": np.nan,
            }
        )

    return result


# ======================================================================
# PRINT OBSERVATION RESULT
# ======================================================================

def print_observation_result(
    row: pd.Series,
    stats: dict,
) -> None:
    """
    Print statistics for one observation.
    """

    print()
    print(
        f"Date:       {row['mid_date']}"
    )

    print(
        f"Cube index: {int(row['cube_index'])}"
    )

    print(
        f"Total:      {stats['total_pixels']:,}"
    )

    print(
        f"Valid:      {stats['valid_pixels']:,}"
    )

    print(
        f"Missing:    {stats['missing_pixels']:,}"
    )

    print(
        f"Valid %:    {stats['valid_percentage']:.4f}%"
    )

    if stats["valid_pixels"] > 0:

        print(
            f"v min:      {stats['min']:.3f}"
        )

        print(
            f"v max:      {stats['max']:.3f}"
        )

        print(
            f"v mean:     {stats['mean']:.3f}"
        )

        print(
            f"v median:   {stats['median']:.3f}"
        )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("       ITS_LIVE 2004 OBSERVATION SEARCH")
    print("=" * 70)

    # ==============================================================
    # LOAD MANIFEST
    # ==============================================================

    print()
    print("=" * 70)
    print("LOADING MANIFEST")
    print("=" * 70)

    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        "Manifest columns:"
    )

    print(
        list(manifest.columns)
    )

    # ==============================================================
    # VALIDATE MANIFEST
    # ==============================================================

    required_columns = {
        "year",
        "cube_index",
        "mid_date",
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:

        raise ValueError(
            "Manifest is missing required "
            f"columns: {missing_columns}"
        )

    # ==============================================================
    # SELECT 2004
    # ==============================================================

    observations = manifest[
        manifest["year"] == YEAR
    ].copy()

    observations = observations.sort_values(
        "mid_date"
    ).reset_index(
        drop=True
    )

    print(
        f"Target year: {YEAR}"
    )

    print(
        f"2004 observations: "
        f"{len(observations)}"
    )

    if observations.empty:

        print(
            "\nNo observations found."
        )

        return

    print()
    print(
        observations[
            [
                "year",
                "cube_index",
                "mid_date",
            ]
        ].to_string(
            index=False
        )
    )

    # ==============================================================
    # LOAD ZARR METADATA
    # ==============================================================

    print()
    print("=" * 70)
    print("LOADING ZARR METADATA")
    print("=" * 70)

    array_meta, array_attrs = (
        load_array_metadata()
    )

    print()
    print(
        "Array shape:",
        array_meta["shape"],
    )

    print(
        "Array chunks:",
        array_meta["chunks"],
    )

    print(
        "Array dtype:",
        array_meta["dtype"],
    )

    print(
        "Missing value:",
        array_attrs.get(
            "missing_value"
        ),
    )

    # ==============================================================
    # VALIDATE MISSING VALUE
    # ==============================================================

    metadata_missing_value = (
        array_attrs.get(
            "missing_value"
        )
    )

    if (
        metadata_missing_value
        is not None
        and metadata_missing_value
        != MISSING_VALUE
    ):

        raise ValueError(
            "Configured missing value "
            f"{MISSING_VALUE} does not match "
            f"Zarr metadata "
            f"{metadata_missing_value}."
        )

    # ==============================================================
    # LOAD COORDINATES
    # ==============================================================

    x, y = load_coordinates()

    print()
    print(
        "Selected Shishper subset:"
    )

    print(
        f"  Y indices: "
        f"{Y_START} -> {Y_END - 1}"
    )

    print(
        f"  X indices: "
        f"{X_START} -> {X_END - 1}"
    )

    print(
        f"  Shape: "
        f"{Y_END - Y_START} x "
        f"{X_END - X_START}"
    )

    print(
        f"  Pixels: "
        f"{(Y_END - Y_START) * (X_END - X_START):,}"
    )

    print()
    print(
        f"  X extent: "
        f"{x[X_START]} -> "
        f"{x[X_END - 1]}"
    )

    print(
        f"  Y extent: "
        f"{y[Y_START]} -> "
        f"{y[Y_END - 1]}"
    )

    # ==============================================================
    # PROCESS OBSERVATIONS
    # ==============================================================

    results = []

    for index, row in observations.iterrows():

        cube_index = int(
            row["cube_index"]
        )

        print()
        print("=" * 70)
        print(
            f"OBSERVATION "
            f"{index + 1}/{len(observations)}"
        )
        print("=" * 70)

        print(
            f"Year:       {YEAR}"
        )

        print(
            f"Cube index: {cube_index}"
        )

        print(
            f"Date:       {row['mid_date']}"
        )

        # ----------------------------------------------------------
        # Extract raw v
        # ----------------------------------------------------------

        values = extract_observation(
            cube_index,
            array_meta,
        )

        # ----------------------------------------------------------
        # Statistics
        # ----------------------------------------------------------

        stats = calculate_statistics(
            values
        )

        print_observation_result(
            row,
            stats,
        )

        results.append(
            {
                "year": YEAR,
                "cube_index": cube_index,
                "mid_date": row["mid_date"],
                **stats,
            }
        )

    # ==============================================================
    # BUILD RESULTS TABLE
    # ==============================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        [
            "valid_pixels",
            "valid_percentage",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    # ==============================================================
    # FINAL SUMMARY
    # ==============================================================

    print()
    print("=" * 70)
    print("2004 OBSERVATION SUMMARY")
    print("=" * 70)

    display_columns = [
        "cube_index",
        "mid_date",
        "valid_pixels",
        "valid_percentage",
        "min",
        "max",
        "mean",
        "median",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ==============================================================
    # BEST OBSERVATION
    # ==============================================================

    print()
    print("=" * 70)
    print("BEST 2004 OBSERVATION BY VALID COVERAGE")
    print("=" * 70)

    best = results_df.iloc[0]

    print(
        f"Cube index: "
        f"{int(best['cube_index'])}"
    )

    print(
        f"Date:       "
        f"{best['mid_date']}"
    )

    print(
        f"Valid pixels: "
        f"{int(best['valid_pixels']):,}"
    )

    print(
        f"Valid percentage: "
        f"{best['valid_percentage']:.4f}%"
    )

    if int(best["valid_pixels"]) > 0:

        print(
            f"v min: "
            f"{best['min']:.3f} m/year"
        )

        print(
            f"v max: "
            f"{best['max']:.3f} m/year"
        )

        print(
            f"v mean: "
            f"{best['mean']:.3f} m/year"
        )

        print(
            f"v median: "
            f"{best['median']:.3f} m/year"
        )

        print()
        print(
            "RESULT: At least one 2004 observation "
            "contains valid velocity data."
        )

    else:

        print()
        print(
            "RESULT: NONE of the 2004 observations "
            "contains valid velocity data in the "
            "selected Shishper spatial subset."
        )

    # ==============================================================
    # SAVE DIAGNOSTIC CSV
    # ==============================================================

    output_path = Path(
        "data/raw/itslive/temporal_v1/"
        "RGI2000-v7.0-G-14-08488_2004_observation_inspection.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIC RESULTS SAVED")
    print("=" * 70)

    print(
        f"Output: {output_path}"
    )

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()