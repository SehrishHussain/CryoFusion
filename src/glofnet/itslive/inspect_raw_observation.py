"""
Inspect one complete ITS_LIVE Zarr observation directly from raw chunks.

Purpose
-------
Diagnose whether the selected ITS_LIVE observation actually contains
valid velocity values across the complete Shishper spatial subset.

This script:
    1. Loads Zarr metadata.
    2. Reads x/y coordinates.
    3. Selects cube_index 13191 (2004 observation).
    4. Uses the same Shishper spatial subset as the downloader.
    5. Downloads all required raw v chunks.
    6. Extracts only the requested time slice from each chunk.
    7. Counts missing and valid raw values.
    8. Reports raw and physical-value statistics.
    9. Reports the spatial locations of valid pixels.

It does NOT modify the downloader and does NOT create a NetCDF.
"""

from __future__ import annotations

import json
import time
from typing import Any

import blosc
import numpy as np
import requests


# ======================================================================
# CONFIGURATION
# ======================================================================

ZARR_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/"
    "N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)

ARRAY_NAME = "v"

CUBE_INDEX = 13191

# These are the spatial indices selected by the downloader
Y_START = 514
Y_STOP = 640       # Python exclusive -> 126 rows

X_START = 486
X_STOP = 594       # Python exclusive -> 108 columns

MISSING_VALUE = -32767

REQUEST_DELAY = 0.15

TIMEOUT = 60


# ======================================================================
# HTTP
# ======================================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": "CryoFusion-ITS-LIVE-Inspector/1.0"
    }
)


def http_get(url: str) -> bytes:
    """
    Download raw bytes from an HTTP URL.
    """
    response = SESSION.get(
        url,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    return response.content


# ======================================================================
# METADATA
# ======================================================================

def load_json(url: str) -> dict[str, Any]:
    """
    Download and decode a JSON resource.
    """
    raw = http_get(url)

    return json.loads(
        raw.decode("utf-8")
    )


def load_array_metadata() -> dict[str, Any]:
    """
    Load v/.zarray metadata.
    """
    url = (
        f"{ZARR_URL}/"
        f"{ARRAY_NAME}/"
        ".zarray"
    )

    print("Loading array metadata:")
    print(url)

    metadata = load_json(url)

    return metadata


def load_array_attributes() -> dict[str, Any]:
    """
    Load v/.zattrs metadata.
    """
    url = (
        f"{ZARR_URL}/"
        f"{ARRAY_NAME}/"
        ".zattrs"
    )

    print()
    print("Loading array attributes:")
    print(url)

    attributes = load_json(url)

    return attributes


# ======================================================================
# COORDINATES
# ======================================================================

def read_coordinate_array(
    array_name: str,
    metadata: dict[str, Any],
) -> np.ndarray:
    """
    Read a complete 1-D Zarr coordinate array.

    This is suitable for x/y because their chunks contain the
    complete coordinate arrays.
    """

    shape = metadata["shape"]
    chunks = metadata["chunks"]

    if len(shape) != 1:
        raise ValueError(
            f"{array_name} is not 1-dimensional."
        )

    if len(chunks) != 1:
        raise ValueError(
            f"{array_name} has unexpected chunk structure: {chunks}"
        )

    dtype = np.dtype(
        metadata["dtype"]
    )

    chunk_size = chunks[0]

    values = []

    start = 0

    while start < shape[0]:

        chunk_index = start // chunk_size

        url = (
            f"{ZARR_URL}/"
            f"{array_name}/"
            f"{chunk_index}"
        )

        raw = http_get(url)

        decoded = decode_chunk(
            raw,
            metadata.get("compressor"),
        )

        chunk_values = np.frombuffer(
            decoded,
            dtype=dtype,
        )

        values.append(
            chunk_values
        )

        start += chunk_size

    result = np.concatenate(
        values
    )

    return result[: shape[0]]


# ======================================================================
# CHUNK DECODING
# ======================================================================

def decode_chunk(
    raw: bytes,
    compressor: dict[str, Any] | None,
) -> bytes:
    """
    Decode one Zarr v2 compressed chunk.

    ITS_LIVE uses Blosc compression.
    """

    if compressor is None:
        return raw

    compressor_id = compressor.get("id")

    if compressor_id != "blosc":
        raise ValueError(
            f"Unsupported compressor: {compressor_id}"
        )

    return blosc.decompress(raw)


# ======================================================================
# RAW CHUNK READING
# ======================================================================

def download_zarr_chunk(
    chunk_y: int,
    chunk_x: int,
    metadata: dict[str, Any],
) -> np.ndarray:
    """
    Download one complete spatial chunk for the selected time chunk.

    Zarr layout:

        v/
          <time_chunk>.<y_chunk>.<x_chunk>

    For cube index 13191 the time chunk is 0 because:

        13191 // 20000 = 0
    """

    chunk_shape = metadata["chunks"]

    time_chunk_size = chunk_shape[0]
    y_chunk_size = chunk_shape[1]
    x_chunk_size = chunk_shape[2]

    time_chunk = (
        CUBE_INDEX // time_chunk_size
    )

    chunk_key = (
        f"{time_chunk}."
        f"{chunk_y}."
        f"{chunk_x}"
    )

    url = (
        f"{ZARR_URL}/"
        f"{ARRAY_NAME}/"
        f"{chunk_key}"
    )

    raw = http_get(url)

    time.sleep(
        REQUEST_DELAY
    )

    decoded = decode_chunk(
        raw,
        metadata.get("compressor"),
    )

    dtype = np.dtype(
        metadata["dtype"]
    )

    values = np.frombuffer(
        decoded,
        dtype=dtype,
    )

    expected_values = (
        time_chunk_size
        * y_chunk_size
        * x_chunk_size
    )

    if values.size != expected_values:
        raise ValueError(
            f"Unexpected chunk size for {chunk_key}: "
            f"{values.size} values; "
            f"expected {expected_values}"
        )

    return values.reshape(
        (
            time_chunk_size,
            y_chunk_size,
            x_chunk_size,
        )
    )


# ======================================================================
# OBSERVATION EXTRACTION
# ======================================================================

def extract_observation(
    metadata: dict[str, Any],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    """
    Extract the complete 126 x 108 v observation for cube_index 13191.
    """

    shape = metadata["shape"]

    chunk_shape = metadata["chunks"]

    total_time = shape[0]
    total_y = shape[1]
    total_x = shape[2]

    time_chunk_size = chunk_shape[0]
    y_chunk_size = chunk_shape[1]
    x_chunk_size = chunk_shape[2]

    if not (
        0 <= CUBE_INDEX < total_time
    ):
        raise IndexError(
            f"Cube index {CUBE_INDEX} outside "
            f"time dimension 0..{total_time - 1}"
        )

    output = np.full(
        (
            Y_STOP - Y_START,
            X_STOP - X_START,
        ),
        MISSING_VALUE,
        dtype=np.int16,
    )

    time_chunk = (
        CUBE_INDEX // time_chunk_size
    )

    time_offset = (
        CUBE_INDEX % time_chunk_size
    )

    y_chunk_start = (
        Y_START // y_chunk_size
    )

    y_chunk_end = (
        (Y_STOP - 1) // y_chunk_size
    )

    x_chunk_start = (
        X_START // x_chunk_size
    )

    x_chunk_end = (
        (X_STOP - 1) // x_chunk_size
    )

    print()
    print("=" * 70)
    print("EXTRACTING RAW OBSERVATION")
    print("=" * 70)

    print(
        f"Cube index: {CUBE_INDEX}"
    )

    print(
        f"Time chunk: {time_chunk}"
    )

    print(
        f"Time offset: {time_offset}"
    )

    print(
        f"Y chunks: {y_chunk_start} -> "
        f"{y_chunk_end}"
    )

    print(
        f"X chunks: {x_chunk_start} -> "
        f"{x_chunk_end}"
    )

    total_chunks = (
        (y_chunk_end - y_chunk_start + 1)
        *
        (x_chunk_end - x_chunk_start + 1)
    )

    print(
        f"Spatial chunks required: "
        f"{total_chunks}"
    )

    completed = 0

    for chunk_y in range(
        y_chunk_start,
        y_chunk_end + 1,
    ):

        for chunk_x in range(
            x_chunk_start,
            x_chunk_end + 1,
        ):

            chunk = download_zarr_chunk(
                chunk_y,
                chunk_x,
                metadata,
            )

            observation = chunk[
                time_offset
            ]

            global_y_start = (
                chunk_y * y_chunk_size
            )

            global_x_start = (
                chunk_x * x_chunk_size
            )

            global_y_end = (
                global_y_start
                + y_chunk_size
            )

            global_x_end = (
                global_x_start
                + x_chunk_size
            )

            overlap_y_start = max(
                Y_START,
                global_y_start,
            )

            overlap_y_end = min(
                Y_STOP,
                global_y_end,
            )

            overlap_x_start = max(
                X_START,
                global_x_start,
            )

            overlap_x_end = min(
                X_STOP,
                global_x_end,
            )

            if (
                overlap_y_start >= overlap_y_end
                or
                overlap_x_start >= overlap_x_end
            ):
                continue

            source_y_start = (
                overlap_y_start
                - global_y_start
            )

            source_y_end = (
                overlap_y_end
                - global_y_start
            )

            source_x_start = (
                overlap_x_start
                - global_x_start
            )

            source_x_end = (
                overlap_x_end
                - global_x_start
            )

            destination_y_start = (
                overlap_y_start
                - Y_START
            )

            destination_y_end = (
                overlap_y_end
                - Y_START
            )

            destination_x_start = (
                overlap_x_start
                - X_START
            )

            destination_x_end = (
                overlap_x_end
                - X_START
            )

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

            completed += 1

            print(
                f"  {completed}/{total_chunks} "
                f"chunks downloaded"
            )

    return output


# ======================================================================
# STATISTICS
# ======================================================================

def inspect_values(
    raw_values: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    attributes: dict[str, Any],
) -> None:
    """
    Print statistics for the complete raw observation.
    """

    print()
    print("=" * 70)
    print("RAW OBSERVATION STATISTICS")
    print("=" * 70)

    total = raw_values.size

    missing_mask = (
        raw_values == MISSING_VALUE
    )

    valid_mask = (
        ~missing_mask
    )

    missing_count = int(
        missing_mask.sum()
    )

    valid_count = int(
        valid_mask.sum()
    )

    missing_percentage = (
        missing_count / total * 100
    )

    valid_percentage = (
        valid_count / total * 100
    )

    print(
        f"Total pixels:       {total:,}"
    )

    print(
        f"Missing pixels:     {missing_count:,}"
    )

    print(
        f"Valid pixels:       {valid_count:,}"
    )

    print(
        f"Missing percentage: {missing_percentage:.4f}%"
    )

    print(
        f"Valid percentage:   {valid_percentage:.4f}%"
    )

    if valid_count == 0:

        print()
        print(
            "NO VALID VELOCITY VALUES "
            "FOUND IN COMPLETE SPATIAL SUBSET."
        )

        return

    valid_values = (
        raw_values[valid_mask]
    )

    print()
    print("RAW INTEGER VALUES")
    print("-" * 70)

    print(
        f"Minimum: {valid_values.min()}"
    )

    print(
        f"Maximum: {valid_values.max()}"
    )

    print(
        f"Mean:    {valid_values.mean():.4f}"
    )

    print(
        f"Median:  {np.median(valid_values):.4f}"
    )

    print()
    print(
        "First 50 valid raw values:"
    )

    print(
        valid_values[:50]
    )

    # --------------------------------------------------------------
    # Scaling metadata
    # --------------------------------------------------------------

    scale_factor = attributes.get(
        "scale_factor"
    )

    add_offset = attributes.get(
        "add_offset"
    )

    print()
    print("=" * 70)
    print("ENCODING / SCALING")
    print("=" * 70)

    print(
        f"scale_factor: {scale_factor}"
    )

    print(
        f"add_offset:   {add_offset}"
    )

    if scale_factor is not None:

        scaled = (
            valid_values.astype(
                np.float64
            )
            * float(scale_factor)
        )

        if add_offset is not None:

            scaled += float(
                add_offset
            )

        print()
        print("SCALED PHYSICAL VALUES")
        print("-" * 70)

        print(
            f"Minimum: {scaled.min():.6f} m/year"
        )

        print(
            f"Maximum: {scaled.max():.6f} m/year"
        )

        print(
            f"Mean:    {scaled.mean():.6f} m/year"
        )

        print(
            f"Median:  {np.median(scaled):.6f} m/year"
        )

    else:

        print()
        print(
            "No scale_factor found in .zattrs."
        )

        print(
            "Raw integer values are NOT being "
            "interpreted as physical velocity."
        )

    # --------------------------------------------------------------
    # Coordinate locations
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("VALID PIXEL LOCATIONS")
    print("=" * 70)

    valid_indices = np.argwhere(
        valid_mask
    )

    print(
        f"Number of valid pixels: "
        f"{len(valid_indices):,}"
    )

    sample_count = min(
        20,
        len(valid_indices),
    )

    print()
    print(
        f"First {sample_count} valid pixels:"
    )

    for row, col in valid_indices[
        :sample_count
    ]:

        print(
            f"  array=({row}, {col}) | "
            f"x={x_coords[col]:.2f} | "
            f"y={y_coords[row]:.2f} | "
            f"raw={raw_values[row, col]}"
        )

    # --------------------------------------------------------------
    # Unique values
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("RAW VALUE DISTRIBUTION")
    print("=" * 70)

    unique, counts = np.unique(
        valid_values,
        return_counts=True,
    )

    order = np.argsort(
        counts
    )[::-1]

    top_n = min(
        20,
        len(unique),
    )

    print(
        f"Unique valid values: "
        f"{len(unique):,}"
    )

    print()
    print(
        f"Most frequent {top_n} valid raw values:"
    )

    for index in order[:top_n]:

        print(
            f"  {unique[index]:8d} : "
            f"{counts[index]:6d} pixels"
        )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    print("=" * 70)
    print("ITS_LIVE RAW 2004 OBSERVATION INSPECTION")
    print("=" * 70)

    print()
    print(
        f"Zarr: {ZARR_URL}"
    )

    print(
        f"Variable: {ARRAY_NAME}"
    )

    print(
        f"Cube index: {CUBE_INDEX}"
    )

    print()
    print(
        "Spatial subset:"
    )

    print(
        f"  Y indices: {Y_START} -> {Y_STOP - 1}"
    )

    print(
        f"  X indices: {X_START} -> {X_STOP - 1}"
    )

    print(
        f"  Shape: "
        f"{Y_STOP - Y_START} x "
        f"{X_STOP - X_START}"
    )

    print(
        f"  Pixels: "
        f"{(Y_STOP - Y_START) * (X_STOP - X_START):,}"
    )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    metadata = load_array_metadata()

    attributes = load_array_attributes()

    print()
    print("=" * 70)
    print("ARRAY METADATA")
    print("=" * 70)

    print(
        json.dumps(
            metadata,
            indent=4,
        )
    )

    print()
    print("=" * 70)
    print("ARRAY ATTRIBUTES")
    print("=" * 70)

    print(
        json.dumps(
            attributes,
            indent=4,
        )
    )

    # --------------------------------------------------------------
    # Validate missing value
    # --------------------------------------------------------------

    metadata_missing = attributes.get(
        "missing_value"
    )

    if metadata_missing is not None:

        if int(metadata_missing) != MISSING_VALUE:

            print()
            print(
                "WARNING:"
            )

            print(
                f".zattrs missing_value = "
                f"{metadata_missing}"
            )

            print(
                f"Configured missing value = "
                f"{MISSING_VALUE}"
            )

    # --------------------------------------------------------------
    # Load coordinates
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING COORDINATES")
    print("=" * 70)

    x_metadata = load_json(
        f"{ZARR_URL}/x/.zarray"
    )

    y_metadata = load_json(
        f"{ZARR_URL}/y/.zarray"
    )

    x_coords = read_coordinate_array(
        "x",
        x_metadata,
    )

    y_coords = read_coordinate_array(
        "y",
        y_metadata,
    )

    print(
        f"x count: {len(x_coords)}"
    )

    print(
        f"x range: "
        f"{x_coords.min()} -> "
        f"{x_coords.max()}"
    )

    print(
        f"y count: {len(y_coords)}"
    )

    print(
        f"y range: "
        f"{y_coords.min()} -> "
        f"{y_coords.max()}"
    )

    # --------------------------------------------------------------
    # Check selected subset
    # --------------------------------------------------------------

    if X_STOP > len(x_coords):

        raise IndexError(
            "X subset extends beyond "
            "coordinate array."
        )

    if Y_STOP > len(y_coords):

        raise IndexError(
            "Y subset extends beyond "
            "coordinate array."
        )

    subset_x = x_coords[
        X_START:X_STOP
    ]

    subset_y = y_coords[
        Y_START:Y_STOP
    ]

    print()
    print(
        "Selected coordinate extent:"
    )

    print(
        f"  X: "
        f"{subset_x.min()} -> "
        f"{subset_x.max()}"
    )

    print(
        f"  Y: "
        f"{subset_y.min()} -> "
        f"{subset_y.max()}"
    )

    # --------------------------------------------------------------
    # Extract complete observation
    # --------------------------------------------------------------

    raw_observation = extract_observation(
        metadata,
        x_coords,
        y_coords,
    )

    # --------------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------------

    inspect_values(
        raw_observation,
        subset_x,
        subset_y,
        attributes,
    )

    # --------------------------------------------------------------
    # Final conclusion
    # --------------------------------------------------------------

    valid_count = int(
        (
            raw_observation
            != MISSING_VALUE
        ).sum()
    )

    print()
    print("=" * 70)
    print("FINAL DIAGNOSTIC")
    print("=" * 70)

    if valid_count == 0:

        print()
        print(
            "RESULT: The complete 2004 Shishper "
            "spatial subset contains ZERO valid "
            "raw v pixels."
        )

        print()
        print(
            "This means the NaN result in the "
            "NetCDF is consistent with the raw "
            "ITS_LIVE data for this observation."
        )

    else:

        print()
        print(
            f"RESULT: The complete 2004 Shishper "
            f"spatial subset contains "
            f"{valid_count:,} valid raw v pixels."
        )

        print()
        print(
            "Therefore the previous NetCDF result "
            "of zero valid pixels indicates that "
            "the extraction or missing-value handling "
            "needs further investigation."
        )

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()