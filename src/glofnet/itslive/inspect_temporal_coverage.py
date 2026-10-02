"""
ITS_LIVE TEMPORAL COVERAGE INSPECTION

Purpose
-------
Inspect a small number of strategically selected ITS_LIVE observations
for the Shishper glacier instead of downloading all observations.

The script distinguishes:

    VALID_DATA
        At least one valid velocity pixel exists in the selected
        Shishper spatial subset.

    ZERO_VALID_DATA
        All pixels in the selected spatial subset are the ITS_LIVE
        missing-value marker.

    DOWNLOAD_FAILURE
        Required Zarr chunks could not be downloaded after a limited
        number of attempts.

Strategy
--------
The manifest contains 38 annual observations from 1988-2025.

By default, only six observations are tested:

    1988
    2000
    2004
    2010
    2018
    2025

The 2004 observation is included because direct access to its Zarr
chunk was already confirmed by diagnose_itslive_access.py.

Run:
    python -m src.glofnet.itslive.inspect_temporal_coverage
"""

from __future__ import annotations

import json
import time
import zlib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from socket import gaierror
from typing import Any

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

ZARR_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/"
    "N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)

VARIABLE = "v"

MANIFEST_PATH = Path(
    "data/raw/itslive/temporal_v1/"
    "RGI2000-v7.0-G-14-08488_representative_dates_v1.csv"
)

OUTPUT_DIR = Path("data/raw/itslive/diagnostics")

OUTPUT_CSV = (
    OUTPUT_DIR
    / "RGI2000-v7.0-G-14-08488_strategic_temporal_coverage.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "RGI2000-v7.0-G-14-08488_strategic_temporal_coverage.json"
)


# Strategic observations.
#
# These are deliberately sparse. The objective is to establish
# temporal coverage behavior before downloading the entire series.
STRATEGIC_YEARS = [
    1988,
    2000,
    2004,
    2010,
    2018,
    2025,
]


# Limited retry policy.
#
# We do NOT want a broken network to cause the program to sit
# retrying for a long time.
MAX_RETRIES = 3
REQUEST_TIMEOUT = 20
RETRY_DELAY_SECONDS = 2


# Shishper RGI geometry in EPSG:4326.
#
# These are the bounds already established for the selected
# RGI glacier.
SHISHPER_MIN_LON = 74.535403
SHISHPER_MAX_LON = 74.680602
SHISHPER_MIN_LAT = 36.352031
SHISHPER_MAX_LAT = 36.489164


# ======================================================================
# EXCEPTIONS
# ======================================================================


class DownloadFailure(Exception):
    """Raised when a Zarr object cannot be downloaded."""


# ======================================================================
# PRINTING
# ======================================================================


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def subsection(title: str) -> None:
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# ======================================================================
# NETWORK
# ======================================================================


def download_bytes(url: str) -> bytes:
    """
    Download a URL with a small bounded retry policy.

    Important:
        This function raises DownloadFailure rather than silently
        converting a network failure into a zero-data result.
    """

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "Cryofusion-ITS-LIVE-Diagnostic/1.0",
                },
            )

            with urlopen(
                request,
                timeout=REQUEST_TIMEOUT,
            ) as response:

                return response.read()

        except (
            HTTPError,
            URLError,
            TimeoutError,
            gaierror,
            OSError,
        ) as exc:

            last_error = exc

            print(
                f"    Download failed "
                f"(attempt {attempt}/{MAX_RETRIES}): {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    raise DownloadFailure(
        f"Unable to download Zarr object after "
        f"{MAX_RETRIES} attempts:\n"
        f"{url}\n"
        f"Last error: {last_error}"
    )


# ======================================================================
# METADATA
# ======================================================================


def load_json(url: str) -> dict[str, Any]:

    data = download_bytes(url)

    try:
        return json.loads(data.decode("utf-8"))

    except Exception as exc:
        raise RuntimeError(
            f"Unable to decode JSON from:\n{url}"
        ) from exc


def load_array_metadata() -> tuple[dict[str, Any], dict[str, Any]]:

    section("LOADING ZARR METADATA")

    zarray_url = f"{ZARR_URL}/{VARIABLE}/.zarray"
    zattrs_url = f"{ZARR_URL}/{VARIABLE}/.zattrs"

    print(f"Loading .zarray:")
    print(zarray_url)

    zarray = load_json(zarray_url)

    print(f"Loading .zattrs:")
    print(zattrs_url)

    zattrs = load_json(zattrs_url)

    print()
    print(f"Array shape:       {zarray['shape']}")
    print(f"Array chunks:      {zarray['chunks']}")
    print(f"Array dtype:       {zarray['dtype']}")
    print(f"Missing value:     {zattrs.get('missing_value')}")
    print(f"Units:             {zattrs.get('units')}")
    print(f"Compressor:        {zarray.get('compressor')}")

    return zarray, zattrs


# ======================================================================
# ZARR CHUNK DECODING
# ======================================================================


def decode_blosc_chunk(
    compressed: bytes,
    zarray: dict[str, Any],
) -> np.ndarray:
    """
    Decode a Blosc-compressed Zarr v2 chunk.

    Uses numcodecs if available.

    This is the same compression scheme observed in the ITS_LIVE
    .zarray metadata.
    """

    try:
        from numcodecs import Blosc

    except ImportError as exc:

        raise RuntimeError(
            "numcodecs is required to decode ITS_LIVE Blosc chunks.\n"
            "Install it with:\n"
            "    pip install numcodecs"
        ) from exc

    compressor = zarray["compressor"]

    decoder = Blosc(
        cname=compressor["cname"],
        clevel=compressor["clevel"],
        shuffle=compressor["shuffle"],
        blocksize=compressor.get("blocksize", 0),
    )

    raw = decoder.decode(compressed)

    dtype = np.dtype(zarray["dtype"])

    expected_values = int(np.prod(zarray["chunks"]))

    values = np.frombuffer(
        raw,
        dtype=dtype,
    )

    if values.size != expected_values:

        raise RuntimeError(
            "Decoded chunk has unexpected number of values.\n"
            f"Expected: {expected_values}\n"
            f"Received: {values.size}"
        )

    return values.reshape(tuple(zarray["chunks"]))


# ======================================================================
# COORDINATES
# ======================================================================


def load_coordinate(name: str) -> np.ndarray:

    subsection(f"LOADING {name.upper()} COORDINATE")

    metadata_url = f"{ZARR_URL}/{name}/.zarray"

    print("Loading coordinate metadata:")
    print(metadata_url)

    metadata = load_json(metadata_url)

    shape = tuple(metadata["shape"])
    chunks = tuple(metadata["chunks"])
    dtype = np.dtype(metadata["dtype"])

    print(f"Shape:       {shape}")
    print(f"Chunks:      {chunks}")
    print(f"Dtype:       {dtype}")
    print(f"Compressor:  {metadata.get('compressor')}")

    # Coordinates are one-dimensional and this cube has one chunk.
    values: list[np.ndarray] = []

    number_of_chunks = int(
        np.ceil(shape[0] / chunks[0])
    )

    for chunk_index in range(number_of_chunks):

        chunk_url = (
            f"{ZARR_URL}/{name}/{chunk_index}"
        )

        print(
            f"Downloading coordinate chunk "
            f"{chunk_index + 1}/{number_of_chunks}"
        )

        compressed = download_bytes(chunk_url)

        compressor = metadata.get("compressor")

        if compressor is None:
            decoded = compressed

        else:

            try:
                from numcodecs import Blosc

            except ImportError as exc:

                raise RuntimeError(
                    "numcodecs is required for coordinate decoding."
                ) from exc

            decoder = Blosc(
                cname=compressor["cname"],
                clevel=compressor["clevel"],
                shuffle=compressor["shuffle"],
                blocksize=compressor.get("blocksize", 0),
            )

            decoded = decoder.decode(compressed)

        array = np.frombuffer(
            decoded,
            dtype=dtype,
        )

        values.append(array)

    coordinate = np.concatenate(values)

    return coordinate[: shape[0]]


def load_coordinates() -> tuple[np.ndarray, np.ndarray]:

    section("LOADING COORDINATES")

    x = load_coordinate("x")
    y = load_coordinate("y")

    print()
    print(f"x count: {len(x)}")
    print(f"x range: {x.min()} -> {x.max()}")

    print(f"y count: {len(y)}")
    print(f"y range: {y.min()} -> {y.max()}")

    return x, y


# ======================================================================
# SPATIAL SUBSET
# ======================================================================


def find_index_range(
    coordinates: np.ndarray,
    minimum: float,
    maximum: float,
) -> tuple[int, int]:

    mask = (
        (coordinates >= minimum)
        & (coordinates <= maximum)
    )

    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise RuntimeError(
            "No coordinate values fall inside requested extent."
        )

    return int(indices.min()), int(indices.max())


def identify_shishper_subset(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, int]:

    section("IDENTIFYING SHISHPER SPATIAL SUBSET")

    # NOTE:
    # The RGI Shishper geometry is in EPSG:4326 while the ITS_LIVE
    # cube coordinates are EPSG:32643.
    #
    # The established Shishper subset below is therefore based on
    # the already validated cube-coordinate extent:
    #
    # X: 458452.5 -> 471292.5
    # Y: 4023187.5 -> 4038187.5

    x_min = 458452.5
    x_max = 471292.5

    y_min = 4023187.5
    y_max = 4038187.5

    x_start, x_end = find_index_range(
        x,
        x_min,
        x_max,
    )

    y_start, y_end = find_index_range(
        y,
        y_min,
        y_max,
    )

    print()
    print(f"Y indices: {y_start} -> {y_end}")
    print(f"X indices: {x_start} -> {x_end}")

    height = y_end - y_start + 1
    width = x_end - x_start + 1

    print(f"Shape:     {height} x {width}")
    print(f"Pixels:    {height * width:,}")

    print()
    print(
        f"X extent:  "
        f"{x[x_start]} -> {x[x_end]}"
    )

    print(
        f"Y extent:  "
        f"{y[y_start]} -> {y[y_end]}"
    )

    return {
        "y_start": y_start,
        "y_end": y_end,
        "x_start": x_start,
        "x_end": x_end,
    }


# ======================================================================
# CHUNK EXTRACTION
# ======================================================================


def required_chunks(
    cube_index: int,
    subset: dict[str, int],
    chunks: list[int],
) -> list[tuple[int, int, int]]:

    time_chunk_size = chunks[0]
    y_chunk_size = chunks[1]
    x_chunk_size = chunks[2]

    time_chunk = cube_index // time_chunk_size

    y_chunk_start = (
        subset["y_start"] // y_chunk_size
    )

    y_chunk_end = (
        subset["y_end"] // y_chunk_size
    )

    x_chunk_start = (
        subset["x_start"] // x_chunk_size
    )

    x_chunk_end = (
        subset["x_end"] // x_chunk_size
    )

    chunk_coordinates = []

    for y_chunk in range(
        y_chunk_start,
        y_chunk_end + 1,
    ):

        for x_chunk in range(
            x_chunk_start,
            x_chunk_end + 1,
        ):

            chunk_coordinates.append(
                (
                    time_chunk,
                    y_chunk,
                    x_chunk,
                )
            )

    return chunk_coordinates


def extract_observation(
    cube_index: int,
    subset: dict[str, int],
    zarray: dict[str, Any],
    missing_value: int,
) -> np.ndarray:

    chunks = zarray["chunks"]

    chunk_coordinates = required_chunks(
        cube_index,
        subset,
        chunks,
    )

    print(
        f"  Time chunk: "
        f"{cube_index // chunks[0]}"
    )

    print(
        f"  Spatial chunks required: "
        f"{len(chunk_coordinates)}"
    )

    # Allocate output with missing values.
    height = (
        subset["y_end"]
        - subset["y_start"]
        + 1
    )

    width = (
        subset["x_end"]
        - subset["x_start"]
        + 1
    )

    output = np.full(
        (height, width),
        missing_value,
        dtype=np.int16,
    )

    time_chunk = cube_index // chunks[0]
    time_offset = cube_index % chunks[0]

    for counter, (
        _time_chunk,
        y_chunk,
        x_chunk,
    ) in enumerate(chunk_coordinates, start=1):

        chunk_url = (
            f"{ZARR_URL}/{VARIABLE}/"
            f"{time_chunk}.{y_chunk}.{x_chunk}"
        )

        compressed = download_bytes(chunk_url)

        decoded = decode_blosc_chunk(
            compressed,
            zarray,
        )

        # Extract the requested time slice.
        spatial = decoded[
            time_offset,
            :,
            :,
        ]

        global_y_start = (
            y_chunk * chunks[1]
        )

        global_x_start = (
            x_chunk * chunks[2]
        )

        global_y_end = (
            global_y_start
            + chunks[1]
            - 1
        )

        global_x_end = (
            global_x_start
            + chunks[2]
            - 1
        )

        # Intersection with requested subset.
        y0 = max(
            subset["y_start"],
            global_y_start,
        )

        y1 = min(
            subset["y_end"],
            global_y_end,
        )

        x0 = max(
            subset["x_start"],
            global_x_start,
        )

        x1 = min(
            subset["x_end"],
            global_x_end,
        )

        if y0 > y1 or x0 > x1:
            continue

        source_y0 = y0 - global_y_start
        source_y1 = y1 - global_y_start + 1

        source_x0 = x0 - global_x_start
        source_x1 = x1 - global_x_start + 1

        target_y0 = (
            y0 - subset["y_start"]
        )

        target_y1 = (
            y1 - subset["y_start"] + 1
        )

        target_x0 = (
            x0 - subset["x_start"]
        )

        target_x1 = (
            x1 - subset["x_start"] + 1
        )

        output[
            target_y0:target_y1,
            target_x0:target_x1,
        ] = spatial[
            source_y0:source_y1,
            source_x0:source_x1,
        ]

        if (
            counter == 1
            or counter == len(chunk_coordinates)
        ):
            print(
                f"    {counter}/"
                f"{len(chunk_coordinates)} chunks"
            )

    return output


# ======================================================================
# OBSERVATION INSPECTION
# ======================================================================


def inspect_observation(
    row: pd.Series,
    subset: dict[str, int],
    zarray: dict[str, Any],
    zattrs: dict[str, Any],
) -> dict[str, Any]:

    year = int(row["year"])
    cube_index = int(row["cube_index"])
    mid_date = str(row["mid_date"])

    subsection(
        f"OBSERVATION: {year}"
    )

    print(f"Date:       {mid_date}")
    print(f"Cube index: {cube_index}")

    started = time.time()

    try:

        values = extract_observation(
            cube_index,
            subset,
            zarray,
            zattrs["missing_value"],
        )

    except DownloadFailure as exc:

        elapsed = time.time() - started

        print()
        print("STATUS: DOWNLOAD_FAILURE")
        print(str(exc))

        return {
            "year": year,
            "cube_index": cube_index,
            "mid_date": mid_date,
            "status": "DOWNLOAD_FAILURE",
            "total_pixels": int(values_size(subset)),
            "valid_pixels": np.nan,
            "missing_pixels": np.nan,
            "valid_percentage": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "elapsed_seconds": elapsed,
            "error": str(exc),
        }

    except Exception as exc:

        elapsed = time.time() - started

        print()
        print("STATUS: DOWNLOAD_FAILURE")
        print(
            "Unexpected data-access/decode error:"
        )
        print(str(exc))

        return {
            "year": year,
            "cube_index": cube_index,
            "mid_date": mid_date,
            "status": "DOWNLOAD_FAILURE",
            "total_pixels": int(values_size(subset)),
            "valid_pixels": np.nan,
            "missing_pixels": np.nan,
            "valid_percentage": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "elapsed_seconds": elapsed,
            "error": str(exc),
        }

    elapsed = time.time() - started

    missing_mask = (
        values == zattrs["missing_value"]
    )

    valid_mask = ~missing_mask

    valid_values = values[valid_mask]

    total = values.size
    valid = valid_values.size
    missing = total - valid

    valid_percentage = (
        valid / total * 100.0
        if total
        else 0.0
    )

    if valid == 0:

        status = "ZERO_VALID_DATA"

        minimum = np.nan
        maximum = np.nan
        mean = np.nan
        median = np.nan

    else:

        status = "VALID_DATA"

        minimum = float(valid_values.min())
        maximum = float(valid_values.max())
        mean = float(valid_values.mean())
        median = float(np.median(valid_values))

    print()
    print("RESULT")
    print(
        f"  Total pixels:   {total:,}"
    )
    print(
        f"  Valid pixels:   {valid:,}"
    )
    print(
        f"  Missing pixels: {missing:,}"
    )
    print(
        f"  Valid %:        {valid_percentage:.4f}%"
    )

    if status == "VALID_DATA":

        print()
        print("VALID VELOCITY STATISTICS")
        print(f"  Minimum: {minimum}")
        print(f"  Maximum: {maximum}")
        print(f"  Mean:    {mean}")
        print(f"  Median:  {median}")

    print()
    print(f"STATUS: {status}")
    print(
        f"Elapsed: {elapsed:.2f} seconds"
    )

    return {
        "year": year,
        "cube_index": cube_index,
        "mid_date": mid_date,
        "status": status,
        "total_pixels": total,
        "valid_pixels": valid,
        "missing_pixels": missing,
        "valid_percentage": valid_percentage,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "median": median,
        "elapsed_seconds": elapsed,
        "error": "",
    }


def values_size(
    subset: dict[str, int],
) -> int:

    height = (
        subset["y_end"]
        - subset["y_start"]
        + 1
    )

    width = (
        subset["x_end"]
        - subset["x_start"]
        + 1
    )

    return height * width


# ======================================================================
# STRATEGIC OBSERVATION SELECTION
# ======================================================================


def select_strategic_observations(
    manifest: pd.DataFrame,
) -> pd.DataFrame:

    subsection(
        "SELECTING STRATEGIC OBSERVATIONS"
    )

    selected = manifest[
        manifest["year"].isin(
            STRATEGIC_YEARS
        )
    ].copy()

    selected = selected.sort_values(
        "year"
    )

    print()
    print(
        f"Manifest observations: "
        f"{len(manifest)}"
    )

    print(
        f"Observations to inspect: "
        f"{len(selected)}"
    )

    print()

    for _, row in selected.iterrows():

        print(
            f"{int(row['year'])}: "
            f"cube {int(row['cube_index'])} "
            f"{row['mid_date']}"
        )

    missing_years = [
        year
        for year in STRATEGIC_YEARS
        if year not in set(
            selected["year"].astype(int)
        )
    ]

    if missing_years:

        raise RuntimeError(
            "The following strategic years are "
            f"missing from the manifest: "
            f"{missing_years}"
        )

    return selected


# ======================================================================
# OUTPUT
# ======================================================================


def save_results(
    results: list[dict[str, Any]],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(results)

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    summary = {
        "zarr": ZARR_URL,
        "variable": VARIABLE,
        "strategic_years": STRATEGIC_YEARS,
        "observations_tested": len(results),
        "status_counts": (
            dataframe["status"]
            .value_counts()
            .to_dict()
        ),
        "results": results,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    print()
    print(
        f"CSV saved:  {OUTPUT_CSV}"
    )

    print(
        f"JSON saved: {OUTPUT_JSON}"
    )


# ======================================================================
# MAIN
# ======================================================================


def main() -> None:

    section(
        "ITS_LIVE TEMPORAL COVERAGE INSPECTION"
    )

    # --------------------------------------------------------------
    # 1. Manifest
    # --------------------------------------------------------------

    section(
        "LOADING OBSERVATION MANIFEST"
    )

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

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

        raise RuntimeError(
            "Manifest is missing required "
            f"columns: {missing_columns}"
        )

    manifest["year"] = (
        pd.to_numeric(
            manifest["year"]
        ).astype(int)
    )

    manifest["cube_index"] = (
        pd.to_numeric(
            manifest["cube_index"]
        ).astype(int)
    )

    manifest["mid_date"] = pd.to_datetime(
        manifest["mid_date"]
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    print(
        f"Total observations: "
        f"{len(manifest)}"
    )

    print(
        f"Date range: "
        f"{manifest['mid_date'].min()} -> "
        f"{manifest['mid_date'].max()}"
    )

    print(
        f"Years: "
        f"{manifest['year'].min()} -> "
        f"{manifest['year'].max()}"
    )

    # --------------------------------------------------------------
    # 2. Annual coverage summary
    # --------------------------------------------------------------

    subsection(
        "OBSERVATIONS PER YEAR"
    )

    print(
        manifest.groupby("year")
        .size()
        .rename("observations")
        .to_string()
    )

    # --------------------------------------------------------------
    # 3. Strategic selection
    # --------------------------------------------------------------

    selected = (
        select_strategic_observations(
            manifest
        )
    )

    # --------------------------------------------------------------
    # 4. Metadata
    # --------------------------------------------------------------

    zarray, zattrs = (
        load_array_metadata()
    )

    # --------------------------------------------------------------
    # 5. Coordinates
    # --------------------------------------------------------------

    try:

        x, y = load_coordinates()

    except DownloadFailure as exc:

        section(
            "COORDINATE DOWNLOAD FAILURE"
        )

        print(str(exc))

        print()
        print(
            "The temporal inspection cannot "
            "continue because the Shishper "
            "spatial subset cannot be determined."
        )

        return

    # --------------------------------------------------------------
    # 6. Shishper subset
    # --------------------------------------------------------------

    subset = (
        identify_shishper_subset(
            x,
            y,
        )
    )

    # --------------------------------------------------------------
    # 7. Observations
    # --------------------------------------------------------------

    section(
        "INSPECTING STRATEGIC OBSERVATIONS"
    )

    results: list[dict[str, Any]] = []

    for position, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):

        print()
        print(
            f"[{position}/{len(selected)}]"
        )

        result = inspect_observation(
            row,
            subset,
            zarray,
            zattrs,
        )

        results.append(result)

    # --------------------------------------------------------------
    # 8. Summary
    # --------------------------------------------------------------

    section(
        "STRATEGIC TEMPORAL COVERAGE SUMMARY"
    )

    dataframe = pd.DataFrame(results)

    display_columns = [
        "year",
        "cube_index",
        "mid_date",
        "status",
        "valid_pixels",
        "valid_percentage",
        "min",
        "max",
        "mean",
        "median",
    ]

    print(
        dataframe[
            display_columns
        ].to_string(index=False)
    )

    # --------------------------------------------------------------
    # 9. Interpretation
    # --------------------------------------------------------------

    section(
        "FINAL DIAGNOSTIC"
    )

    valid_count = int(
        (
            dataframe["status"]
            == "VALID_DATA"
        ).sum()
    )

    zero_count = int(
        (
            dataframe["status"]
            == "ZERO_VALID_DATA"
        ).sum()
    )

    failure_count = int(
        (
            dataframe["status"]
            == "DOWNLOAD_FAILURE"
        ).sum()
    )

    print(
        f"Strategic observations tested: "
        f"{len(dataframe)}"
    )

    print(
        f"VALID_DATA:       {valid_count}"
    )

    print(
        f"ZERO_VALID_DATA:  {zero_count}"
    )

    print(
        f"DOWNLOAD_FAILURE: {failure_count}"
    )

    print()

    if valid_count:

        print(
            "VALID VELOCITY DATA WAS FOUND "
            "IN AT LEAST ONE STRATEGIC OBSERVATION."
        )

    if zero_count:

        print(
            "Some observations were successfully "
            "downloaded but contained zero valid "
            "velocity pixels in the selected "
            "Shishper subset."
        )

    if failure_count:

        print(
            "Some observations could not be "
            "tested because of download/access "
            "failures."
        )

        print(
            "Those observations MUST NOT be "
            "interpreted as zero velocity coverage."
        )

    print()
    print(
        "Important distinction:"
    )

    print(
        "  ZERO_VALID_DATA  = data successfully "
        "accessed, but no valid pixels."
    )

    print(
        "  DOWNLOAD_FAILURE = data could not be "
        "accessed, so coverage is unknown."
    )

    print(
        "  VALID_DATA       = at least one valid "
        "velocity pixel was found."
    )

    # --------------------------------------------------------------
    # 10. Save
    # --------------------------------------------------------------

    section(
        "SAVING DIAGNOSTIC RESULTS"
    )

    save_results(results)

    section(
        "INSPECTION COMPLETE"
    )


if __name__ == "__main__":
    main()