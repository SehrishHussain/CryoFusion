
"""
ITS_LIVE MANIFEST -> MID_DATE TEMPORAL DIAGNOSTIC

Purpose
-------
Isolate and verify the temporal relationship between the existing
38-row strategic manifest and the actual ITS_LIVE `mid_date` Zarr
coordinate.

This script intentionally DOES NOT:
    - load the velocity (`v`) array
    - inspect Shishper spatial chunks
    - download any velocity chunks
    - perform any spatial/S3 velocity diagnostic

It only:
    1. Loads the existing 38-row manifest.
    2. Loads mid_date/.zarray and mid_date/.zattrs.
    3. Retrieves the actual cube date for every manifest cube_index.
    4. Compares calendar dates, not nanosecond timestamps.
    5. Flags DATE_MATCH / DATE_MISMATCH.
    6. Flags duplicate cube indices.
    7. Flags non-monotonic cube-index relationships.
    8. Explicitly highlights 2012 / cube_index 171.
    9. Saves the complete mapping to CSV.

Important
---------
A DATE_MATCH means:

    manifest_mid_date.date() == actual_cube_mid_date.date()

The time-of-day is deliberately ignored.

A DATE_MISMATCH means the calendar dates differ.

A non-monotonic cube index is a separate diagnostic flag. It does NOT
automatically mean that an individual DATE_MATCH is invalid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import urllib.request
import zlib


# ============================================================================
# CONFIGURATION
# ============================================================================

MANIFEST_PATH = Path(
    "data/raw/itslive/temporal_v1/"
    "RGI2000-v7.0-G-14-08488_representative_dates_v1.csv"
)

OUTPUT_DIR = Path("data/raw/itslive/diagnostics")

OUTPUT_CSV = (
    OUTPUT_DIR
    / "RGI2000-v7.0-G-14-08488_manifest_mid_date_mapping.csv"
)

TIME_ZARR_BASE = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr/"
    "mid_date"
)

DOWNLOAD_TIMEOUT = 20

SUSPICIOUS_YEAR = 2012
SUSPICIOUS_CUBE_INDEX = 171


# ============================================================================
# EXCEPTIONS
# ============================================================================


class DownloadFailure(RuntimeError):
    """The remote mid_date object could not be downloaded."""


class DecodeFailure(RuntimeError):
    """The downloaded mid_date object could not be decoded."""


class ManifestValidationFailure(RuntimeError):
    """The local manifest is invalid or incomplete."""


# ============================================================================
# OUTPUT HELPERS
# ============================================================================


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def subsection(title: str) -> None:
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


# ============================================================================
# NETWORK
# ============================================================================


def download_once(url: str) -> bytes:
    """
    Download one mid_date Zarr object exactly once.

    There is intentionally no retry logic here. If S3/network access fails,
    the script stops rather than silently turning an access problem into
    a temporal mismatch.
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CryoFusion-ITS-LIVE-Temporal-Diagnostic/1.0",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=DOWNLOAD_TIMEOUT,
        ) as response:

            status = getattr(response, "status", 200)

            if status != 200:
                raise DownloadFailure(
                    f"HTTP {status} while downloading:\n{url}"
                )

            data = response.read()

            if not data:
                raise DownloadFailure(
                    f"Empty HTTP response:\n{url}"
                )

            return data

    except DownloadFailure:
        raise

    except Exception as exc:
        raise DownloadFailure(
            f"{type(exc).__name__}: {exc}\nURL: {url}"
        ) from exc


# ============================================================================
# JSON / ZARR METADATA
# ============================================================================


def load_json(url: str) -> dict[str, Any]:
    data = download_once(url)

    try:
        return json.loads(data.decode("utf-8"))

    except Exception as exc:
        raise DecodeFailure(
            f"Downloaded object is not valid JSON:\n{url}\n{exc}"
        ) from exc


def load_mid_date_metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    section("LOADING MID_DATE ZARR METADATA")

    zarray_url = f"{TIME_ZARR_BASE}/.zarray"
    zattrs_url = f"{TIME_ZARR_BASE}/.zattrs"

    print(f".zarray: {zarray_url}")
    zarray = load_json(zarray_url)

    print(f".zattrs: {zattrs_url}")
    zattrs = load_json(zattrs_url)

    print()
    print(f"shape:      {zarray.get('shape')}")
    print(f"chunks:     {zarray.get('chunks')}")
    print(f"dtype:      {zarray.get('dtype')}")
    print(f"compressor: {zarray.get('compressor')}")
    print(f"fill_value: {zarray.get('fill_value')}")
    print(f"attributes: {zattrs}")

    return zarray, zattrs


# ============================================================================
# MANIFEST
# ============================================================================


def load_manifest() -> pd.DataFrame:
    section("LOADING EXISTING MANIFEST")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Rows:     {len(manifest)}")
    print(f"Columns:  {list(manifest.columns)}")

    required_columns = {
        "year",
        "cube_index",
        "mid_date",
    }

    missing = required_columns - set(manifest.columns)

    if missing:
        raise ManifestValidationFailure(
            f"Manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    if len(manifest) != 38:
        raise ManifestValidationFailure(
            f"Expected the existing 38-row manifest, "
            f"but found {len(manifest)} rows."
        )

    try:
        manifest["mid_date"] = pd.to_datetime(
            manifest["mid_date"],
            errors="raise",
        )
    except Exception as exc:
        raise ManifestValidationFailure(
            f"Could not parse manifest mid_date values: {exc}"
        ) from exc

    try:
        manifest["cube_index"] = pd.to_numeric(
            manifest["cube_index"],
            errors="raise",
        ).astype("int64")
    except Exception as exc:
        raise ManifestValidationFailure(
            f"Could not parse cube_index values as integers: {exc}"
        ) from exc

    print()
    print(
        f"Manifest date range: "
        f"{manifest['mid_date'].min()} -> "
        f"{manifest['mid_date'].max()}"
    )

    print(
        f"Manifest cube-index range: "
        f"{manifest['cube_index'].min()} -> "
        f"{manifest['cube_index'].max()}"
    )

    return manifest


# ============================================================================
# ZARR DECODING
# ============================================================================


def decode_compressed_chunk(
    compressed: bytes,
    zarray: dict[str, Any],
) -> np.ndarray:

    compressor = zarray.get("compressor")

    if compressor is None:
        raise DecodeFailure(
            "mid_date .zarray has no compressor metadata."
        )

    compressor_id = compressor.get("id")

    if compressor_id == "blosc":

        try:
            import blosc
        except ImportError as exc:
            raise DecodeFailure(
                "python-blosc is required to decode this Zarr array.\n"
                "Install with: pip install blosc"
            ) from exc

        try:
            decoded = blosc.decompress(compressed)

        except Exception as exc:
            raise DecodeFailure(
                f"Blosc decompression failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    elif compressor_id in ("zlib", None):

        try:
            decoded = zlib.decompress(compressed)

        except Exception as exc:
            raise DecodeFailure(
                f"zlib decompression failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    else:
        raise DecodeFailure(
            f"Unsupported Zarr compressor: {compressor_id}"
        )

    try:
        dtype = np.dtype(zarray["dtype"])

        expected_values = int(
            np.prod(zarray["chunks"])
        )

        array = np.frombuffer(
            decoded,
            dtype=dtype,
        )

        if array.size != expected_values:
            raise DecodeFailure(
                "Decoded chunk has an unexpected number of values. "
                f"Expected={expected_values}, "
                f"actual={array.size}"
            )

        return array.reshape(
            tuple(zarray["chunks"])
        )

    except DecodeFailure:
        raise

    except Exception as exc:
        raise DecodeFailure(
            "Could not convert decoded bytes into the expected "
            f"Zarr array: {type(exc).__name__}: {exc}"
        ) from exc


# ============================================================================
# TIME DECODING
# ============================================================================


def decode_time_value(
    value: Any,
    zarray: dict[str, Any],
    zattrs: dict[str, Any],
) -> pd.Timestamp:

    try:
        value_array = np.asarray(value)

        # Native numpy datetime64.
        if np.issubdtype(
            value_array.dtype,
            np.datetime64,
        ):
            return pd.Timestamp(value)

        # Numeric time coordinate.
        units = (
            zattrs.get("units")
            or zarray.get("units")
            or "ns"
        )

        units_lower = str(units).lower()

        if "nanosecond" in units_lower or units_lower == "ns":
            return pd.to_datetime(
                int(value),
                unit="ns",
            )

        if "microsecond" in units_lower or units_lower == "us":
            return pd.to_datetime(
                int(value),
                unit="us",
            )

        if "millisecond" in units_lower or units_lower == "ms":
            return pd.to_datetime(
                int(value),
                unit="ms",
            )

        if "second" in units_lower or units_lower == "s":
            return pd.to_datetime(
                int(value),
                unit="s",
            )

        if "day" in units_lower:
            return pd.to_datetime(
                int(value),
                unit="D",
                origin="unix",
            )

        raise DecodeFailure(
            f"Unsupported mid_date units: {units}"
        )

    except DecodeFailure:
        raise

    except Exception as exc:
        raise DecodeFailure(
            f"Could not decode mid_date value {value!r}: {exc}"
        ) from exc


# ============================================================================
# MID_DATE LOOKUP
# ============================================================================


def get_actual_cube_date(
    cube_index: int,
    zarray: dict[str, Any],
    zattrs: dict[str, Any],
) -> tuple[pd.Timestamp, int, int]:

    shape = tuple(
        int(value)
        for value in zarray["shape"]
    )

    chunks = tuple(
        int(value)
        for value in zarray["chunks"]
    )

    if len(shape) != 1 or len(chunks) != 1:
        raise DecodeFailure(
            f"Expected 1-D mid_date array, "
            f"got shape={shape}, chunks={chunks}"
        )

    time_length = shape[0]
    chunk_size = chunks[0]

    if cube_index < 0 or cube_index >= time_length:
        raise ManifestValidationFailure(
            f"cube_index={cube_index} is outside "
            f"mid_date range [0, {time_length - 1}]"
        )

    chunk_index = cube_index // chunk_size
    offset = cube_index % chunk_size

    chunk_url = (
        f"{TIME_ZARR_BASE}/{chunk_index}"
    )

    print(
        f"    cube_index={cube_index} "
        f"-> chunk={chunk_index}, offset={offset}"
    )
    print(f"    URL: {chunk_url}")

    compressed = download_once(chunk_url)

    decoded = decode_compressed_chunk(
        compressed=compressed,
        zarray=zarray,
    )

    try:
        raw_value = decoded[offset]

    except Exception as exc:
        raise DecodeFailure(
            f"Could not read offset={offset} "
            f"from decoded mid_date chunk: {exc}"
        ) from exc

    actual_date = decode_time_value(
        value=raw_value,
        zarray=zarray,
        zattrs=zattrs,
    )

    return (
        actual_date,
        chunk_index,
        offset,
    )


# ============================================================================
# DUPLICATE INDEX FLAGGING
# ============================================================================


def add_duplicate_flags(
    mapping: pd.DataFrame,
) -> pd.DataFrame:

    duplicate_mask = mapping["cube_index"].duplicated(
        keep=False
    )

    mapping["duplicate_cube_index"] = duplicate_mask

    mapping["duplicate_group_size"] = (
        mapping.groupby("cube_index")["cube_index"]
        .transform("size")
    )

    return mapping


# ============================================================================
# NON-MONOTONIC INDEX FLAGGING
# ============================================================================


def add_non_monotonic_flags(
    mapping: pd.DataFrame,
) -> pd.DataFrame:

    mapping = mapping.sort_values(
        "manifest_mid_date"
    ).reset_index(drop=True)

    previous_index = (
        mapping["cube_index"].shift(1)
    )

    mapping["previous_cube_index"] = previous_index

    mapping["cube_index_decreased"] = (
        mapping["cube_index"] < previous_index
    )

    # A row is involved in a non-monotonic transition if either:
    #   current index < previous index
    # OR
    #   previous index > current index.
    #
    # The current row is the useful row to flag because it identifies
    # where chronological order first decreases.
    mapping["non_monotonic_cube_index"] = (
        mapping["cube_index_decreased"]
    )

    return mapping


# ============================================================================
# 2012 / INDEX 171 HIGHLIGHT
# ============================================================================


def highlight_2012(mapping: pd.DataFrame) -> None:

    subsection(
        "EXPLICIT 2012 / CUBE_INDEX 171 INVESTIGATION"
    )

    rows = mapping[
        (
            mapping["year"].astype(int)
            == SUSPICIOUS_YEAR
        )
        &
        (
            mapping["cube_index"].astype(int)
            == SUSPICIOUS_CUBE_INDEX
        )
    ]

    if rows.empty:
        print(
            "WARNING: No manifest row has both "
            "year=2012 and cube_index=171."
        )
        return

    for _, row in rows.iterrows():

        print()
        print("FOUND:")
        print(
            f"  manifest year:          {row['year']}"
        )
        print(
            f"  manifest timestamp:     {row['manifest_mid_date']}"
        )
        print(
            f"  manifest calendar date: "
            f"{row['manifest_calendar_date']}"
        )
        print(
            f"  cube_index:              "
            f"{row['cube_index']}"
        )
        print(
            f"  actual cube timestamp:   "
            f"{row['actual_cube_mid_date']}"
        )
        print(
            f"  actual calendar date:    "
            f"{row['actual_cube_calendar_date']}"
        )
        print(
            f"  calendar-date status:    "
            f"{row['date_status']}"
        )
        print(
            f"  duplicate index:         "
            f"{row['duplicate_cube_index']}"
        )
        print(
            f"  non-monotonic index:     "
            f"{row['non_monotonic_cube_index']}"
        )

        print()

        if row["date_status"] == "DATE_MATCH":
            print(
                "CONCLUSION FOR 2012/171: "
                "The manifest's calendar date matches "
                "the actual cube date at index 171."
            )
            print(
                "The unusual numeric index remains a separate "
                "ordering issue and should not be treated as a "
                "date mismatch."
            )

        elif row["date_status"] == "DATE_MISMATCH":
            print(
                "CONCLUSION FOR 2012/171: "
                "The manifest calendar date does NOT match "
                "the actual cube date at index 171."
            )


# ============================================================================
# MAIN MAPPING
# ============================================================================


def build_mapping(
    manifest: pd.DataFrame,
    zarray: dict[str, Any],
    zattrs: dict[str, Any],
) -> pd.DataFrame:

    section(
        "RETRIEVING ACTUAL MID_DATE FOR ALL 38 MANIFEST ROWS"
    )

    rows: list[dict[str, Any]] = []

    for position, (_, manifest_row) in enumerate(
        manifest.iterrows(),
        start=1,
    ):

        year = int(manifest_row["year"])
        cube_index = int(
            manifest_row["cube_index"]
        )
        manifest_timestamp = pd.Timestamp(
            manifest_row["mid_date"]
        )

        print()
        print(
            f"[{position}/38] "
            f"year={year} | "
            f"manifest={manifest_timestamp} | "
            f"cube_index={cube_index}"
        )

        actual_date, chunk_index, offset = (
            get_actual_cube_date(
                cube_index=cube_index,
                zarray=zarray,
                zattrs=zattrs,
            )
        )

        manifest_calendar_date = (
            manifest_timestamp.date()
        )

        actual_calendar_date = (
            actual_date.date()
        )

        if (
            manifest_calendar_date
            == actual_calendar_date
        ):
            date_status = "DATE_MATCH"
        else:
            date_status = "DATE_MISMATCH"

        print(
            f"    actual cube date: "
            f"{actual_date}"
        )

        print(
            f"    calendar comparison: "
            f"{date_status}"
        )

        rows.append(
            {
                "manifest_row": position,
                "year": year,
                "manifest_mid_date": manifest_timestamp,
                "manifest_calendar_date": (
                    manifest_calendar_date
                ),
                "cube_index": cube_index,
                "actual_cube_mid_date": actual_date,
                "actual_cube_calendar_date": (
                    actual_calendar_date
                ),
                "date_status": date_status,
                "time_chunk": chunk_index,
                "time_offset": offset,
            }
        )

    mapping = pd.DataFrame(rows)

    mapping = add_duplicate_flags(
        mapping
    )

    mapping = add_non_monotonic_flags(
        mapping
    )

    # Keep chronological manifest order.
    mapping = mapping.sort_values(
        "manifest_mid_date"
    ).reset_index(drop=True)

    return mapping


# ============================================================================
# SAVE
# ============================================================================


def save_mapping(
    mapping: pd.DataFrame,
) -> None:

    section("SAVING COMPLETE TEMPORAL MAPPING")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = OUTPUT_CSV.with_suffix(
        ".tmp.csv"
    )

    mapping.to_csv(
        temp_path,
        index=False,
    )

    temp_path.replace(
        OUTPUT_CSV
    )

    print(
        f"Saved: {OUTPUT_CSV}"
    )
    print(
        f"Rows:  {len(mapping)}"
    )


# ============================================================================
# SUMMARY
# ============================================================================


def print_summary(
    mapping: pd.DataFrame,
) -> None:

    section(
        "TEMPORAL MAPPING SUMMARY"
    )

    match_count = int(
        (
            mapping["date_status"]
            == "DATE_MATCH"
        ).sum()
    )

    mismatch_count = int(
        (
            mapping["date_status"]
            == "DATE_MISMATCH"
        ).sum()
    )

    duplicate_rows = int(
        mapping["duplicate_cube_index"].sum()
    )

    non_monotonic_rows = int(
        mapping["non_monotonic_cube_index"].sum()
    )

    print(
        f"Total manifest rows:          {len(mapping)}"
    )

    print(
        f"DATE_MATCH:                    {match_count}"
    )

    print(
        f"DATE_MISMATCH:                 {mismatch_count}"
    )

    print(
        f"Rows with duplicate index:     {duplicate_rows}"
    )

    print(
        f"Non-monotonic index transitions:"
        f" {non_monotonic_rows}"
    )

    print()

    if mismatch_count:
        subsection("DATE MISMATCHES")

        print(
            mapping.loc[
                mapping["date_status"]
                == "DATE_MISMATCH",
                [
                    "year",
                    "manifest_mid_date",
                    "cube_index",
                    "actual_cube_mid_date",
                    "date_status",
                ],
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "All 38 manifest rows match the actual "
            "cube calendar date."
        )

    if duplicate_rows:
        subsection(
            "DUPLICATE CUBE INDICES"
        )

        print(
            mapping.loc[
                mapping["duplicate_cube_index"],
                [
                    "year",
                    "manifest_mid_date",
                    "cube_index",
                ],
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "No duplicate cube indices found."
        )

    if non_monotonic_rows:
        subsection(
            "NON-MONOTONIC CUBE-INDEX TRANSITIONS"
        )

        print(
            mapping.loc[
                mapping["non_monotonic_cube_index"],
                [
                    "year",
                    "manifest_mid_date",
                    "cube_index",
                    "previous_cube_index",
                    "non_monotonic_cube_index",
                ],
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "No non-monotonic cube-index transitions found."
        )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    section(
        "ITS_LIVE MANIFEST -> MID_DATE TEMPORAL DIAGNOSTIC"
    )

    print(
        "This run is TEMPORAL ONLY."
    )
    print(
        "No velocity (`v`) chunks will be downloaded."
    )
    print(
        "No Shishper spatial chunks will be inspected."
    )
    print(
        "All 38 manifest rows will be checked."
    )
    print(
        "Date comparison uses CALENDAR DATE only."
    )

    # ------------------------------------------------------------------
    # 1. Load the existing 38-row manifest.
    # ------------------------------------------------------------------

    manifest = load_manifest()

    # ------------------------------------------------------------------
    # 2. Load mid_date metadata.
    # ------------------------------------------------------------------

    zarray, zattrs = load_mid_date_metadata()

    # ------------------------------------------------------------------
    # 3. Validate that the claimed indices fit the mid_date array.
    # ------------------------------------------------------------------

    time_length = int(
        zarray["shape"][0]
    )

    print()
    print(
        f"mid_date time dimension: "
        f"{time_length}"
    )

    out_of_range = manifest[
        (
            manifest["cube_index"] < 0
        )
        |
        (
            manifest["cube_index"]
            >= time_length
        )
    ]

    if not out_of_range.empty:
        print()
        print(
            "OUT-OF-RANGE CUBE INDICES:"
        )
        print(
            out_of_range[
                [
                    "year",
                    "mid_date",
                    "cube_index",
                ]
            ].to_string(
                index=False
            )
        )

        raise ManifestValidationFailure(
            "Manifest contains cube indices outside "
            "the mid_date array."
        )

    # ------------------------------------------------------------------
    # 4. Retrieve actual date for ALL 38 rows.
    # ------------------------------------------------------------------

    mapping = build_mapping(
        manifest=manifest,
        zarray=zarray,
        zattrs=zattrs,
    )

    # ------------------------------------------------------------------
    # 5. Explicit 2012 / 171 investigation.
    # ------------------------------------------------------------------

    highlight_2012(
        mapping
    )

    # ------------------------------------------------------------------
    # 6. Save complete mapping.
    # ------------------------------------------------------------------

    save_mapping(
        mapping
    )

    # ------------------------------------------------------------------
    # 7. Print summary.
    # ------------------------------------------------------------------

    print_summary(
        mapping
    )

    section(
        "TEMPORAL DIAGNOSTIC COMPLETE"
    )

    print(
        "No velocity chunks were downloaded."
    )

    print(
        f"Complete mapping: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
