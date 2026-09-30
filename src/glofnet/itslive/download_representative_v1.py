"""
ITS_LIVE Representative Downloader V1

Purpose
-------
Download exactly one representative ITS_LIVE observation for the V1
smoke test.

Current smoke-test target:
    year       = 2004
    cube_index = 13191
    mid_date   = 2004-07-20 05:24:06.421783040

Important design decisions
--------------------------
1. The existing representative manifest is the source of truth.
2. The manifest is NOT modified.
3. Only the 2004 observation is selected in this smoke-test version.
4. The exact manifest cube index must be 13191.
5. The ITS_LIVE Zarr is accessed through direct HTTP requests.
6. xr.open_zarr() is intentionally NOT used because the current
   fsspec/xarray/Zarr combination incorrectly discovers this store
   as an empty Dataset.
7. The Zarr v2 consolidated metadata (.zmetadata) is read directly.
8. Only spatial chunks intersecting the Shishper glacier are downloaded.
9. Only the requested time index is extracted from those chunks.
10. The result is written as NetCDF.

Output
------
data/raw/itslive/temporal_v1/
    RGI2000-v7.0-G-14-08488_2004-07-20T05-24-06.nc
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
import time
from numcodecs import Blosc


# ============================================================================
# CONFIGURATION
# ============================================================================

GLACIER_ID = "RGI2000-v7.0-G-14-08488"

# Existing manifest -- DO NOT MODIFY.
MANIFEST_PATH = Path(
    "data/raw/itslive/temporal_v1/"
    f"{GLACIER_ID}_representative_dates_v1.csv"
)

OUTPUT_DIR = Path(
    "data/raw/itslive/temporal_v1"
)

# Exact smoke-test target.
SMOKE_TEST_YEAR = 2004
EXPECTED_CUBE_INDEX = 13191

EXPECTED_MID_DATE = pd.Timestamp(
    "2004-07-20 05:24:06.421783040"
)

EXPECTED_OUTPUT_NAME = (
    f"{GLACIER_ID}_2004-07-20T05-24-06.nc"
)

# --------------------------------------------------------------------------
# Shishper glacier bounds in EPSG:32643.
#
# These are the projected bounds used by the validator.
# --------------------------------------------------------------------------

GLACIER_BOUNDS_EPSG32643 = (
    458345.7236606957,   # minx
    4023078.799718212,   # miny
    471360.6348490992,   # maxx
    4038286.3500373554,  # maxy
)

# --------------------------------------------------------------------------
# Working ITS_LIVE Zarr URL.
# --------------------------------------------------------------------------

ZARR_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/"
    "N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)

# HTTP timeout.
HTTP_TIMEOUT = 120

# HTTP session.
SESSION = requests.Session()
CHUNK_CACHE = {}
SESSION.headers.update({
    "User-Agent": "Cryofusion-ITS-LIVE-Downloader/1.0",
    "Accept": "*/*",
})

adapter = requests.adapters.HTTPAdapter(
    pool_connections=2,
    pool_maxsize=2,
    max_retries=0,
)

SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)

# ITS_LIVE velocity variables to extract.
VELOCITY_VARIABLES = (
    "v",
    "vx",
    "vy",
    "v_error",
    "interp_mask",
)


# ============================================================================
# HTTP HELPERS
# ============================================================================

def http_get(
    url,
    *,
    timeout=120,
    max_retries=8,
    retry_delay=2.0,
):
    """
    Robust synchronous HTTP GET for ITS_LIVE S3/Zarr objects.

    Handles transient connection failures such as:
      - RemoteDisconnected
      - ConnectionError
      - ReadTimeout
      - HTTP 429
      - HTTP 500/502/503/504

    Uses the shared requests.Session so TCP connections can be reused.
    """

    last_error = None

    for attempt in range(1, max_retries + 1):

        try:
            response = SESSION.get(
                url,
                timeout=timeout,
                stream=False,
            )

            # Retry transient HTTP failures.
            if response.status_code in {429, 500, 502, 503, 504}:

                retry_after = response.headers.get("Retry-After")

                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = retry_delay
                else:
                    delay = retry_delay * attempt

                print(
                    f"HTTP {response.status_code}; "
                    f"retry {attempt}/{max_retries} "
                    f"after {delay:.1f}s"
                )

                response.close()
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response.content

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:

            last_error = exc

            if attempt == max_retries:
                break

            delay = retry_delay * attempt

            print(
                f"HTTP connection failure; "
                f"retry {attempt}/{max_retries} "
                f"after {delay:.1f}s"
            )
            print(f"Reason: {type(exc).__name__}: {exc}")

            time.sleep(delay)

        except Exception as exc:
            last_error = exc
            raise

    raise RuntimeError(
        f"HTTP GET failed after {max_retries} attempts:\n"
        f"{url}\n"
        f"Last error: {last_error}"
    )
def get_json(url: str) -> dict:
    """
    Download and decode a JSON object over HTTP.

    http_get() returns raw response bytes.
    """

    response_bytes = http_get(url)

    try:
        return json.loads(
            response_bytes.decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Response was not valid JSON:\n{url}"
        ) from exc

# ============================================================================
# GLACIER / MANIFEST
# ============================================================================

def load_glacier():
    """
    Confirm the glacier bounds being used.

    The actual RGI geometry loading is intentionally not required here because
    the V1 validator already establishes the projected glacier bounds and
    those bounds are fixed for this downloader.
    """

    print("\n===== LOADING GLACIER =====")
    print(f"Glacier: {GLACIER_ID}")

    minx, miny, maxx, maxy = GLACIER_BOUNDS_EPSG32643

    print("Bounds:")
    print(f"minx: {minx}")
    print(f"miny: {miny}")
    print(f"maxx: {maxx}")
    print(f"maxy: {maxy}")


def load_manifest() -> pd.DataFrame:
    """
    Load the existing representative manifest unchanged.
    """

    print("\n===== LOADING MANIFEST =====")
    print(f"Manifest: {MANIFEST_PATH}")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Representative manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    print("Manifest columns:")
    print(list(manifest.columns))

    print(
        f"Manifest observations: {len(manifest)}"
    )

    required_columns = {
        "year",
        "cube_index",
        "mid_date",
    }

    missing = required_columns - set(manifest.columns)

    if missing:
        raise RuntimeError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    return manifest


def select_smoke_test_observation(
    manifest: pd.DataFrame,
) -> pd.Series:
    """
    Select exactly the 2004 observation from the existing manifest.

    The manifest is not modified.

    The exact cube index is checked against 13191.
    """

    print("\n===== SMOKE TEST =====")
    print(f"Smoke-test year: {SMOKE_TEST_YEAR}")

    selected = manifest[
        manifest["year"].astype(int) == SMOKE_TEST_YEAR
    ].copy()

    print(
        f"Observations selected: {len(selected)}"
    )

    if len(selected) != 1:
        raise RuntimeError(
            f"Expected exactly one observation for "
            f"year {SMOKE_TEST_YEAR}, "
            f"found {len(selected)}."
        )

    row = selected.iloc[0]

    cube_index = int(row["cube_index"])

    manifest_mid_date = pd.to_datetime(
        row["mid_date"]
    )

    print("\nSelected observation:")
    print(
        f"year={int(row['year'])} | "
        f"cube index={cube_index} | "
        f"date={manifest_mid_date}"
    )

    if cube_index != EXPECTED_CUBE_INDEX:
        raise RuntimeError(
            "Manifest cube index does not match the required "
            "smoke-test cube index.\n"
            f"Expected: {EXPECTED_CUBE_INDEX}\n"
            f"Found:    {cube_index}"
        )

    if manifest_mid_date != EXPECTED_MID_DATE:
        raise RuntimeError(
            "Manifest mid_date does not match the required "
            "smoke-test date.\n"
            f"Expected: {EXPECTED_MID_DATE}\n"
            f"Found:    {manifest_mid_date}"
        )

    print("\nSmoke-test selection checks: PASS")

    return row


# ============================================================================
# ZARR METADATA
# ============================================================================

def test_zarr_http_access():
    """
    Confirm that the direct synchronous HTTP access mechanism works.

    http_get() returns raw bytes, not a requests.Response object.
    """

    print("\n===== TESTING ITS_LIVE HTTP ACCESS =====")
    print("Access method: synchronous requests")

    zgroup_url = f"{ZARR_URL}/.zgroup"

    print("Testing .zgroup...")
    print(zgroup_url)

    response_bytes = http_get(zgroup_url)

    print(".zgroup request: PASS")

    try:
        zgroup = json.loads(response_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "Failed to decode ITS_LIVE .zgroup response as JSON."
        ) from exc

    print(json.dumps(zgroup, indent=4))

    if zgroup.get("zarr_format") != 2:
        raise RuntimeError(
            f"Expected Zarr v2 store, found: "
            f"{zgroup.get('zarr_format')}"
        )

    print("Zarr format: 2")

def load_zarr_metadata() -> dict:
    """
    Read consolidated Zarr v2 metadata directly.

    This avoids xr.open_zarr() discovery.
    """

    print("\n===== LOADING ZARR METADATA =====")

    metadata_url = f"{ZARR_URL}/.zmetadata"

    print(metadata_url)

    metadata_document = get_json(metadata_url)

    if "metadata" not in metadata_document:
        raise RuntimeError(
            "ITS_LIVE .zmetadata does not contain "
            "the expected 'metadata' object."
        )

    metadata = metadata_document["metadata"]

    print(
        f"Metadata entries: {len(metadata)}"
    )

    required = [
        ".zgroup",
        ".zattrs",
        "mid_date/.zarray",
        "mid_date/.zattrs",
        "x/.zarray",
        "x/.zattrs",
        "y/.zarray",
        "y/.zattrs",
    ]

    for variable in VELOCITY_VARIABLES:
        required.append(f"{variable}/.zarray")
        required.append(f"{variable}/.zattrs")

    missing = [
        key for key in required
        if key not in metadata
    ]

    if missing:
        raise RuntimeError(
            "Required Zarr metadata entries are missing:\n"
            + "\n".join(missing)
        )

    return metadata


def print_zarr_structure(metadata: dict):
    """
    Print the relevant Zarr structure.
    """

    print("\n===== ZARR STRUCTURE =====")

    for name in (
        "mid_date",
        "x",
        "y",
        "v",
        "vx",
        "vy",
        "v_error",
        "interp_mask",
    ):
        array_meta = metadata[f"{name}/.zarray"]

        print(
            f"{name}: "
            f"shape={array_meta['shape']} | "
            f"chunks={array_meta['chunks']} | "
            f"dtype={array_meta['dtype']}"
        )


# ============================================================================
# ZARR CHUNK DECODING
# ============================================================================

def decode_blosc(
    compressed: bytes,
    compressor_metadata: dict,
) -> bytes:
    """
    Decode a Zarr v2 Blosc-compressed chunk.
    """

    if compressor_metadata is None:
        return compressed

    compressor_id = compressor_metadata.get("id")

    if compressor_id != "blosc":
        raise RuntimeError(
            "Unsupported Zarr compressor: "
            f"{compressor_id}"
        )

    decoder = Blosc(
        cname=compressor_metadata["cname"],
        clevel=compressor_metadata["clevel"],
        shuffle=compressor_metadata["shuffle"],
        blocksize=compressor_metadata.get(
            "blocksize",
            0,
        ),
    )

    return decoder.decode(compressed)


def read_zarr_chunk(
    array_name: str,
    chunk_key: str,
    metadata: dict,
) -> np.ndarray:
    """
    Download and decode one Zarr v2 chunk.

    Parameters
    ----------
    array_name : str
        Zarr array name, e.g. "x", "y", "v", "vx", "vy".

    chunk_key : str
        Zarr chunk key, e.g. "0" for a 1-D coordinate chunk or
        "0.51.48" for a 3-D velocity chunk.

    metadata : dict
        Parsed Zarr .zmetadata["metadata"] dictionary.

    Returns
    -------
    np.ndarray
        Decoded chunk values.
    """

    # ---------------------------------------------------------
    # Build the chunk URL FIRST
    # ---------------------------------------------------------
    chunk_url = (
        f"{ZARR_URL}/"
        f"{array_name}/"
        f"{chunk_key}"
    )

    # ---------------------------------------------------------
    # Persistent cache
    #
    # Do NOT create {} inside this function because that would
    # clear the cache every time the function is called.
    # ---------------------------------------------------------
    global CHUNK_CACHE

    if chunk_url in CHUNK_CACHE:
        return CHUNK_CACHE[chunk_url]

    # ---------------------------------------------------------
    # Get array metadata
    # ---------------------------------------------------------
    metadata_key = f"{array_name}/.zarray"

    if metadata_key not in metadata:
        raise KeyError(
            f"Missing Zarr metadata for array: {array_name}"
        )

    array_meta = metadata[metadata_key]

    # ---------------------------------------------------------
    # Download raw chunk bytes
    #
    # http_get() returns bytes in this script.
    # ---------------------------------------------------------
    raw_response = http_get(chunk_url)

    # Be gentle with the S3 endpoint.
    time.sleep(0.15)

    # ---------------------------------------------------------
    # Decode Blosc-compressed Zarr chunk
    # ---------------------------------------------------------
    raw = decode_blosc(
        raw_response,
        array_meta.get("compressor"),
    )

    # ---------------------------------------------------------
    # Convert raw bytes into NumPy values
    # ---------------------------------------------------------
    dtype = np.dtype(
        array_meta["dtype"]
    )

    values = np.frombuffer(
        raw,
        dtype=dtype,
    )

    # ---------------------------------------------------------
    # Cache the decoded NumPy array
    # ---------------------------------------------------------
    CHUNK_CACHE[chunk_url] = values

    return values


# ============================================================================
# COORDINATES
# ============================================================================

def read_1d_coordinate(
    name: str,
    metadata: dict,
) -> np.ndarray:
    """
    Read a 1-D Zarr coordinate.

    x and y each have one 833-element chunk.
    """

    array_meta = metadata[f"{name}/.zarray"]

    shape = array_meta["shape"]
    chunks = array_meta["chunks"]

    if len(shape) != 1:
        raise RuntimeError(
            f"{name} is not one-dimensional: {shape}"
        )

    n = shape[0]
    chunk_size = chunks[0]

    result = np.empty(
        n,
        dtype=np.dtype(
            array_meta["dtype"]
        ),
    )

    for start in range(
        0,
        n,
        chunk_size,
    ):

        chunk_index = start // chunk_size

        values = read_zarr_chunk(
            name,
            str(chunk_index),
            metadata,
        )

        end = min(
            start + chunk_size,
            n,
        )

        result[start:end] = values[
            : end - start
        ]

    return result


def find_spatial_indices(
    x: np.ndarray,
    y: np.ndarray,
):
    """
    Find the exact x/y coordinate indices intersecting the glacier bounds.
    """

    minx, miny, maxx, maxy = (
        GLACIER_BOUNDS_EPSG32643
    )

    x_indices = np.where(
        (x >= minx) &
        (x <= maxx)
    )[0]

    y_indices = np.where(
        (y >= miny) &
        (y <= maxy)
    )[0]

    if len(x_indices) == 0:
        raise RuntimeError(
            "No X coordinates overlap the glacier."
        )

    if len(y_indices) == 0:
        raise RuntimeError(
            "No Y coordinates overlap the glacier."
        )

    return x_indices, y_indices


# ============================================================================
# MID DATE
# ============================================================================

def read_mid_date_value(
    cube_index: int,
    metadata: dict,
) -> float:
    """
    Read exactly one mid_date value.

    ITS_LIVE metadata says:

        units = days since 1970-01-01
    """

    array_meta = metadata[
        "mid_date/.zarray"
    ]

    chunk_size = array_meta["chunks"][0]

    chunk_index = cube_index // chunk_size
    offset = cube_index % chunk_size

    values = read_zarr_chunk(
        "mid_date",
        str(chunk_index),
        metadata,
    )

    if offset >= len(values):
        raise RuntimeError(
            f"mid_date offset {offset} is outside "
            f"chunk of size {len(values)}."
        )

    return float(values[offset])


def decode_mid_date(
    value: float,
) -> pd.Timestamp:
    """
    Convert ITS_LIVE CF time value to pandas Timestamp.
    """

    epoch = pd.Timestamp(
        "1970-01-01",
        tz=None,
    )

    return epoch + pd.to_timedelta(
        value,
        unit="D",
    )


# ============================================================================
# OBSERVATION DATE VALIDATION
# ============================================================================

def validate_observation_date(
    cube_index: int,
    manifest_mid_date: pd.Timestamp,
    metadata: dict,
) -> pd.Timestamp:
    """
    Confirm that the requested cube index corresponds to the
    manifest observation date.
    """

    print("\n===== OBSERVATION DATE CHECK =====")

    raw_value = read_mid_date_value(
        cube_index,
        metadata,
    )

    zarr_mid_date = decode_mid_date(
        raw_value
    )

    print(
        f"Cube index:     {cube_index}"
    )
    print(
        f"Manifest date:  {manifest_mid_date}"
    )
    print(
        f"Zarr date:      {zarr_mid_date}"
    )

    difference = abs(
        (
            zarr_mid_date -
            manifest_mid_date
        ).total_seconds()
    )

    print(
        f"Date difference: {difference} seconds"
    )

    if difference > 1.0:
        raise RuntimeError(
            "Manifest and Zarr observation dates "
            "do not match."
        )

    print("Observation date: PASS")

    return zarr_mid_date


# ============================================================================
# SPATIAL CHUNK EXTRACTION
# ============================================================================

def extract_variable_observation(
    variable: str,
    cube_index: int,
    x_indices: np.ndarray,
    y_indices: np.ndarray,
    metadata: dict,
    progress_number: int,
    progress_total: int,
) -> np.ndarray:
    """
    Extract one observation from a 3-D Zarr array.

    Zarr layout:

        (mid_date, y, x)

    Chunk layout:

        (20000, 10, 10)

    Therefore the requested cube index lives in one time chunk,
    but each spatial 10x10 chunk must be downloaded in full because
    Zarr stores the compressed chunk as the HTTP object.
    """

    array_meta = metadata[
        f"{variable}/.zarray"
    ]

    dtype = np.dtype(
        array_meta["dtype"]
    )

    fill_value = array_meta.get(
        "fill_value"
    )

    missing_value = (
        metadata[f"{variable}/.zattrs"]
        .get("missing_value")
    )

    time_chunk_size = array_meta[
        "chunks"
    ][0]

    y_chunk_size = array_meta[
        "chunks"
    ][1]

    x_chunk_size = array_meta[
        "chunks"
    ][2]

    time_chunk = (
        cube_index //
        time_chunk_size
    )

    time_offset = (
        cube_index %
        time_chunk_size
    )

    y_chunk_first = (
        int(y_indices[0]) //
        y_chunk_size
    )

    y_chunk_last = (
        int(y_indices[-1]) //
        y_chunk_size
    )

    x_chunk_first = (
        int(x_indices[0]) //
        x_chunk_size
    )

    x_chunk_last = (
        int(x_indices[-1]) //
        x_chunk_size
    )

    number_of_chunks = (
        (y_chunk_last - y_chunk_first + 1)
        *
        (x_chunk_last - x_chunk_first + 1)
    )

    print(
        f"\n----- VARIABLE: {variable} "
        f"({progress_number}/{progress_total}) -----"
    )

    print(
        f"cube index: {cube_index}"
    )

    print(
        f"time chunk: {time_chunk}"
    )

    print(
        f"time offset: {time_offset}"
    )

    print(
        f"Y chunks: "
        f"{y_chunk_first} -> {y_chunk_last}"
    )

    print(
        f"X chunks: "
        f"{x_chunk_first} -> {x_chunk_last}"
    )

    print(
        f"Spatial chunks required: "
        f"{number_of_chunks}"
    )

    output = np.full(
        (
            len(y_indices),
            len(x_indices),
        ),
        np.nan,
        dtype=np.float32,
    )

    completed = 0

    for yc in range(
        y_chunk_first,
        y_chunk_last + 1,
    ):

        for xc in range(
            x_chunk_first,
            x_chunk_last + 1,
        ):

            chunk_key = (
                f"{time_chunk}."
                f"{yc}."
                f"{xc}"
            )

            values = read_zarr_chunk(
                variable,
                chunk_key,
                metadata,
            )

            chunk_y_size = y_chunk_size
            chunk_x_size = x_chunk_size

            expected_elements = (
                time_chunk_size
                *
                chunk_y_size
                *
                chunk_x_size
            )

            if values.size != expected_elements:
                raise RuntimeError(
                    f"Unexpected chunk size for "
                    f"{variable}/{chunk_key}.\n"
                    f"Expected: {expected_elements}\n"
                    f"Found:    {values.size}"
                )

            values = values.reshape(
                time_chunk_size,
                chunk_y_size,
                chunk_x_size,
            )

            observation = values[
                time_offset
            ]

            global_y_start = (
                yc * y_chunk_size
            )

            global_x_start = (
                xc * x_chunk_size
            )

            global_y_end = (
                global_y_start +
                chunk_y_size
            )

            global_x_end = (
                global_x_start +
                chunk_x_size
            )

            selected_y = np.where(
                (y_indices >= global_y_start)
                &
                (y_indices < global_y_end)
            )[0]

            selected_x = np.where(
                (x_indices >= global_x_start)
                &
                (x_indices < global_x_end)
            )[0]

            for output_y in selected_y:

                global_y = int(
                    y_indices[output_y]
                )

                local_y = (
                    global_y -
                    global_y_start
                )

                for output_x in selected_x:

                    global_x = int(
                        x_indices[output_x]
                    )

                    local_x = (
                        global_x -
                        global_x_start
                    )

                    value = observation[
                        local_y,
                        local_x,
                    ]

                    # ------------------------------------------------------
                    # Velocity variables
                    # ------------------------------------------------------

                    if variable != "interp_mask":

                        if missing_value is not None:
                            if value == missing_value:
                                output[
                                    output_y,
                                    output_x,
                                ] = np.nan
                                continue

                        if fill_value is not None:
                            if value == fill_value:
                                output[
                                    output_y,
                                    output_x,
                                ] = np.nan
                                continue

                        output[
                            output_y,
                            output_x,
                        ] = float(value)

                    # ------------------------------------------------------
                    # interp_mask
                    #
                    # 0 = measured
                    # 1 = interpolated
                    #
                    # Do NOT convert 0 to NaN.
                    # ------------------------------------------------------

                    else:

                        output[
                            output_y,
                            output_x,
                        ] = float(value)

            completed += 1

            print(
                f"  {completed}/{number_of_chunks} "
                f"chunks downloaded"
            )

    return output


# ============================================================================
# DATASET CONSTRUCTION
# ============================================================================

def build_dataset(
    x: np.ndarray,
    y: np.ndarray,
    zarr_mid_date: pd.Timestamp,
    cube_index: int,
    variable_data: dict[str, np.ndarray],
    metadata: dict,
) -> xr.Dataset:
    """
    Construct the final xarray Dataset representing one ITS_LIVE
    observation.
    """

    data_vars = {}

    for variable in VELOCITY_VARIABLES:

        data = variable_data[
            variable
        ]

        data_vars[variable] = (
            ("y", "x"),
            data.astype(
                np.float32,
                copy=False,
            ),
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "x": (
                "x",
                x.astype(
                    np.float64,
                    copy=False,
                ),
            ),
            "y": (
                "y",
                y.astype(
                    np.float64,
                    copy=False,
                ),
            ),
        },
    )

    # ------------------------------------------------------------------------
    # Preserve relevant ITS_LIVE root attributes.
    # ------------------------------------------------------------------------

    root_attrs = metadata.get(
        ".zattrs",
        {},
    )

    for key, value in root_attrs.items():
        ds.attrs[key] = value

    # ------------------------------------------------------------------------
    # Add observation-specific attributes used by V1 validation.
    # ------------------------------------------------------------------------

    ds.attrs[
        "glacier_id"
    ] = GLACIER_ID

    ds.attrs[
        "observation_date"
    ] = str(zarr_mid_date)

    ds.attrs[
        "cube_index"
    ] = int(cube_index)

    ds.attrs[
        "interp_mask_status"
    ] = "preserved_not_interpreted"

    # ------------------------------------------------------------------------
    # Add variable attributes from ITS_LIVE metadata.
    # ------------------------------------------------------------------------

    for variable in VELOCITY_VARIABLES:

        variable_attrs = metadata[
            f"{variable}/.zattrs"
        ]

        for key, value in variable_attrs.items():

            # _ARRAY_DIMENSIONS is Zarr-specific metadata and does not
            # need to be written as a NetCDF variable attribute.
            if key == "_ARRAY_DIMENSIONS":
                continue

            ds[variable].attrs[key] = value

    return ds


# ============================================================================
# NETCDF OUTPUT
# ============================================================================

def save_netcdf(
    ds: xr.Dataset,
    output_path: Path,
):
    """
    Save the extracted observation as NetCDF.
    """

    print("\n===== SAVING NETCDF =====")
    print(f"Output: {output_path}")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        print(
            "Existing output file found."
        )
        print(
            "It will be replaced."
        )

        output_path.unlink()

    ds.to_netcdf(
        output_path,
        engine="netcdf4",
    )

    print("NetCDF saved successfully.")

    size_mb = (
        output_path.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"File size: {size_mb:.2f} MB"
    )


# ============================================================================
# OUTPUT SANITY CHECK
# ============================================================================

def sanity_check_output(
    ds: xr.Dataset,
    expected_mid_date: pd.Timestamp,
):
    """
    Perform basic checks before declaring the downloader successful.
    """

    print("\n===== OUTPUT SANITY CHECK =====")

    expected_variables = set(
        VELOCITY_VARIABLES
    )

    actual_variables = set(
        ds.data_vars
    )

    missing = (
        expected_variables -
        actual_variables
    )

    if missing:
        raise RuntimeError(
            "Output dataset is missing variables:\n"
            + "\n".join(sorted(missing))
        )

    print(
        "Required variables: PASS"
    )

    if "x" not in ds.coords:
        raise RuntimeError(
            "Output is missing x coordinate."
        )

    if "y" not in ds.coords:
        raise RuntimeError(
            "Output is missing y coordinate."
        )

    print(
        "Coordinates: PASS"
    )

    if ds.sizes["x"] <= 0:
        raise RuntimeError(
            "Output has zero x dimension."
        )

    if ds.sizes["y"] <= 0:
        raise RuntimeError(
            "Output has zero y dimension."
        )

    print(
        f"Dimensions: PASS "
        f"(y={ds.dims['y']}, x={ds.dims['x']})"
    )

    output_date = pd.Timestamp(
        ds.attrs["observation_date"]
    )

    difference = abs(
        (
            output_date -
            expected_mid_date
        ).total_seconds()
    )

    print(
        f"Observation date difference: "
        f"{difference} seconds"
    )

    if difference > 1:
        raise RuntimeError(
            "Output observation date does not "
            "match the manifest."
        )

    print(
        "Observation date: PASS"
    )

    # ------------------------------------------------------------------------
    # Spatial overlap.
    # ------------------------------------------------------------------------

    minx, miny, maxx, maxy = (
        GLACIER_BOUNDS_EPSG32643
    )

    data_minx = float(
        ds.x.min()
    )

    data_maxx = float(
        ds.x.max()
    )

    data_miny = float(
        ds.y.min()
    )

    data_maxy = float(
        ds.y.max()
    )

    x_overlap = (
        data_maxx >= minx
        and
        data_minx <= maxx
    )

    y_overlap = (
        data_maxy >= miny
        and
        data_miny <= maxy
    )

    print(
        f"X overlap: {x_overlap}"
    )

    print(
        f"Y overlap: {y_overlap}"
    )

    if not x_overlap or not y_overlap:
        raise RuntimeError(
            "Output does not spatially overlap "
            "the Shishper glacier."
        )

    print(
        "Spatial overlap: PASS"
    )

    print(
        "\nOutput sanity checks: PASS"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 55)
    print(
        "       ITS_LIVE REPRESENTATIVE DOWNLOADER V1"
    )
    print("=" * 55)

    # ------------------------------------------------------------------------
    # 1. Glacier
    # ------------------------------------------------------------------------

    load_glacier()

    # ------------------------------------------------------------------------
    # 2. Existing manifest
    # ------------------------------------------------------------------------

    manifest = load_manifest()

    # ------------------------------------------------------------------------
    # 3. Select ONLY 2004
    # ------------------------------------------------------------------------

    row = select_smoke_test_observation(
        manifest
    )

    cube_index = int(
        row["cube_index"]
    )

    manifest_mid_date = pd.to_datetime(
        row["mid_date"]
    )

    # ------------------------------------------------------------------------
    # 4. HTTP access
    # ------------------------------------------------------------------------

    test_zarr_http_access()

    # ------------------------------------------------------------------------
    # 5. Read Zarr metadata
    # ------------------------------------------------------------------------

    metadata = load_zarr_metadata()

    print_zarr_structure(
        metadata
    )

    # ------------------------------------------------------------------------
    # 6. Read coordinates
    # ------------------------------------------------------------------------

    print("\n===== READING ZARR COORDINATES =====")

    x = read_1d_coordinate(
        "x",
        metadata,
    )

    y = read_1d_coordinate(
        "y",
        metadata,
    )

    print(
        f"x coordinate count: {len(x)}"
    )

    print(
        f"y coordinate count: {len(y)}"
    )

    print(
        f"x range: {x.min()} -> {x.max()}"
    )

    print(
        f"y range: {y.min()} -> {y.max()}"
    )

    # ------------------------------------------------------------------------
    # 7. Find glacier spatial indices
    # ------------------------------------------------------------------------

    print("\n===== SPATIAL SUBSET =====")

    x_indices, y_indices = (
        find_spatial_indices(
            x,
            y,
        )
    )

    x_subset = x[
        x_indices
    ]

    y_subset = y[
        y_indices
    ]

    print(
        f"x indices: "
        f"{x_indices[0]} -> {x_indices[-1]}"
    )

    print(
        f"y indices: "
        f"{y_indices[0]} -> {y_indices[-1]}"
    )

    print(
        f"Output dimensions: "
        f"y={len(y_indices)}, "
        f"x={len(x_indices)}"
    )

    print(
        f"Output X extent: "
        f"{x_subset.min()} -> {x_subset.max()}"
    )

    print(
        f"Output Y extent: "
        f"{y_subset.min()} -> {y_subset.max()}"
    )

    # ------------------------------------------------------------------------
    # 8. Validate cube index against Zarr mid_date
    # ------------------------------------------------------------------------

    zarr_mid_date = (
        validate_observation_date(
            cube_index,
            manifest_mid_date,
            metadata,
        )
    )

    # ------------------------------------------------------------------------
    # 9. Extract the one observation
    # ------------------------------------------------------------------------

    print("\n===== EXTRACTING OBSERVATION =====")

    print(
        f"Year:       {SMOKE_TEST_YEAR}"
    )

    print(
        f"Cube index: {cube_index}"
    )

    print(
        f"Date:       {zarr_mid_date}"
    )

    variable_data = {}

    total_variables = len(
        VELOCITY_VARIABLES
    )

    for number, variable in enumerate(
        VELOCITY_VARIABLES,
        start=1,
    ):

        variable_data[variable] = (
            extract_variable_observation(
                variable=variable,
                cube_index=cube_index,
                x_indices=x_indices,
                y_indices=y_indices,
                metadata=metadata,
                progress_number=number,
                progress_total=total_variables,
            )
        )

    # ------------------------------------------------------------------------
    # 10. Build Dataset
    # ------------------------------------------------------------------------

    print("\n===== BUILDING DATASET =====")

    ds = build_dataset(
        x=x_subset,
        y=y_subset,
        zarr_mid_date=zarr_mid_date,
        cube_index=cube_index,
        variable_data=variable_data,
        metadata=metadata,
    )

    print(ds)

    # ------------------------------------------------------------------------
    # 11. Sanity check before writing
    # ------------------------------------------------------------------------

    sanity_check_output(
        ds,
        expected_mid_date=manifest_mid_date,
    )

    # ------------------------------------------------------------------------
    # 12. Save exact requested filename
    # ------------------------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        EXPECTED_OUTPUT_NAME
    )

    save_netcdf(
        ds,
        output_path,
    )

    # ------------------------------------------------------------------------
    # 13. Close dataset
    # ------------------------------------------------------------------------

    ds.close()

    # ------------------------------------------------------------------------
    # 14. STOP.
    #
    # This V1 smoke-test downloader deliberately does not process 1988,
    # 1995, 2015, or 2025.
    # ------------------------------------------------------------------------

    print("\n" + "=" * 55)
    print(
        "SMOKE TEST DOWNLOAD COMPLETE"
    )
    print("=" * 55)

    print(
        f"Year:       {SMOKE_TEST_YEAR}"
    )

    print(
        f"Cube index: {cube_index}"
    )

    print(
        f"Date:       {manifest_mid_date}"
    )

    print(
        f"Output:     {output_path}"
    )

    print(
        "\nOnly the 2004 observation was processed."
    )

    print(
        "Downloader stopped after one observation."
    )


if __name__ == "__main__":
    main()