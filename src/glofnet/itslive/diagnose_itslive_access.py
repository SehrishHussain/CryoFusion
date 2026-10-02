
"""
ITS_LIVE single-observation network/data-access diagnostic.

Purpose
-------
Diagnose why direct ITS_LIVE Zarr chunk downloads are failing before
running the full temporal coverage inspection.

Tests:
1. DNS / hostname resolution
2. Basic HTTPS access to ITS_LIVE S3
3. Python urllib access
4. Python requests access (if installed)
5. Direct access to Zarr metadata
6. Direct access to one velocity Zarr chunk
7. Decode/decompress the velocity chunk
8. Inspect the 2004 observation contained in that chunk

Important
---------
This script intentionally downloads ONLY:
    - a few small metadata objects
    - one velocity chunk

It does NOT download the complete Shishper spatial subset.

Known 2004 observation:
    cube_index = 13191
    mid_date   = 2004-07-20 05:24:06.421783040

Known chunk containing the Shishper spatial subset:
    time chunk = 0
    Y chunk    = 51
    X chunk    = 48

Therefore:
    v/0.51.48

This diagnostic is designed to distinguish:

    DNS_FAILURE
    URLLIB_FAILURE
    REQUESTS_FAILURE
    HTTP_FAILURE
    ZARR_METADATA_FAILURE
    CHUNK_DOWNLOAD_FAILURE
    CHUNK_DECODE_FAILURE
    CHUNK_ACCESS_OK

from actual data contents such as:

    ZERO_VALID_IN_CHUNK
    VALID_DATA_IN_CHUNK

The script does NOT interpret failure to access the chunk as missing
velocity data.
"""

from __future__ import annotations

import json
import socket
import struct
import sys
import time
import zlib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

ZARR_BASE = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/"
    "N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)

VARIABLE = "v"

# Known 2004 observation
CUBE_INDEX_2004 = 13191
OBSERVATION_DATE_2004 = "2004-07-20 05:24:06.421783040"

# The Shishper subset previously identified:
Y_START = 514
Y_END = 639

X_START = 486
X_END = 593

# Chunk dimensions:
TIME_CHUNK_SIZE = 20000
Y_CHUNK_SIZE = 10
X_CHUNK_SIZE = 10

# Known first spatial chunk intersecting the Shishper subset.
TIME_CHUNK = CUBE_INDEX_2004 // TIME_CHUNK_SIZE
Y_CHUNK = Y_START // Y_CHUNK_SIZE
X_CHUNK = X_START // X_CHUNK_SIZE

CHUNK_URL = (
    f"{ZARR_BASE}/{VARIABLE}/"
    f"{TIME_CHUNK}.{Y_CHUNK}.{X_CHUNK}"
)

ZARRAY_URL = f"{ZARR_BASE}/{VARIABLE}/.zarray"
ZATTRS_URL = f"{ZARR_BASE}/{VARIABLE}/.zattrs"

HOSTNAME = "its-live-data.s3.amazonaws.com"

MAX_RETRIES = 2
REQUEST_TIMEOUT = 15

OUTPUT_DIR = Path("data/raw/itslive/diagnostics")
OUTPUT_FILE = OUTPUT_DIR / "itslive_access_diagnostic_2004.json"


# ---------------------------------------------------------------------
# OUTPUT HELPERS
# ---------------------------------------------------------------------

results: dict = {
    "diagnostic": "ITS_LIVE single-observation access diagnostic",
    "observation": {
        "year": 2004,
        "cube_index": CUBE_INDEX_2004,
        "mid_date": OBSERVATION_DATE_2004,
    },
    "zarr": ZARR_BASE,
    "chunk": {
        "time_chunk": TIME_CHUNK,
        "y_chunk": Y_CHUNK,
        "x_chunk": X_CHUNK,
        "url": CHUNK_URL,
    },
    "tests": [],
}


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


def record_test(
    name: str,
    status: str,
    message: str = "",
    **extra,
) -> None:
    entry = {
        "test": name,
        "status": status,
        "message": message,
    }

    entry.update(extra)

    results["tests"].append(entry)


# ---------------------------------------------------------------------
# DNS TEST
# ---------------------------------------------------------------------

def test_dns() -> bool:
    subsection("TEST 1: DNS RESOLUTION")

    print(f"Hostname: {HOSTNAME}")
    print("Resolving hostname...")

    try:
        addresses = socket.getaddrinfo(
            HOSTNAME,
            443,
            type=socket.SOCK_STREAM,
        )

        unique_addresses = sorted(
            {
                item[4][0]
                for item in addresses
                if item[4]
            }
        )

        print("DNS resolution: SUCCESS")
        print("Resolved addresses:")

        for address in unique_addresses:
            print(f"  {address}")

        record_test(
            "dns_resolution",
            "SUCCESS",
            "Hostname resolved successfully.",
            hostname=HOSTNAME,
            addresses=unique_addresses,
        )

        return True

    except Exception as exc:
        print("DNS resolution: FAILURE")
        print(f"Error: {exc}")

        record_test(
            "dns_resolution",
            "DNS_FAILURE",
            str(exc),
            hostname=HOSTNAME,
        )

        return False


# ---------------------------------------------------------------------
# SOCKET / TCP TEST
# ---------------------------------------------------------------------

def test_tcp() -> bool:
    subsection("TEST 2: TCP CONNECTION")

    print(f"Host: {HOSTNAME}")
    print("Port: 443")
    print("Timeout:", REQUEST_TIMEOUT, "seconds")

    try:
        start = time.perf_counter()

        with socket.create_connection(
            (HOSTNAME, 443),
            timeout=REQUEST_TIMEOUT,
        ):
            elapsed = time.perf_counter() - start

        print("TCP connection: SUCCESS")
        print(f"Connection time: {elapsed:.3f} seconds")

        record_test(
            "tcp_connection",
            "SUCCESS",
            "TCP connection to port 443 succeeded.",
            elapsed_seconds=elapsed,
        )

        return True

    except Exception as exc:
        print("TCP connection: FAILURE")
        print(f"Error: {exc}")

        record_test(
            "tcp_connection",
            "TCP_FAILURE",
            str(exc),
        )

        return False


# ---------------------------------------------------------------------
# URLLIB TEST
# ---------------------------------------------------------------------

def urllib_get(
    url: str,
    *,
    timeout: int = REQUEST_TIMEOUT,
) -> tuple[bytes, int, dict]:

    request = Request(
        url,
        headers={
            "User-Agent": (
                "CryoFusion-ITS-LIVE-Diagnostic/1.0"
            )
        },
    )

    with urlopen(request, timeout=timeout) as response:
        data = response.read()

        status = getattr(response, "status", None)

        headers = {
            key: value
            for key, value in response.headers.items()
        }

        return data, status, headers


def test_urllib_metadata() -> bool:
    subsection("TEST 3: PYTHON URLLIB -> ZARR METADATA")

    print("URL:")
    print(ZARRAY_URL)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"Attempt {attempt}/{MAX_RETRIES}")

        try:
            start = time.perf_counter()

            data, status, headers = urllib_get(ZARRAY_URL)

            elapsed = time.perf_counter() - start

            print("urllib metadata access: SUCCESS")
            print("HTTP status:", status)
            print("Bytes:", len(data))
            print(f"Elapsed: {elapsed:.3f} seconds")

            record_test(
                "urllib_metadata",
                "SUCCESS",
                "urllib successfully downloaded .zarray.",
                http_status=status,
                bytes=len(data),
                elapsed_seconds=elapsed,
            )

            return True

        except HTTPError as exc:
            print("HTTP ERROR")
            print("Status:", exc.code)
            print("Reason:", exc.reason)

            record_test(
                "urllib_metadata",
                "HTTP_FAILURE",
                str(exc),
                http_status=exc.code,
            )

            return False

        except URLError as exc:
            print("URL ERROR")
            print("Reason:", exc.reason)

            if attempt == MAX_RETRIES:
                record_test(
                    "urllib_metadata",
                    "URLLIB_FAILURE",
                    str(exc),
                )
                return False

        except Exception as exc:
            print("Unexpected error:", exc)

            record_test(
                "urllib_metadata",
                "URLLIB_FAILURE",
                str(exc),
            )

            return False

    return False


# ---------------------------------------------------------------------
# REQUESTS TEST
# ---------------------------------------------------------------------

def test_requests_metadata() -> bool:
    subsection("TEST 4: REQUESTS -> ZARR METADATA")

    try:
        import requests
    except ImportError:
        print("requests is not installed.")
        print("Skipping requests test.")

        record_test(
            "requests_metadata",
            "SKIPPED",
            "requests package is not installed.",
        )

        return False

    print("URL:")
    print(ZARRAY_URL)

    try:
        start = time.perf_counter()

        response = requests.get(
            ZARRAY_URL,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": (
                    "CryoFusion-ITS-LIVE-Diagnostic/1.0"
                )
            },
        )

        elapsed = time.perf_counter() - start

        print("requests access completed.")
        print("HTTP status:", response.status_code)
        print("Bytes:", len(response.content))
        print(f"Elapsed: {elapsed:.3f} seconds")

        if response.ok:
            record_test(
                "requests_metadata",
                "SUCCESS",
                "requests successfully downloaded .zarray.",
                http_status=response.status_code,
                bytes=len(response.content),
                elapsed_seconds=elapsed,
            )
            return True

        record_test(
            "requests_metadata",
            "HTTP_FAILURE",
            response.text[:500],
            http_status=response.status_code,
        )

        return False

    except requests.exceptions.RequestException as exc:
        print("requests failure:")
        print(exc)

        record_test(
            "requests_metadata",
            "REQUESTS_FAILURE",
            str(exc),
        )

        return False

    except Exception as exc:
        print("Unexpected error:")
        print(exc)

        record_test(
            "requests_metadata",
            "REQUESTS_FAILURE",
            str(exc),
        )

        return False


# ---------------------------------------------------------------------
# ZARR METADATA
# ---------------------------------------------------------------------

def load_json_with_urllib(url: str) -> dict:
    data, status, _ = urllib_get(url)

    if status != 200:
        raise RuntimeError(
            f"Unexpected HTTP status: {status}"
        )

    return json.loads(data.decode("utf-8"))


def test_zarr_metadata() -> tuple[dict | None, dict | None]:
    subsection("TEST 5: ZARR METADATA")

    try:
        zarray = load_json_with_urllib(ZARRAY_URL)
        print(".zarray: SUCCESS")

        zattrs = load_json_with_urllib(ZATTRS_URL)
        print(".zattrs: SUCCESS")

        print()
        print("Zarr shape:", zarray.get("shape"))
        print("Zarr chunks:", zarray.get("chunks"))
        print("Zarr dtype:", zarray.get("dtype"))
        print("Compressor:", zarray.get("compressor"))

        print()
        print("Attributes:")
        print(
            json.dumps(
                zattrs,
                indent=2,
            )
        )

        record_test(
            "zarr_metadata",
            "SUCCESS",
            "Both .zarray and .zattrs were loaded.",
            shape=zarray.get("shape"),
            chunks=zarray.get("chunks"),
            dtype=zarray.get("dtype"),
            compressor=zarray.get("compressor"),
        )

        return zarray, zattrs

    except Exception as exc:
        print("Zarr metadata failure:")
        print(exc)

        record_test(
            "zarr_metadata",
            "ZARR_METADATA_FAILURE",
            str(exc),
        )

        return None, None


# ---------------------------------------------------------------------
# VELOCITY CHUNK DOWNLOAD
# ---------------------------------------------------------------------

def download_chunk_urllib() -> bytes | None:
    subsection("TEST 6: DIRECT VELOCITY CHUNK ACCESS")

    print("Known 2004 observation:")
    print("  Cube index:", CUBE_INDEX_2004)
    print("  Date:", OBSERVATION_DATE_2004)

    print()
    print("Chunk coordinates:")
    print("  Time chunk:", TIME_CHUNK)
    print("  Y chunk:", Y_CHUNK)
    print("  X chunk:", X_CHUNK)

    print()
    print("Chunk URL:")
    print(CHUNK_URL)

    for attempt in range(1, MAX_RETRIES + 1):
        print()
        print(
            f"Download attempt {attempt}/{MAX_RETRIES}"
        )

        try:
            start = time.perf_counter()

            data, status, headers = urllib_get(
                CHUNK_URL,
                timeout=REQUEST_TIMEOUT,
            )

            elapsed = time.perf_counter() - start

            print("Chunk download: SUCCESS")
            print("HTTP status:", status)
            print("Compressed bytes:", len(data))
            print(f"Elapsed: {elapsed:.3f} seconds")

            record_test(
                "velocity_chunk_download",
                "SUCCESS",
                "Velocity chunk downloaded successfully.",
                http_status=status,
                compressed_bytes=len(data),
                elapsed_seconds=elapsed,
                content_encoding=headers.get(
                    "Content-Encoding"
                ),
                content_type=headers.get(
                    "Content-Type"
                ),
            )

            return data

        except HTTPError as exc:
            print("HTTP ERROR")
            print("Status:", exc.code)
            print("Reason:", exc.reason)

            record_test(
                "velocity_chunk_download",
                "HTTP_FAILURE",
                str(exc),
                http_status=exc.code,
            )

            return None

        except URLError as exc:
            print("URL ERROR")
            print("Reason:", exc.reason)

            if attempt == MAX_RETRIES:
                record_test(
                    "velocity_chunk_download",
                    "CHUNK_DOWNLOAD_FAILURE",
                    str(exc),
                )
                return None

        except Exception as exc:
            print("Unexpected error:")
            print(exc)

            record_test(
                "velocity_chunk_download",
                "CHUNK_DOWNLOAD_FAILURE",
                str(exc),
            )

            return None

    return None


# ---------------------------------------------------------------------
# BLOSC DECODER
# ---------------------------------------------------------------------

def decode_blosc(data: bytes, zarray: dict) -> bytes:
    """
    Decode a Zarr v2 Blosc chunk.

    Uses numcodecs if installed. This is preferable because ITS_LIVE
    chunks use Blosc compression.
    """

    try:
        from numcodecs import Blosc
    except ImportError as exc:
        raise RuntimeError(
            "numcodecs is required to decode the ITS_LIVE "
            "Blosc chunk. Install it with: "
            "pip install numcodecs"
        ) from exc

    compressor = zarray.get("compressor")

    if not compressor:
        raise RuntimeError(
            "Zarr metadata contains no compressor."
        )

    codec = Blosc(
        cname=compressor["cname"],
        clevel=compressor["clevel"],
        shuffle=compressor["shuffle"],
        blocksize=compressor.get("blocksize", 0),
    )

    return codec.decode(data)


# ---------------------------------------------------------------------
# CHUNK DECODING / INSPECTION
# ---------------------------------------------------------------------

def inspect_decoded_chunk(
    compressed: bytes,
    zarray: dict,
    zattrs: dict,
) -> None:

    subsection("TEST 7: DECODE VELOCITY CHUNK")

    try:
        decoded = decode_blosc(
            compressed,
            zarray,
        )

        print("Chunk decompression: SUCCESS")
        print("Decoded bytes:", len(decoded))

        dtype = zarray["dtype"]

        if dtype == "<i2":
            np_dtype = "<i2"
        elif dtype == "int16":
            np_dtype = "int16"
        else:
            raise RuntimeError(
                f"Unsupported dtype for diagnostic: {dtype}"
            )

        import numpy as np

        values = np.frombuffer(
            decoded,
            dtype=np_dtype,
        )

        expected_shape = tuple(
            zarray["chunks"]
        )

        expected_values = 1

        for dimension in expected_shape:
            expected_values *= dimension

        print("Decoded dtype:", values.dtype)
        print("Decoded values:", values.size)
        print("Expected values:", expected_values)
        print("Expected chunk shape:", expected_shape)

        if values.size != expected_values:
            raise RuntimeError(
                "Decoded value count does not match "
                "the expected Zarr chunk size."
            )

        missing_value = zattrs.get(
            "missing_value",
            -32767,
        )

        missing_mask = values == missing_value
        valid_mask = ~missing_mask

        missing_count = int(
            missing_mask.sum()
        )

        valid_count = int(
            valid_mask.sum()
        )

        print()
        print("Chunk statistics:")
        print("  Total values:", values.size)
        print("  Missing:", missing_count)
        print("  Valid:", valid_count)
        print(
            "  Missing %:",
            f"{100 * missing_count / values.size:.4f}",
        )
        print(
            "  Valid %:",
            f"{100 * valid_count / values.size:.4f}",
        )

        if valid_count > 0:
            valid_values = values[valid_mask]

            print()
            print("Valid value statistics:")
            print(
                "  Minimum:",
                int(valid_values.min()),
            )
            print(
                "  Maximum:",
                int(valid_values.max()),
            )
            print(
                "  Mean:",
                float(valid_values.mean()),
            )
            print(
                "  Median:",
                float(np.median(valid_values)),
            )

            status = "VALID_DATA_IN_CHUNK"

        else:
            print()
            print(
                "RESULT: The downloaded chunk contains "
                "zero valid velocity values."
            )

            status = "ZERO_VALID_IN_CHUNK"

        record_test(
            "velocity_chunk_decode",
            status,
            "Chunk decoded and velocity values inspected.",
            decoded_bytes=len(decoded),
            total_values=int(values.size),
            missing_values=missing_count,
            valid_values=valid_count,
            missing_percentage=(
                100 * missing_count / values.size
            ),
            valid_percentage=(
                100 * valid_count / values.size
            ),
        )

    except Exception as exc:
        print("Chunk decode failure:")
        print(exc)

        record_test(
            "velocity_chunk_decode",
            "CHUNK_DECODE_FAILURE",
            str(exc),
        )


# ---------------------------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------------------------

def print_interpretation() -> None:
    section("DIAGNOSTIC INTERPRETATION")

    statuses = [
        test["status"]
        for test in results["tests"]
    ]

    print("Observed test statuses:")
    for status in statuses:
        print(f"  - {status}")

    print()

    dns_ok = any(
        test["test"] == "dns_resolution"
        and test["status"] == "SUCCESS"
        for test in results["tests"]
    )

    urllib_ok = any(
        test["test"] == "urllib_metadata"
        and test["status"] == "SUCCESS"
        for test in results["tests"]
    )

    requests_ok = any(
        test["test"] == "requests_metadata"
        and test["status"] == "SUCCESS"
        for test in results["tests"]
    )

    chunk_ok = any(
        test["test"] == "velocity_chunk_download"
        and test["status"] == "SUCCESS"
        for test in results["tests"]
    )

    chunk_valid = any(
        test["test"] == "velocity_chunk_decode"
        and test["status"] == "VALID_DATA_IN_CHUNK"
        for test in results["tests"]
    )

    chunk_zero = any(
        test["test"] == "velocity_chunk_decode"
        and test["status"] == "ZERO_VALID_IN_CHUNK"
        for test in results["tests"]
    )

    print("Interpretation:")

    if not dns_ok:
        print(
            "  DNS FAILURE: The ITS_LIVE hostname could not "
            "be resolved."
        )
        print(
            "  This is a local/network/DNS problem."
        )
        return

    print(
        "  DNS resolution works."
    )

    if urllib_ok:
        print(
            "  urllib can access ITS_LIVE metadata."
        )
    else:
        print(
            "  urllib cannot access ITS_LIVE metadata."
        )

    if requests_ok:
        print(
            "  requests can access ITS_LIVE metadata."
        )
    else:
        print(
            "  requests did not successfully access "
            "ITS_LIVE metadata."
        )

    if chunk_ok:
        print(
            "  Direct velocity chunk access works."
        )
    else:
        print(
            "  Direct velocity chunk access failed."
        )

    print()

    if chunk_valid:
        print(
            "FINAL RESULT:"
        )
        print(
            "  VALID VELOCITY DATA WAS FOUND IN THE "
            "TESTED 2004 CHUNK."
        )
        print(
            "  The next step is to run the strategic "
            "temporal coverage inspection."
        )

    elif chunk_zero:
        print(
            "FINAL RESULT:"
        )
        print(
            "  The tested chunk downloaded and decoded "
            "successfully but contained ZERO valid values."
        )
        print(
            "  This is a genuine data-coverage observation "
            "for this chunk."
        )

    elif chunk_ok:
        print(
            "FINAL RESULT:"
        )
        print(
            "  The chunk was accessible, but decoding/"
            "inspection did not establish valid velocity data."
        )

    else:
        print(
            "FINAL RESULT:"
        )
        print(
            "  The 2004 velocity chunk could not be tested."
        )
        print(
            "  Do NOT interpret this as missing ITS_LIVE "
            "velocity data."
        )


# ---------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------

def save_results() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results["generated_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
            default=str,
        )

    print()
    print("Diagnostic results saved:")
    print(f"  {OUTPUT_FILE}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    section(
        "ITS_LIVE SINGLE-OBSERVATION ACCESS DIAGNOSTIC"
    )

    print("Purpose:")
    print(
        "Determine whether the ITS_LIVE 2004 observation "
        "can be accessed and decoded."
    )

    print()
    print("Observation:")
    print("  Year:       2004")
    print("  Cube index:", CUBE_INDEX_2004)
    print("  Date:", OBSERVATION_DATE_2004)

    print()
    print("Zarr:")
    print(ZARR_BASE)

    print()
    print("Target chunk:")
    print(CHUNK_URL)

    # -------------------------------------------------------------
    # 1. DNS
    # -------------------------------------------------------------

    dns_ok = test_dns()

    if not dns_ok:
        section("STOPPING EARLY")

        print(
            "The ITS_LIVE hostname cannot be resolved."
        )
        print(
            "There is no value in downloading Zarr chunks "
            "until DNS/network access is restored."
        )

        save_results()
        return

    # -------------------------------------------------------------
    # 2. TCP
    # -------------------------------------------------------------

    test_tcp()

    # -------------------------------------------------------------
    # 3. urllib metadata
    # -------------------------------------------------------------

    urllib_metadata_ok = test_urllib_metadata()

    # -------------------------------------------------------------
    # 4. requests metadata
    # -------------------------------------------------------------

    requests_metadata_ok = test_requests_metadata()

    # -------------------------------------------------------------
    # 5. Zarr metadata
    # -------------------------------------------------------------

    zarray, zattrs = test_zarr_metadata()

    if zarray is None or zattrs is None:
        section("STOPPING BEFORE CHUNK TEST")

        print(
            "Zarr metadata could not be loaded."
        )
        print(
            "The velocity chunk cannot be decoded safely "
            "without the Zarr compressor/dtype metadata."
        )

        print_interpretation()
        save_results()
        return

    # -------------------------------------------------------------
    # 6. Direct velocity chunk
    # -------------------------------------------------------------

    compressed = download_chunk_urllib()

    if compressed is None:
        print_interpretation()
        save_results()
        return

    # -------------------------------------------------------------
    # 7. Decode and inspect
    # -------------------------------------------------------------

    inspect_decoded_chunk(
        compressed,
        zarray,
        zattrs,
    )

    # -------------------------------------------------------------
    # 8. Interpretation
    # -------------------------------------------------------------

    print_interpretation()

    # -------------------------------------------------------------
    # 9. Save
    # -------------------------------------------------------------

    save_results()

    section("DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()

