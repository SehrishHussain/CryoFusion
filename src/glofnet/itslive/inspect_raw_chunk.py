import json
import time

import numpy as np

from .download_representative_v1 import (
    ZARR_URL,
    http_get,
    decode_blosc,
)


VARIABLE = "v"

# This corresponds to:
# time chunk 0
# y chunk 51
# x chunk 48
#
# The 2004 observation is time index 13191.
CHUNK_KEY = "0.51.48"


def main():
    print("=" * 70)
    print("ITS_LIVE RAW ZARR CHUNK INSPECTION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load array metadata
    # ------------------------------------------------------------

    metadata_url = f"{ZARR_URL}/{VARIABLE}/.zarray"

    print()
    print("Loading metadata:")
    print(metadata_url)

    response = http_get(metadata_url)

    array_meta = json.loads(
        response.decode("utf-8")
    )

    print()
    print("Array metadata:")
    print(json.dumps(array_meta, indent=4))

   # ------------------------------------------------------------
    # Load array attributes
    # ------------------------------------------------------------

    attrs_url = f"{ZARR_URL}/{VARIABLE}/.zattrs"

    print()
    print("Loading attributes:")
    print(attrs_url)

    response = http_get(attrs_url)

    attrs = json.loads(
        response.decode("utf-8")
    )

    print()
    print("Array attributes:")
    print(json.dumps(attrs, indent=4))

   # ------------------------------------------------------------
    # Download chunk
    # ------------------------------------------------------------

    chunk_url = (
        f"{ZARR_URL}/"
        f"{VARIABLE}/"
        f"{CHUNK_KEY}"
    )

    print()
    print("Downloading chunk:")
    print(chunk_url)

    response = http_get(chunk_url)

    print()
    print("Raw compressed bytes:", len(response))

    time.sleep(0.15)

    # ------------------------------------------------------------
    # Decode Blosc
    # ------------------------------------------------------------

    raw = decode_blosc(
        response,
        array_meta.get("compressor"),
    )

    dtype = np.dtype(array_meta["dtype"])

    values = np.frombuffer(
        raw,
        dtype=dtype,
    )

    print()
    print("=" * 70)
    print("DECODED CHUNK")
    print("=" * 70)

    print("dtype:", values.dtype)
    print("number of values:", values.size)

    print("expected chunk size:")

    expected = (
        array_meta["chunks"][0]
        * array_meta["chunks"][1]
        * array_meta["chunks"][2]
    )

    print(expected)

    print()
    print("minimum:", values.min())
    print("maximum:", values.max())
    print("mean:", values.mean())
    print("median:", np.median(values))

    print()
    print("unique values:", len(np.unique(values)))

    print()
    print("missing value:", attrs.get("missing_value"))

    missing_value = attrs.get("missing_value")

    if missing_value is not None:
        missing_count = np.count_nonzero(
            values == missing_value
        )

        print(
            "missing-value count:",
            missing_count,
        )

        print(
            "missing-value percentage:",
            100.0 * missing_count / values.size,
        )

    print()
    print("First 100 values:")
    print(values[:100])

    print()
    print("=" * 70)
    print("VALUES AROUND 2004 TIME OFFSET")
    print("=" * 70)

    time_index = 13191
    time_chunk_size = array_meta["chunks"][0]

    time_offset = time_index % time_chunk_size

    print("time index:", time_index)
    print("time chunk size:", time_chunk_size)
    print("time offset:", time_offset)

    spatial_size = (
        array_meta["chunks"][1]
        * array_meta["chunks"][2]
    )

    start = time_offset * spatial_size
    end = start + spatial_size

    observation_values = values[start:end]

    print()
    print("Observation slice size:", observation_values.size)

    print(
        "Observation min:",
        observation_values.min(),
    )

    print(
        "Observation max:",
        observation_values.max(),
    )

    print(
        "Observation mean:",
        observation_values.mean(),
    )

    print(
        "Observation median:",
        np.median(observation_values),
    )

    if missing_value is not None:
        valid = observation_values[
            observation_values != missing_value
        ]

        print()
        print("Valid observation values:", valid.size)

        if valid.size:
            print("Valid min:", valid.min())
            print("Valid max:", valid.max())
            print("Valid mean:", valid.mean())
            print("Valid median:", np.median(valid))

            print()
            print("First 50 valid values:")
            print(valid[:50])

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()