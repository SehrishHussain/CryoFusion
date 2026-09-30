from __future__ import annotations

import json
import requests


ZARR_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/"
    "N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)


def main() -> None:

    url = f"{ZARR_URL}/.zmetadata"

    print("=" * 70)
    print("ITS_LIVE ZARR METADATA INSPECTION")
    print("=" * 70)

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    metadata = response.json()["metadata"]

    variables = [
        "v",
        "vx",
        "vy",
        "v_error",
        "interp_mask",
    ]

    for variable in variables:

        key = f"{variable}/.zarray"

        print()
        print("-" * 70)
        print(variable)
        print("-" * 70)

        meta = metadata[key]

        print(json.dumps(meta, indent=4))


if __name__ == "__main__":
    main()