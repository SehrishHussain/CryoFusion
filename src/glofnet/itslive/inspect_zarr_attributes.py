import json
import requests


ZARR_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/"
    "N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)

VARIABLES = [
    "v",
    "vx",
    "vy",
    "v_error",
    "interp_mask",
]


def main():
    print("=" * 70)
    print("ITS_LIVE ZARR ARRAY ATTRIBUTES INSPECTION")
    print("=" * 70)

    session = requests.Session()

    for variable in VARIABLES:
        print()
        print("-" * 70)
        print(variable)
        print("-" * 70)

        url = f"{ZARR_URL}/{variable}/.zattrs"

        print(f"URL: {url}")

        response = session.get(
            url,
            timeout=30,
        )

        response.raise_for_status()

        attrs = response.json()

        print(json.dumps(attrs, indent=4, default=str))


if __name__ == "__main__":
    main()