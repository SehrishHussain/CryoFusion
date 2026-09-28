import os

import xarray as xr

from glofnet.common.find_glacier import load_glacier
from glofnet.itslive.config import GLACIER_ID


CUBE_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)

OUTPUT_DIR = "data/raw/itslive"


# V1 variables:
# - Core velocity variables are extracted.
# - interp_mask is preserved but not interpreted.
VARIABLES = [
    "v",
    "vx",
    "vy",
    "v_error",
    "interp_mask",
]


def main():

    print("===== LOADING GLACIER =====")

    glacier = load_glacier(GLACIER_ID).to_crs("EPSG:32643")

    minx, miny, maxx, maxy = glacier.total_bounds

    print("Glacier:", GLACIER_ID)
    print("Bounds:")
    print("minx:", minx)
    print("miny:", miny)
    print("maxx:", maxx)
    print("maxy:", maxy)

    print("\n===== OPENING ITS_LIVE ZARR =====")

    ds = xr.open_zarr(
        CUBE_URL,
        consolidated=True,
    )

    print("Zarr opened successfully.")

    print("\n===== SPATIAL SUBSET =====")

    subset = ds.sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny),
    )

    print("Subset dimensions:")
    print(subset.sizes)

    print("\n===== SELECTING OBSERVATION =====")

    # V1: start with one known-good observation.
    observation_index = 7

    obs = subset.isel(
        mid_date=observation_index
    )

    print("Observation index:", observation_index)
    print("mid_date:", obs["mid_date"].values)

    print("\n===== EXTRACTING DATA =====")

    # Load the relatively small spatial observation into memory.
    obs = obs[VARIABLES].load()

    print("Extraction successful.")

    for variable in VARIABLES:

        data = obs[variable]

        print(
            f"{variable}: "
            f"shape={data.shape}, "
            f"dtype={data.dtype}"
        )

    print("\n===== VALID VELOCITY =====")

    valid = obs["v"].notnull()

    print(
        "Valid velocity pixels:",
        valid.sum().item()
    )

    print("\n===== VELOCITY SUMMARY =====")

    print(
        "v min:",
        obs["v"].where(valid).min().item()
    )

    print(
        "v max:",
        obs["v"].where(valid).max().item()
    )

    print(
        "v mean:",
        obs["v"].where(valid).mean().item()
    )

    print(
        "vx mean:",
        obs["vx"].where(valid).mean().item()
    )

    print(
        "vy mean:",
        obs["vy"].where(valid).mean().item()
    )

    print(
        "v_error mean:",
        obs["v_error"].where(valid).mean().item()
    )

    print("\n===== SAVING =====")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{GLACIER_ID}_observation_{observation_index}.nc",
    )

    obs.to_netcdf(output_path)

    print("Saved successfully:")
    print(output_path)


if __name__ == "__main__":
    main()