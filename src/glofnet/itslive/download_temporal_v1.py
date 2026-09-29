import os
import numpy as np
import xarray as xr

from glofnet.common.find_glacier import load_glacier
from glofnet.itslive.config import GLACIER_ID


# ============================================================
# CONFIGURATION
# ============================================================

CUBE_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)

OUTPUT_DIR = "data/raw/itslive/temporal_v1"

# Smoke test:
# Use the observation that we already successfully extracted
# in the single-observation V1 pipeline.
OBSERVATION_INDEX = 7

VARIABLES = [
    "v",
    "vx",
    "vy",
    "v_error",
    "interp_mask",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 50)
    print("       ITS_LIVE TEMPORAL V1 SMOKE TEST")
    print("=" * 50)

    # --------------------------------------------------------
    # LOAD GLACIER
    # --------------------------------------------------------

    print("\n===== LOADING GLACIER =====")

    glacier = load_glacier(GLACIER_ID).to_crs("EPSG:32643")

    minx, miny, maxx, maxy = glacier.total_bounds

    print("Glacier:", GLACIER_ID)

    print("Bounds:")
    print("minx:", minx)
    print("miny:", miny)
    print("maxx:", maxx)
    print("maxy:", maxy)

    # --------------------------------------------------------
    # OPEN ITS_LIVE ZARR
    # --------------------------------------------------------

    print("\n===== OPENING ITS_LIVE ZARR =====")

    ds = xr.open_zarr(
        CUBE_URL,
        consolidated=True,
    )

    print("Zarr opened successfully.")

    # --------------------------------------------------------
    # SPATIAL SUBSET
    # --------------------------------------------------------

    print("\n===== SPATIAL SUBSET =====")

    subset = ds.sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny),
    )

    print("Subset dimensions:")
    print(subset.sizes)

    # --------------------------------------------------------
    # CHECK OBSERVATION INDEX
    # --------------------------------------------------------

    print("\n===== OBSERVATION CHECK =====")

    total_observations = subset.sizes["mid_date"]

    print("Total observations:", total_observations)
    print("Requested observation index:", OBSERVATION_INDEX)

    if OBSERVATION_INDEX >= total_observations:
        raise IndexError(
            f"Observation index {OBSERVATION_INDEX} is outside "
            f"the available range 0-{total_observations - 1}"
        )

    observation_date = subset["mid_date"].isel(
        mid_date=OBSERVATION_INDEX
    ).values

    print("Observation date:", observation_date)

    # --------------------------------------------------------
    # EXTRACT ONE OBSERVATION
    # --------------------------------------------------------

    print("\n===== EXTRACTING OBSERVATION =====")

    print("Loading data...")

    obs = subset.isel(
        mid_date=OBSERVATION_INDEX
    )[VARIABLES].load()

    print("Extraction successful.")

    # --------------------------------------------------------
    # VARIABLE CHECK
    # --------------------------------------------------------

    print("\n===== VARIABLES =====")

    for variable in VARIABLES:

        data = obs[variable]

        print(
            f"{variable}: "
            f"shape={data.shape}, "
            f"dtype={data.dtype}"
        )

    # --------------------------------------------------------
    # VALID VELOCITY
    # --------------------------------------------------------

    print("\n===== VALID VELOCITY =====")

    valid = obs["v"].notnull()

    valid_count = int(valid.sum().item())
    total_pixels = int(valid.size)

    valid_percentage = (
        valid_count / total_pixels * 100
        if total_pixels > 0
        else 0
    )

    print("Total pixels:", total_pixels)
    print("Valid velocity pixels:", valid_count)
    print(f"Valid percentage: {valid_percentage:.2f}%")

    # --------------------------------------------------------
    # VELOCITY SUMMARY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # OUTPUT PATH
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    timestamp = np.datetime_as_string(
        np.datetime64(observation_date),
        unit="s",
    )

    safe_timestamp = timestamp.replace(":", "-")

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{GLACIER_ID}_{safe_timestamp}.nc",
    )

    # --------------------------------------------------------
    # SAVE NETCDF
    # --------------------------------------------------------

    print("\n===== SAVING =====")

    print("Output:")
    print(output_path)

    obs.to_netcdf(
        output_path,
        engine="scipy",
    )

    print("\nSaved successfully.")

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 50)
    print("       SMOKE TEST COMPLETE")
    print("=" * 50)

    print("\nObservation:")
    print("Cube index:", OBSERVATION_INDEX)
    print("Date:", observation_date)

    print("\nData:")
    print("Shape:", obs["v"].shape)
    print("Valid pixels:", valid_count)
    print(f"Valid percentage: {valid_percentage:.2f}%")

    print("\nOutput file:")
    print(output_path)


if __name__ == "__main__":
    main()