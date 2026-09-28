
import xarray as xr
import numpy as np

from glofnet.common.find_glacier import load_glacier
from glofnet.itslive.config import GLACIER_ID


CUBE_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)


def main():

    glacier = load_glacier(GLACIER_ID).to_crs("EPSG:32643")
    minx, miny, maxx, maxy = glacier.total_bounds

    ds = xr.open_zarr(
        CUBE_URL,
        consolidated=True,
    )

    v_full = ds["v"]

    print("\n===== V CHUNKING =====")
    print("Shape:", v_full.shape)
    print("Chunks:", v_full.chunks)

    subset = ds.sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny),
    )

    print("\n===== SUBSET =====")
    print(subset.sizes)

    obs = subset.isel(mid_date=7)

    print("\n===== LOADING V =====")
    v = obs["v"].load()
    print("V SUCCESS")

    # Define valid velocity pixels immediately after loading v
    valid = v.notnull()

    print("\n===== LOADING VX =====")
    vx = obs["vx"].load()
    print("VX SUCCESS")

    print("\n===== LOADING VY =====")
    vy = obs["vy"].load()
    print("VY SUCCESS")

    print("\n===== LOADING V_ERROR =====")
    v_error = obs["v_error"].load()
    print("V_ERROR SUCCESS")

    print("\n===== LOADING INTERP =====")
    interp = obs["interp_mask"].load()
    print("INTERP SUCCESS")

    # ---------------------------------------------------------
    # INTERPOLATION MASK
    # ---------------------------------------------------------

    print("\n===== INTERP MASK =====")

    # Only inspect interpolation values where velocity is valid
    interp_valid = interp.where(valid).values

    # Remove NaN values from the interpolation mask
    interp_values = interp_valid[~np.isnan(interp_valid)]

    if interp_values.size > 0:

        values, counts = np.unique(
            interp_values,
            return_counts=True,
        )

        for value, count in zip(values, counts):
            print(f"Value {value}: {count} pixels")

    else:
        print("No valid interpolation-mask values found.")

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print("\n===== RESULTS =====")

    print("Shape:", v.shape)
    print("Valid pixels:", valid.sum().item())

    print("\nv")
    print("Min:", v.where(valid).min().item())
    print("Max:", v.where(valid).max().item())
    print("Mean:", v.where(valid).mean().item())

    print("\nvx")
    print("Mean:", vx.where(valid).mean().item())

    print("\nvy")
    print("Mean:", vy.where(valid).mean().item())

    print("\nv_error")
    print("Mean:", v_error.where(valid).mean().item())

    # ---------------------------------------------------------
    # EXPLICIT INTERPOLATION COUNTS
    # ---------------------------------------------------------

    print("\n===== INTERPOLATION COUNTS =====")

    interpolated = ((interp == 1) & valid).sum().item()
    non_interpolated = ((interp == 0) & valid).sum().item()

    print("Interpolated valid pixels:", interpolated)
    print("Non-interpolated valid pixels:", non_interpolated)

    print("\n===== RAW INTERP MASK =====")

    interp_values = interp.values.flatten()

    print("Total pixels:", interp_values.size)
    print("NaN pixels:", np.isnan(interp_values).sum())

    values, counts = np.unique(
        interp_values[~np.isnan(interp_values)],
        return_counts=True,
    )

    for value, count in zip(values, counts):
        print(f"Value {value}: {count} pixels")


if __name__ == "__main__":
    main()


