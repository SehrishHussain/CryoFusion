import xarray as xr
import pandas as pd

from glofnet.itslive.config import GLACIER_ID

CUBE_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)


def main():

    ds = xr.open_zarr(
        CUBE_URL,
        consolidated=True,
    )

    dates = pd.to_datetime(ds["mid_date"].values)

    print("\n===== TEMPORAL RANGE =====")
    print("Minimum:", dates.min())
    print("Maximum:", dates.max())

    print("\n===== SORTING =====")
    print("Already sorted:", dates.is_monotonic_increasing)

    print("\n===== UNIQUE DATES =====")
    print("Unique:", dates.nunique())
    print("Total:", len(dates))

    print("\n===== OBSERVATIONS BY YEAR =====")
    print(dates.year.value_counts().sort_index())

    print("\n===== OBSERVATIONS BY MONTH =====")
    print(dates.month.value_counts().sort_index())

    print("\n===== FIRST 20 =====")
    print(dates[:20])

    print("\n===== LAST 20 =====")
    print(dates[-20:])


if __name__ == "__main__":
    main()