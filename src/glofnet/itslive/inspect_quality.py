import xarray as xr

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

    subset = ds[
        [
            "v",
            "vx",
            "vy",
            "v_error",
            "interp_mask",
            "roi_valid_percentage",
        ]
    ].sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny),
    )

    print("\n===== SPATIAL SUBSET =====")
    print(subset.sizes)

    # ---------------------------------------------------------
    # Inspect only 10 temporal observations
    # ---------------------------------------------------------

    sample = subset.isel(
        mid_date=slice(0, 10)
    )

    print("\n===== TEMPORAL SAMPLE =====")

    print(
        sample["mid_date"].values
    )

    for variable in [
        "v",
        "vx",
        "vy",
        "v_error",
        "interp_mask",
        "roi_valid_percentage",
    ]:

        da = sample[variable]

        print(f"\n===== {variable} =====")

        print("Shape:", da.shape)

        print(
            "Min:",
            da.min(skipna=True).compute().item()
        )

        print(
            "Max:",
            da.max(skipna=True).compute().item()
        )

        print(
            "Mean:",
            da.mean(skipna=True).compute().item()
        )

        print(
            "NaN %:",
            da.isnull()
            .mean()
            .compute()
            .item()
            * 100
        )


if __name__ == "__main__":
    main()