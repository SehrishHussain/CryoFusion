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

    subset = ds.sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny),
    )

    dates = subset["mid_date"].values

    print("\n===== OBSERVATIONS =====")

    for i in range(20):

        print(f"\n--- Observation {i} ---")

        print("mid_date:", dates[i])

        print(
            "img1:",
            subset["acquisition_date_img1"].isel(mid_date=i).values
        )

        print(
            "img2:",
            subset["acquisition_date_img2"].isel(mid_date=i).values
        )

        print(
            "mission img1:",
            subset["mission_img1"].isel(mid_date=i).values
        )

        print(
            "mission img2:",
            subset["mission_img2"].isel(mid_date=i).values
        )

        print(
            "sensor img1:",
            subset["sensor_img1"].isel(mid_date=i).values
        )

        print(
            "sensor img2:",
            subset["sensor_img2"].isel(mid_date=i).values
        )

        print(
            "roi_valid_percentage:",
            subset["roi_valid_percentage"].isel(mid_date=i).values
        )

        v = subset["v"].isel(mid_date=i)

        print(
            "v valid pixels:",
            v.notnull().sum().compute().item()
        )

        v_error = subset["v_error"].isel(mid_date=i)

        print(
            "v_error valid pixels:",
            v_error.notnull().sum().compute().item()
        )


if __name__ == "__main__":
    main()