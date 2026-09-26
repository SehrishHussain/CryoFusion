import xarray as xr

CUBE_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)

print("Opening Zarr...")

try:
    ds = xr.open_zarr(
        CUBE_URL,
        consolidated=True,
    )

    print("SUCCESS")
    print(ds.sizes)
    print(list(ds.data_vars))

except Exception as e:
    print(type(e).__name__, e)