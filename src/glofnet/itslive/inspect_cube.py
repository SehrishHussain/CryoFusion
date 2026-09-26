import xarray as xr

from glofnet.common.console import (
    print_header,
    print_section,
    print_key_value,
    print_success,
)

CUBE_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)


def main():

    print_header("ITS_LIVE Zarr Cube Inspection")

    print_section("Opening Zarr")

    ds = xr.open_zarr(
        CUBE_URL,
        consolidated=True,
    )

    print_success("Zarr opened successfully")

    print_section("Dataset")

    print(ds)

    print_section("Dimensions")

    for name, size in ds.sizes.items():
        print_key_value(name, size)

    print_section("Coordinates")

    for name in ds.coords:
        print_key_value(name, str(ds[name]))

    print_section("Variables")

    for name in ds.data_vars:
        variable = ds[name]

        print_key_value("Variable", name)
        print_key_value("Dimensions", variable.dims)
        print_key_value("Shape", variable.shape)
        print_key_value("Dtype", variable.dtype)
        print()


if __name__ == "__main__":
    main()