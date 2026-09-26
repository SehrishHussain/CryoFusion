import xarray as xr

from glofnet.common.find_glacier import load_glacier
from glofnet.itslive.config import GLACIER_ID


CUBE_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)


def main():

    # ---------------------------------------------------------
    # 1. Load glacier
    # ---------------------------------------------------------

    glacier = load_glacier(GLACIER_ID)

    print("\n===== GLACIER =====")
    print("ID:", GLACIER_ID)
    print("CRS:", glacier.crs)

    # ---------------------------------------------------------
    # 2. Reproject glacier to cube CRS
    # ---------------------------------------------------------

    glacier_utm = glacier.to_crs("EPSG:32643")

    minx, miny, maxx, maxy = glacier_utm.total_bounds

    print("\n===== GLACIER PROJECTED BOUNDS =====")
    print("minx:", minx)
    print("miny:", miny)
    print("maxx:", maxx)
    print("maxy:", maxy)

    # ---------------------------------------------------------
    # 3. Open remote Zarr
    # ---------------------------------------------------------

    print("\n===== OPENING ZARR =====")

    ds = xr.open_zarr(
        CUBE_URL,
        consolidated=True,
    )

    print("Opened.")

    # ---------------------------------------------------------
    # 4. Select spatial region
    # ---------------------------------------------------------

    # x is ascending
    x_min = ds.x.values.min()
    x_max = ds.x.values.max()

    # y is descending
    y_max = ds.y.values.max()
    y_min = ds.y.values.min()

    # Clip requested glacier bounds to cube bounds
    subset_minx = max(minx, x_min)
    subset_maxx = min(maxx, x_max)
    subset_miny = max(miny, y_min)
    subset_maxy = min(maxy, y_max)

    print("\n===== SUBSET BOUNDS =====")
    print("x:", subset_minx, "→", subset_maxx)
    print("y:", subset_maxy, "→", subset_miny)

    # ---------------------------------------------------------
    # 5. Spatial subset
    # ---------------------------------------------------------

    subset = ds[
        [
            "v",
            "vx",
            "vy",
            "v_error",
            "interp_mask",
        ]
    ].sel(
        x=slice(subset_minx, subset_maxx),
        y=slice(subset_maxy, subset_miny),
    )

    # ---------------------------------------------------------
    # 6. Inspect
    # ---------------------------------------------------------

    print("\n===== SPATIAL SUBSET =====")

    print(subset)

    print("\n===== SUBSET SIZES =====")

    for name, size in subset.sizes.items():
        print(f"{name}: {size}")


if __name__ == "__main__":
    main()