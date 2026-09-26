"""
Evaluate an ITS_LIVE granule for the configured glacier.

The evaluation score is the number of valid velocity pixels
contained within the glacier bounding box.
"""

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from glofnet.common.find_glacier import load_glacier
from glofnet.common.geospatial import (
    get_dataset_crs,
    reproject_geometry,
)
from glofnet.itslive.config import GLACIER_ID


VELOCITY_VARIABLES = [
    "vx",
    "vy",
    "v",
    "v_error",
]


def evaluate_granule(path: Path) -> int:
    """
    Evaluate one ITS_LIVE granule.

    Parameters
    ----------
    path : Path
        Path to a downloaded ITS_LIVE NetCDF file.

    Returns
    -------
    int
        Number of valid glacier velocity pixels.
    """

    ds = xr.open_dataset(path)

    try:
        # --------------------------------------------------------------
        # Attach CRS
        # --------------------------------------------------------------
        crs = get_dataset_crs(ds)
        ds = ds.rio.write_crs(crs)

        # --------------------------------------------------------------
        # Load glacier geometry
        # --------------------------------------------------------------
        glacier = load_glacier(GLACIER_ID)
        glacier = reproject_geometry(glacier, crs)

        xmin, ymin, xmax, ymax = glacier.total_bounds

        # --------------------------------------------------------------
        # Crop to glacier bounding box
        # --------------------------------------------------------------
        cropped = ds.rio.clip_box(
            minx=xmin,
            miny=ymin,
            maxx=xmax,
            maxy=ymax,
        )

        try:
            # Use velocity magnitude as the scoring variable.
            data = cropped["v"].values

            valid_pixels = np.count_nonzero(
                ~np.isnan(data)
            )

        finally:
            cropped.close()

    finally:
        ds.close()

    return int(valid_pixels)


def main():

    from glofnet.common.paths import ITSLIVE_RAW_DIRECTORY

    datasets = sorted(
        ITSLIVE_RAW_DIRECTORY.glob("*.nc")
    )

    if not datasets:
        raise FileNotFoundError(
            "No ITS_LIVE datasets found."
        )

    for dataset in datasets:

        score = evaluate_granule(dataset)

        print("=" * 70)
        print(dataset.name)
        print(f"Valid glacier pixels: {score:,}")


if __name__ == "__main__":
    main()