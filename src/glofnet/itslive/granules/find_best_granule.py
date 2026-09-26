"""
Find the first ITS_LIVE granule that contains valid velocity
measurements over the configured glacier.
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
from glofnet.itslive.granules.download_granules import download_granule
from glofnet.itslive.granules.search_granules import search_granules


def has_valid_pixels(path: Path) -> bool:
    """
    Return True if the granule contains at least one valid
    velocity pixel inside the glacier bounding box.
    """

    with xr.open_dataset(path) as ds:

        crs = get_dataset_crs(ds)
        ds = ds.rio.write_crs(crs)

        glacier = load_glacier(GLACIER_ID)
        glacier = reproject_geometry(glacier, crs)

        xmin, ymin, xmax, ymax = glacier.total_bounds

        cropped = ds.rio.clip_box(
            minx=xmin,
            miny=ymin,
            maxx=xmax,
            maxy=ymax,
        )

        has_data = False

        print()

        for var in ["vx", "vy", "v", "v_error"]:

            valid = np.count_nonzero(
                ~np.isnan(cropped[var].values)
            )

            print(f"{var:<8}: {valid:,} valid pixels")

            if valid > 0:
                has_data = True

        cropped.close()

    return has_data


def main():

    granules = search_granules()

    print(f"\nTesting {len(granules)} candidate granules...\n")

    for i, granule in enumerate(granules, start=1):

        print("=" * 80)
        print(f"Candidate {i}/{len(granules)}")
        print(f"Platform : {granule.platform}")
        print(f"Quality  : {granule.percent_valid_pixels}%")
        print(granule.filename)

        path = download_granule(granule)

        print("\nChecking glacier pixels...")

        if has_valid_pixels(path):

            print("\nSUCCESS")
            print("Found a usable granule:")
            print(path)

            return

        print("\nNo valid glacier pixels.")
        print("Trying next candidate...\n")

        # Leave the downloaded file in place while debugging.
        # Uncomment later if you want to clean up automatically.
        #
        # path.unlink()

    print("\nNo suitable granule found.")
    print("All candidate granules contained only NaN values.")


if __name__ == "__main__":
    main()