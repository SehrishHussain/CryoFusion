"""
Explore the ITS_LIVE Python API.

This script searches for glacier velocity granules using the official
ITS_LIVE Python package and inspects the first few search results.

The purpose is to understand the structure of the objects returned by the
API before implementing the production search and download pipeline.
"""

from datetime import date
from time import perf_counter

import itslive

from glofnet.common.find_glacier import load_glacier
from glofnet.itslive.config import (
    GLACIER_ID,
    START_DATE,
    END_DATE,
)


def glacier_to_geojson(glacier):
    """
    Convert a glacier polygon to a GeoJSON geometry.

    Parameters
    ----------
    glacier : GeoDataFrame
        GeoDataFrame containing a single glacier.

    Returns
    -------
    dict
        GeoJSON geometry.
    """
    return glacier.geometry.iloc[0].__geo_interface__


def main():
    """Search for a few ITS_LIVE velocity granules and inspect them."""

    glacier = load_glacier(GLACIER_ID)
    geometry = glacier_to_geojson(glacier)

    print("=" * 70)
    print("ITS_LIVE API EXPLORATION")
    print("=" * 70)

    print(f"ITS_LIVE module : {itslive}")
    print()

    print("Searching for matching velocity granules...\n")
    start = perf_counter()


    stream = itslive.velocity_pairs.find_streaming(
        geojson=geometry,
        start=date.fromisoformat(START_DATE),
        end=date.fromisoformat(END_DATE),
    )
    found = 0
    start = perf_counter()
    for i, granule in enumerate(stream, start=1):

        if i == 1:
            elapsed = perf_counter() - start
            print(f"First result after {elapsed:.2f} seconds")

            print("=" * 70)
            print(f"GRANULE {i}")
            print("=" * 70)

            print(f"Object type : {type(granule)}")
            print()
            print(granule)
            print()

            found += 1

            # Stop after inspecting a few results.
        if found == 5:
            break

    if found == 0:
        print("No matching granules were found.")

    print()
    print(f"Displayed {found} granule(s).")




if __name__ == "__main__":
    main()