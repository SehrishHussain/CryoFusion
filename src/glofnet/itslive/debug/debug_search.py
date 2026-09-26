"""
Visualize ITS_LIVE STAC search results.

This script plots:

1. The selected glacier.
2. The bounding boxes of all ITS_LIVE granules returned by the STAC search.

This helps verify whether the search is selecting appropriate granules
before downloading any data.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from glofnet.common.find_glacier import load_glacier
from glofnet.itslive.config import GLACIER_ID
from glofnet.itslive.granules.search_granules import search_granules


def main():

    glacier = load_glacier(GLACIER_ID)

    granules = search_granules()

    print(f"Found {len(granules)} granules.")

    fig, ax = plt.subplots(figsize=(10, 10))

    # ------------------------------------------------------------
    # Plot glacier
    # ------------------------------------------------------------

    glacier.boundary.plot(
        ax=ax,
        color="red",
        linewidth=2,
        label="Glacier",
    )

    # ------------------------------------------------------------
    # Plot granule bounding boxes
    # ------------------------------------------------------------

    for i, granule in enumerate(granules, start=1):

        west, south, east, north = granule.bbox
         # Print granule information
        print(
            f"{i:2d}: "
            f"{granule.platform:3s} "
            f"Quality={granule.percent_valid_pixels}% "
            f"BBox={granule.bbox}"
    )

        xs = [west, east, east, west, west]
        ys = [south, south, north, north, south]

        ax.plot(
            xs,
            ys,
            color="blue",
            linewidth=1,
        )

        ax.text(
            (west + east) / 2,
            (south + north) / 2,
            str(i),
            fontsize=8,
            color="blue",
            ha="center",
            va="center",
        )

        print(
            f"{i:2d}: "
            f"{granule.platform:3s}  "
            f"{granule.percent_valid_pixels}%"
        )

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------

    ax.set_title("ITS_LIVE Search Results")

    ax.set_xlabel("Longitude")

    ax.set_ylabel("Latitude")

    ax.set_aspect("equal")

    plt.show()
    ax.relim()
    ax.autoscale_view()


if __name__ == "__main__":
    main()