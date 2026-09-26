"""
Download and display ITS_LIVE overview images for candidate granules.
"""

from pathlib import Path
import webbrowser

import requests

from glofnet.itslive.granules.search_granules import connect_to_catalog
from glofnet.itslive.config import (
    COLLECTION_NAME,
    START_DATE,
    END_DATE,
    MAX_ITEMS,
    GLACIER_ID,
)
from glofnet.common.find_glacier import load_glacier


OUTPUT_DIRECTORY = Path("debug_overviews")


def main():

    glacier = load_glacier(GLACIER_ID)

    west, south, east, north = glacier.total_bounds

    catalog = connect_to_catalog()

    search = catalog.search(
        collections=[COLLECTION_NAME],
        bbox=[west, south, east, north],
        datetime=f"{START_DATE}/{END_DATE}",
        max_items=MAX_ITEMS,
    )

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    for i, item in enumerate(search.items(), start=1):

        overview = item.assets.get("overview")

        if overview is None:
            continue

        filename = OUTPUT_DIRECTORY / f"{i:02d}_{item.id}.png"

        print(f"Downloading overview {i}")
        print(item.id)

        response = requests.get(overview.href)
        response.raise_for_status()

        filename.write_bytes(response.content)

        print(f"Saved: {filename}")

        # Automatically open it
        webbrowser.open(filename.resolve().as_uri())

        # Only inspect the first two for now
        if i == 2:
            break


if __name__ == "__main__":
    main()