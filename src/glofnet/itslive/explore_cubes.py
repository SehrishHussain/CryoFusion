"""
Explore the ITS_LIVE STAC catalog.

This script is for exploratory purposes only. It helps us understand the
ITS_LIVE STAC API before implementing the production search_cube.py module.

Workflow
--------
1. Load the glacier from the RGI inventory.
2. Compute its bounding box.
3. Connect to the ITS_LIVE STAC catalog.
4. Search for ITS_LIVE Zarr cubes intersecting the glacier.
5. Inspect returned STAC Items.
"""

from pystac_client import Client

from glofnet.common.console import (
    print_header,
    print_section,
    print_key_value,
    print_success,
    print_warning,
)
from glofnet.common.find_glacier import load_glacier
from glofnet.itslive.config import (
    GLACIER_ID,
    STAC_URL,
    COLLECTION_NAME,
    MAX_ITEMS,
)


def connect_to_catalog():
    """Connect to the ITS_LIVE STAC catalog."""

    print_section("Connecting to ITS_LIVE STAC")

    catalog = Client.open(STAC_URL)

    print_success("Connected successfully")

    return catalog


def compute_bbox(glacier):
    """
    Compute the glacier bounding box.

    Returns
    -------
    list
        Bounding box in STAC format:
        [west, south, east, north]
    """

    west, south, east, north = glacier.total_bounds

    print_section("Bounding Box")

    print_key_value("West", west)
    print_key_value("South", south)
    print_key_value("East", east)
    print_key_value("North", north)

    return [west, south, east, north]


def search_cubes(catalog, bbox):
    """Search the ITS_LIVE cube collection."""

    print_section("Searching ITS_LIVE Cubes")

    search = catalog.search(
        collections=[COLLECTION_NAME],
        bbox=bbox,
        max_items=MAX_ITEMS,
    )

    items = list(search.items())

    print_key_value("Matching Cubes", len(items))

    return items


def print_assets(item):
    """Print all assets contained in a STAC Item."""

    print_section("Assets")

    for asset_name, asset in item.assets.items():

        print_key_value("Asset", asset_name)
        print_key_value("Title", asset.title)
        print_key_value("Media Type", asset.media_type)
        print_key_value("URL", asset.href)

        print()


def print_properties(item):
    """Print STAC properties."""

    print_section("Properties")

    for key, value in item.properties.items():
        print_key_value(key, value)


def print_item_summary(item):
    """Print a summary of one STAC Item."""

    print_section("Cube Summary")

    print_key_value("ID", item.id)
    print_key_value("Collection", item.collection_id)
    print_key_value("BBox", item.bbox)
    print_key_value("Geometry", item.geometry["type"])


def main():

    print_header("ITS_LIVE STAC Exploration")

    glacier = load_glacier(GLACIER_ID)

    print_success("Glacier loaded")

    print_key_value("Glacier ID", GLACIER_ID)

    bbox = compute_bbox(glacier)

    catalog = connect_to_catalog()

    items = search_cubes(catalog, bbox)

    if len(items) == 0:

        print_warning("No ITS_LIVE cubes found.")

        return

    for i, item in enumerate(items, start=1):

        print_header(f"Cube {i}")

        print_item_summary(item)

        print_assets(item)

        print_properties(item)


if __name__ == "__main__":
    main()