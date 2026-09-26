"""
Debug the ITS_LIVE STAC search.

This script performs several independent searches to determine why
the filtered search is returning zero granules.
"""

from pystac_client import Client

from glofnet.common.find_glacier import load_glacier
from glofnet.itslive.config import (
    COLLECTION_NAME,
    END_DATE,
    GLACIER_ID,
    START_DATE,
    STAC_URL,
)


def connect_to_catalog():
    """Connect to the ITS_LIVE STAC catalog."""
    return Client.open(STAC_URL)


def compute_bbox(glacier):
    west, south, east, north = glacier.total_bounds
    return [west, south, east, north]


def print_search_results(title, search):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    try:
        url = search.url_with_parameters()

        print("Request:")

        if len(url) > 200:
            print(url[:200] + " ...")
            print(f"(URL length: {len(url):,} characters)")
        else:
            print(url)

    except Exception:
        print("Request unavailable")

    items = list(search.items())

    print(f"\nReturned {len(items)} item(s)\n")

    for item in items[:5]:
        print(item.id)
        print("BBox     :", item.bbox)
        print("Datetime :", item.properties.get("datetime"))
        print()

    return items


def search_granules():

    glacier = load_glacier(GLACIER_ID)

    bbox = compute_bbox(glacier)

    centroid = (
        float(glacier.cenlon.iloc[0]),
        float(glacier.cenlat.iloc[0]),
    )

    catalog = connect_to_catalog()

    print("=" * 80)
    print("GLACIER")
    print("=" * 80)

    print(glacier[["rgi_id", "cenlon", "cenlat"]])

    print("\nBounding box")
    print(bbox)

    print("\nCentroid")
    print(centroid)

    print("\n" + "=" * 80)
    print("CATALOG")
    print("=" * 80)

    print("STAC URL :", STAC_URL)
    print("Collection:", COLLECTION_NAME)

    print("\nCollections:")

    for collection in catalog.get_collections():
        print("-", collection.id)

    #
    # ------------------------------------------------------------
    # TEST 0
    # Collection metadata
    # ------------------------------------------------------------
    #

    print("\n" + "=" * 80)
    print("TEST 0 - COLLECTION METADATA")
    print("=" * 80)

    collection = catalog.get_collection(COLLECTION_NAME)

    print("Spatial extent:")
    print(collection.extent.spatial.bboxes)

    print()

    print("Temporal extent:")
    print(collection.extent.temporal.intervals)

    #
    # ------------------------------------------------------------
    # TEST 1
    # Collection only
    # ------------------------------------------------------------
    #

    search = catalog.search(
        collections=[COLLECTION_NAME],
        max_items=5,
    )

    print_search_results(
        "TEST 1 - COLLECTION ONLY",
        search,
    )

    #
    # ------------------------------------------------------------
    # TEST 2
    # Datetime only
    # ------------------------------------------------------------
    #

    search = catalog.search(
        collections=[COLLECTION_NAME],
        datetime=f"{START_DATE}/{END_DATE}",
        max_items=5,
    )

    print_search_results(
        "TEST 2 - DATETIME",
        search,
    )

    #
    # ------------------------------------------------------------
    # TEST 3
    # Glacier bbox
    # ------------------------------------------------------------
    #

    search = catalog.search(
        collections=[COLLECTION_NAME],
        bbox=bbox,
        max_items=5,
    )

    print_search_results(
        "TEST 3 - GLACIER BBOX",
        search,
    )

    #
    # ------------------------------------------------------------
    # TEST 4
    # Glacier bbox + datetime
    # ------------------------------------------------------------
    #

    search = catalog.search(
        collections=[COLLECTION_NAME],
        bbox=bbox,
        datetime=f"{START_DATE}/{END_DATE}",
        max_items=5,
    )

    print_search_results(
        "TEST 4 - GLACIER BBOX + DATETIME",
        search,
    )

    #
    # ------------------------------------------------------------
    # TEST 5
    # Polygon intersects
    # ------------------------------------------------------------
    #

    search = catalog.search(
        collections=[COLLECTION_NAME],
        intersects=glacier.geometry.iloc[0].__geo_interface__,
        max_items=5,
    )

    print_search_results(
        "TEST 5 - POLYGON INTERSECTS",
        search,
    )

    #
    # ------------------------------------------------------------
    # TEST 6
    # Point intersects
    # ------------------------------------------------------------
    #

    point = {
        "type": "Point",
        "coordinates": [centroid[0], centroid[1]],
    }

    search = catalog.search(
        collections=[COLLECTION_NAME],
        intersects=point,
        max_items=5,
    )

    print_search_results(
        "TEST 6 - POINT INTERSECTS",
        search,
    )

    #
    # ------------------------------------------------------------
    # TEST 7
    # Large Karakoram bbox
    # ------------------------------------------------------------
    #

    karakoram_bbox = [
        70.0,
        34.0,
        78.0,
        39.0,
    ]

    search = catalog.search(
        collections=[COLLECTION_NAME],
        bbox=karakoram_bbox,
        max_items=5,
    )

    print_search_results(
        "TEST 7 - KARAKORAM BBOX",
        search,
    )


def main():

    search_granules()

    print("\nDebugging complete.")


if __name__ == "__main__":
    main()