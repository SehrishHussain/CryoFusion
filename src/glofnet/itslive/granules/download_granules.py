"""
Download ITS_LIVE velocity granules.

This module downloads NetCDF velocity granules discovered by
search_granules.py and stores them in the local raw data directory.
"""

from pathlib import Path

import requests

from glofnet.common.paths import RAW_DIRECTORY
from glofnet.itslive.models import GranuleInfo
from glofnet.itslive.granules.search_granules import search_granules



def download_granule(granule: GranuleInfo) -> Path:
    """
    Download a single ITS_LIVE velocity granule.

    Parameters
    ----------
    granule : GranuleInfo
        Granule metadata returned by search_granules().

    Returns
    -------
    Path
        Local path to the downloaded NetCDF file.
    """

    output_dir = RAW_DIRECTORY / "itslive"
    output_dir.mkdir(parents=True, exist_ok=True)

    destination = output_dir / granule.filename

    if destination.exists():
        print(f"✓ Already downloaded: {destination.name}")
        return destination

    print(f"Downloading: {granule.filename}")

    response = requests.get(granule.url, stream=True)
    response.raise_for_status()

   
    
    print("Status:", response.status_code)
    print("Content-Type:", response.headers.get("Content-Type"))
    print("Content-Length:", response.headers.get("Content-Length"))

    with destination.open("wb") as file:
        bytes_written = 0

        for i, chunk in enumerate(
            response.iter_content(chunk_size=1024 * 1024),
            start=1,
        ):
            if not chunk:
                continue

            file.write(chunk)
            bytes_written += len(chunk)

            print(
                f"Chunk {i}: "
                f"{len(chunk):,} bytes "
                f"(total {bytes_written:,})"
            )

        print(f"✓ Saved: {destination}")
        print(f"Final size: {destination.stat().st_size:,} bytes")

    return destination


def download_granules() -> list[Path]:
    """
    Download all matching ITS_LIVE granules.

    Returns
    -------
    list[Path]
        Paths to the downloaded NetCDF files.
    """

    granules = search_granules()

    if not granules:
        raise RuntimeError("No ITS_LIVE granules found.")

    downloaded_files: list[Path] = []

    for granule in granules:
        downloaded_files.append(download_granule(granule))

    return downloaded_files


def main():

    downloaded = download_granules()

    print("\nDownloaded files:\n")
    

    for file in downloaded:
        print(file)


if __name__ == "__main__":
    main()