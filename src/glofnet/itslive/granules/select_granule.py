"""
Select the best ITS_LIVE granule for the configured glacier.

The best granule is the one containing the largest number of
valid velocity pixels over the glacier.
"""

from dataclasses import dataclass
from pathlib import Path

from glofnet.itslive.granules.download_granules import download_granule
from glofnet.itslive.granules.evaluate_granule import evaluate_granule
from glofnet.itslive.models import GranuleInfo
from glofnet.itslive.granules.search_granules import search_granules


@dataclass
class SelectedGranule:
    """
    Result of the granule selection process.
    """

    granule: GranuleInfo
    path: Path
    valid_pixels: int


def select_granule() -> SelectedGranule:
    """
    Search all candidate granules and select the best one.

    Returns
    -------
    SelectedGranule
        Selected granule together with its downloaded file
        and evaluation score.
    """

    granules = search_granules()

    if not granules:
        raise RuntimeError("No ITS_LIVE granules found.")

    best: SelectedGranule | None = None

    print(f"\nEvaluating {len(granules)} candidate granules...\n")

    for index, granule in enumerate(granules, start=1):

        print("=" * 70)
        print(f"Candidate {index}/{len(granules)}")
        print(f"Platform : {granule.platform}")
        print(f"Quality  : {granule.percent_valid_pixels}%")
        print(granule.filename)

        path = download_granule(granule)

        print("\nChecking glacier pixels...\n")

        valid_pixels = evaluate_granule(path)

        print(f"Valid glacier pixels: {valid_pixels:,}")

        if (
            best is None
            or valid_pixels > best.valid_pixels
        ):
            best = SelectedGranule(
                granule=granule,
                path=path,
                valid_pixels=valid_pixels,
            )

    if best is None or best.valid_pixels == 0:
        raise RuntimeError(
            "No ITS_LIVE granule contains valid glacier pixels."
        )

    print("\n" + "=" * 70)
    print("BEST GRANULE")
    print("=" * 70)
    print(best.granule.filename)
    print(f"Valid glacier pixels: {best.valid_pixels:,}")

    return best


def main():

    selected = select_granule()

    print("\nSelected granule")
    print(selected.granule.filename)

    print("\nDownloaded file")
    print(selected.path)

    print(f"\nValid pixels: {selected.valid_pixels:,}")


if __name__ == "__main__":
    main()