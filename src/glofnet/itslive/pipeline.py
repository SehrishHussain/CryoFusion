"""
Complete ITS_LIVE data pipeline.

Workflow
--------
1. Search the ITS_LIVE STAC catalog.
2. Evaluate candidate granules.
3. Select the best granule.
4. Preprocess the selected granule.
"""

from glofnet.itslive.preprocess.pipeline import (
    run_pipeline as preprocess_pipeline,
)
from glofnet.itslive.granules.select_granule import select_granule


def run_pipeline():
    """
    Execute the complete ITS_LIVE workflow.
    """

    print("=" * 70)
    print("ITS_LIVE DATA PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------------
    # Search, evaluate and select the best granule.
    # --------------------------------------------------------------

    print("\n[1/2] Selecting best granule...")

    selected = select_granule()

    print("\nSelected granule:")
    print(selected.granule.filename)

    print(f"Valid glacier pixels: {selected.valid_pixels:,}")

    print("\nLocal dataset:")
    print(selected.path)

    # --------------------------------------------------------------
    # Preprocess the selected granule.
    # --------------------------------------------------------------

    print("\n[2/2] Running preprocessing pipeline...")

    final_dataset = preprocess_pipeline(
        selected.path
    )

    print("\nFinal dataset:")
    print(final_dataset)

    print("\n" + "=" * 70)
    print("ITS_LIVE PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return final_dataset


def main():

    run_pipeline()


if __name__ == "__main__":
    main()