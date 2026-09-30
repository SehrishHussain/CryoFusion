import os

import numpy as np
import pandas as pd
import xarray as xr

from glofnet.common.find_glacier import load_glacier
from glofnet.itslive.config import GLACIER_ID


CUBE_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)

OUTPUT_DIR = "data/raw/itslive/temporal_v1"

MANIFEST_PATH = os.path.join(
    OUTPUT_DIR,
    f"{GLACIER_ID}_representative_dates_v1.csv",
)


def main():

    print("=" * 50)
    print("   ITS_LIVE REPRESENTATIVE DATE SELECTION V1")
    print("=" * 50)

    # --------------------------------------------------
    # LOAD GLACIER
    # --------------------------------------------------

    print("\n===== LOADING GLACIER =====")

    glacier = load_glacier(GLACIER_ID).to_crs("EPSG:32643")

    minx, miny, maxx, maxy = glacier.total_bounds

    print("Glacier:", GLACIER_ID)
    print("Bounds:")
    print("minx:", minx)
    print("miny:", miny)
    print("maxx:", maxx)
    print("maxy:", maxy)

    # --------------------------------------------------
    # OPEN ITS_LIVE
    # --------------------------------------------------

    print("\n===== OPENING ITS_LIVE ZARR =====")

    ds = xr.open_zarr(
        CUBE_URL,
        consolidated=True,
    )

    print("Zarr opened successfully.")

    # --------------------------------------------------
    # SPATIAL SUBSET
    # --------------------------------------------------

    print("\n===== SPATIAL SUBSET =====")

    subset = ds.sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny),
    )

    print("Subset dimensions:")
    print(subset.sizes)

    # --------------------------------------------------
    # LOAD TEMPORAL COORDINATE
    # --------------------------------------------------

    print("\n===== LOADING TEMPORAL COORDINATE =====")

    dates = subset["mid_date"].load().values

    print("Total observations:", len(dates))

    # --------------------------------------------------
    # CONVERT TO DATAFRAME
    # --------------------------------------------------

    print("\n===== BUILDING TEMPORAL TABLE =====")

    df = pd.DataFrame(
        {
            "cube_index": np.arange(len(dates)),
            "mid_date": pd.to_datetime(dates),
        }
    )

    df["year"] = df["mid_date"].dt.year

    # Remove invalid dates if any exist.
    df = df.dropna(subset=["mid_date"])

    print("Valid observations:", len(df))
    print(
        "Year range:",
        df["year"].min(),
        "→",
        df["year"].max(),
    )

    # --------------------------------------------------
    # SELECT REPRESENTATIVE OBSERVATION PER YEAR
    # --------------------------------------------------

    print("\n===== SELECTING REPRESENTATIVE DATES =====")

    selected = []

    for year, group in df.groupby("year"):

        group = group.sort_values("mid_date")

        # Select the temporal middle observation
        # available for this year.
        middle_position = len(group) // 2

        representative = group.iloc[middle_position]

        selected.append(
            {
                "year": int(year),
                "cube_index": int(
                    representative["cube_index"]
                ),
                "mid_date": representative["mid_date"],
            }
        )

    selected_df = pd.DataFrame(selected)

    selected_df = selected_df.sort_values(
        "mid_date"
    ).reset_index(drop=True)

    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

    print("\n===== REPRESENTATIVE OBSERVATIONS =====")

    print(
        "Number of selected observations:",
        len(selected_df),
    )

    for i, row in selected_df.iterrows():

        print(
            f"{i:03d}: "
            f"year={int(row['year'])} | "
            f"cube index={int(row['cube_index'])} | "
            f"date={row['mid_date']}"
        )

    # --------------------------------------------------
    # CHECK TEMPORAL COVERAGE
    # --------------------------------------------------

    print("\n===== TEMPORAL COVERAGE =====")

    print(
        "First selected date:",
        selected_df["mid_date"].min(),
    )

    print(
        "Last selected date:",
        selected_df["mid_date"].max(),
    )

    print(
        "Number of years represented:",
        selected_df["year"].nunique(),
    )

    # --------------------------------------------------
    # CHECK DUPLICATE INDICES
    # --------------------------------------------------

    duplicate_indices = selected_df[
        selected_df["cube_index"].duplicated()
    ]

    print("\n===== DUPLICATE CHECK =====")

    print(
        "Duplicate cube indices:",
        len(duplicate_indices),
    )

    if len(duplicate_indices) == 0:
        print("Duplicate check: PASS")
    else:
        print("Duplicate check: REVIEW")

    # --------------------------------------------------
    # SAVE MANIFEST
    # --------------------------------------------------

    print("\n===== SAVING MANIFEST =====")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    selected_df.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    print("Manifest saved:")
    print(MANIFEST_PATH)

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("   REPRESENTATIVE DATE SELECTION COMPLETE")
    print("=" * 50)

    print("\nSummary:")
    print(
        "Total cube observations:",
        len(df),
    )

    print(
        "Years represented:",
        selected_df["year"].nunique(),
    )

    print(
        "Selected observations:",
        len(selected_df),
    )

    print(
        "Manifest:",
        MANIFEST_PATH,
    )


if __name__ == "__main__":
    main()