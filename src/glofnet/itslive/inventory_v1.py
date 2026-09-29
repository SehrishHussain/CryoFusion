import xarray as xr
import numpy as np

from glofnet.common.find_glacier import load_glacier
from glofnet.itslive.config import GLACIER_ID


CUBE_URL = (
    "https://its-live-data.s3.amazonaws.com/"
    "datacubes/v2-updated-october2024/N30E070/"
    "ITS_LIVE_vel_EPSG32643_G0120_X450000_Y4050000.zarr"
)


def main():

    print("========================================")
    print("       ITS_LIVE V1 TEMPORAL INVENTORY")
    print("========================================")

    # ---------------------------------------------------------
    # 1. LOAD GLACIER
    # ---------------------------------------------------------

    print("\n===== LOADING GLACIER =====")

    glacier = load_glacier(GLACIER_ID).to_crs("EPSG:32643")

    minx, miny, maxx, maxy = glacier.total_bounds

    print("Glacier:", GLACIER_ID)

    print("Bounds:")
    print("minx:", minx)
    print("miny:", miny)
    print("maxx:", maxx)
    print("maxy:", maxy)

    # ---------------------------------------------------------
    # 2. OPEN ITS_LIVE CUBE
    # ---------------------------------------------------------

    print("\n===== OPENING ITS_LIVE ZARR =====")

    ds = xr.open_zarr(
        CUBE_URL,
        consolidated=True,
    )

    print("Zarr opened successfully.")

    # ---------------------------------------------------------
    # 3. SPATIAL SUBSET
    # ---------------------------------------------------------

    print("\n===== SPATIAL SUBSET =====")

    subset = ds.sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny),
    )

    print("Subset dimensions:")
    print(subset.sizes)

    # ---------------------------------------------------------
    # 4. TEMPORAL DIMENSION
    # ---------------------------------------------------------

    print("\n===== TEMPORAL DIMENSION =====")

    dates = subset["mid_date"]

    number_of_observations = dates.size

    print(
        "Number of observations:",
        number_of_observations
    )

    print(
        "Date dtype:",
        dates.dtype
    )

    # ---------------------------------------------------------
    # 5. FIRST / LAST OBSERVATION
    # ---------------------------------------------------------

    print("\n===== TEMPORAL RANGE =====")

    first_date = dates.isel(mid_date=0).values
    last_date = dates.isel(
        mid_date=number_of_observations - 1
    ).values

    print("First observation:", first_date)
    print("Last observation:", last_date)

    # ---------------------------------------------------------
    # 6. LOAD DATES ONLY
    # ---------------------------------------------------------

    print("\n===== LOADING DATES =====")

    date_values = dates.values

    print("Dates loaded successfully.")

    # Convert to datetime64
    date_values = np.asarray(
        date_values,
        dtype="datetime64[ns]"
    )

    # ---------------------------------------------------------
    # 7. CHECK FOR DUPLICATE DATES
    # ---------------------------------------------------------

    print("\n===== DUPLICATE DATE CHECK =====")

    unique_dates = np.unique(date_values)

    duplicate_count = (
        len(date_values) - len(unique_dates)
    )

    print("Total dates:", len(date_values))
    print("Unique dates:", len(unique_dates))
    print("Duplicate dates:", duplicate_count)

    if duplicate_count == 0:
        print("Duplicate date check: PASS")
    else:
        print("Duplicate date check: REVIEW")

    # ---------------------------------------------------------
    # 8. CHECK TEMPORAL ORDER
    # ---------------------------------------------------------

    print("\n===== TEMPORAL ORDER CHECK =====")

    differences = np.diff(date_values)

    negative_differences = (
        differences < np.timedelta64(0, "ns")
    ).sum()

    print(
        "Out-of-order observations:",
        negative_differences
    )

    if negative_differences == 0:
        print("Temporal ordering: PASS")
    else:
        print("Temporal ordering: REVIEW")

    # ---------------------------------------------------------
    # 9. TEMPORAL GAPS
    # ---------------------------------------------------------

    print("\n===== TEMPORAL GAPS =====")

    gap_days = (
        differences
        / np.timedelta64(1, "D")
    )

    if len(gap_days) > 0:

        print(
            "Minimum gap (days):",
            gap_days.min()
        )

        print(
            "Maximum gap (days):",
            gap_days.max()
        )

        print(
            "Mean gap (days):",
            gap_days.mean()
        )

        print(
            "Median gap (days):",
            np.median(gap_days)
        )

    # ---------------------------------------------------------
    # 10. LARGE TEMPORAL GAPS
    # ---------------------------------------------------------

    print("\n===== LARGE TEMPORAL GAPS =====")

    threshold_days = 365

    large_gap_indices = np.where(
        gap_days > threshold_days
    )[0]

    print(
        f"Gaps > {threshold_days} days:",
        len(large_gap_indices)
    )

    # Print only the first 20 large gaps
    if len(large_gap_indices) > 0:

        print("\nFirst large gaps:")

        for index in large_gap_indices[:20]:

            start = date_values[index]
            end = date_values[index + 1]

            gap = gap_days[index]

            print(
                f"{start} -> {end} "
                f"({gap:.1f} days)"
            )

    # ---------------------------------------------------------
    # 11. YEAR DISTRIBUTION
    # ---------------------------------------------------------

    print("\n===== OBSERVATIONS BY YEAR =====")

    years = (
        date_values
        .astype("datetime64[Y]")
        .astype(int)
        + 1970
    )

    unique_years, year_counts = np.unique(
        years,
        return_counts=True
    )

    for year, count in zip(
        unique_years,
        year_counts
    ):
        print(
            f"{year}: {count} observations"
        )

    # ---------------------------------------------------------
    # 12. SAMPLE DATES
    # ---------------------------------------------------------

    print("\n===== SAMPLE DATES =====")

    sample_count = 10

    print("\nFirst observations:")

    for i in range(
        min(sample_count, len(date_values))
    ):
        print(
            f"{i}: {date_values[i]}"
        )

    print("\nLast observations:")

    start_index = max(
        0,
        len(date_values) - sample_count
    )

    for i in range(
        start_index,
        len(date_values)
    ):
        print(
            f"{i}: {date_values[i]}"
        )

    # ---------------------------------------------------------
    # 13. CHECK SPATIAL GRID
    # ---------------------------------------------------------

    print("\n===== SPATIAL GRID =====")

    x = subset["x"].values
    y = subset["y"].values

    print("X pixels:", len(x))
    print("Y pixels:", len(y))

    print("X min:", x.min())
    print("X max:", x.max())

    print("Y min:", y.min())
    print("Y max:", y.max())

    # ---------------------------------------------------------
    # 14. IMPORTANT: DO NOT LOAD VELOCITY DATA
    # ---------------------------------------------------------

    print("\n===== DATA DOWNLOAD STATUS =====")

    print(
        "No velocity arrays were downloaded."
    )

    print(
        "Only the temporal coordinate "
        "and spatial metadata were inspected."
    )

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------

    print("\n========================================")
    print("       TEMPORAL INVENTORY COMPLETE")
    print("========================================")

    print("\nSummary:")
    print(
        "Observations:",
        number_of_observations
    )

    print(
        "First date:",
        first_date
    )

    print(
        "Last date:",
        last_date
    )

    print(
        "Unique dates:",
        len(unique_dates)
    )

    print(
        "Large gaps > 365 days:",
        len(large_gap_indices)
    )

    print(
        "\nNext step: use this inventory to "
        "design the V1 temporal downloader."
    )


if __name__ == "__main__":
    main()