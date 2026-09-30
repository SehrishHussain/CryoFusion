import os

import numpy as np
import pandas as pd
import xarray as xr

from glofnet.common.find_glacier import load_glacier
from glofnet.itslive.config import GLACIER_ID


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "data/raw/itslive/temporal_v1"

MANIFEST_PATH = os.path.join(
    OUTPUT_DIR,
    f"{GLACIER_ID}_representative_dates_v1.csv",
)

SMOKE_TEST = True

SMOKE_TEST_YEARS = [
    1988,
    1995,
    2004,
    2015,
    2025,
]

EXPECTED_VARIABLES = [
    "v",
    "vx",
    "vy",
    "v_error",
    "interp_mask",
]


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def get_output_path(date):
    date_string = pd.Timestamp(date).strftime(
        "%Y-%m-%dT%H-%M-%S"
    )

    return os.path.join(
        OUTPUT_DIR,
        f"{GLACIER_ID}_{date_string}.nc",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header("ITS_LIVE REPRESENTATIVE VALIDATION V1")

    # --------------------------------------------------------
    # LOAD GLACIER
    # --------------------------------------------------------

    print("\n===== LOADING GLACIER =====")

    glacier = load_glacier(GLACIER_ID).to_crs("EPSG:32643")

    minx, miny, maxx, maxy = glacier.total_bounds

    print("Glacier:", GLACIER_ID)
    print("Bounds:")
    print("minx:", minx)
    print("miny:", miny)
    print("maxx:", maxx)
    print("maxy:", maxy)

    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    print("\n===== LOADING MANIFEST =====")
    print("Manifest:")
    print(MANIFEST_PATH)

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    print("Manifest columns:")
    print(list(manifest.columns))

    required_columns = {"year", "cube_index", "mid_date"}
    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print(f"Manifest observations: {len(manifest)}")

    # --------------------------------------------------------
    # SELECT OBSERVATIONS
    # --------------------------------------------------------

    if SMOKE_TEST:

        print("\n===== SMOKE TEST =====")

        observations = manifest[
            manifest["year"].isin(SMOKE_TEST_YEARS)
        ].copy()

        # Preserve chronological order for easier inspection.
        observations = observations.sort_values(
            "mid_date"
        ).reset_index(drop=True)

        print(f"Smoke-test years: {SMOKE_TEST_YEARS}")
        print(
            f"Observations to validate: "
            f"{len(observations)}"
        )

    else:

        print("\n===== FULL VALIDATION =====")

        observations = manifest.copy()

        observations = observations.sort_values(
            "mid_date"
        ).reset_index(drop=True)

        print(
            f"Observations to validate: "
            f"{len(observations)}"
        )

    # Make sure every requested smoke-test year was found.
    if SMOKE_TEST:

        found_years = set(
            observations["year"].astype(int)
        )

        missing_years = set(SMOKE_TEST_YEARS) - found_years

        if missing_years:
            raise ValueError(
                "Manifest does not contain requested "
                f"smoke-test years: {sorted(missing_years)}"
            )

    print("\nSelected observations:")

    for position, (_, row) in enumerate(
        observations.iterrows(),
        start=1,
    ):

        year = int(row["year"])
        cube_index = int(row["cube_index"])
        date = pd.Timestamp(row["mid_date"])

        print(
            f"{position:03d}: "
            f"year={year} | "
            f"cube index={cube_index} | "
            f"date={date}"
        )

    # --------------------------------------------------------
    # VALIDATION SUMMARY
    # --------------------------------------------------------

    validated = []
    failed = []
    missing = []

    # --------------------------------------------------------
    # VALIDATE SELECTED OBSERVATIONS
    # --------------------------------------------------------

    for position, (_, row) in enumerate(
        observations.iterrows(),
        start=1,
    ):

        year = int(row["year"])
        cube_index = int(row["cube_index"])

        # IMPORTANT:
        # The manifest column is 'mid_date', not 'date'.
        date = pd.Timestamp(row["mid_date"])

        output_path = get_output_path(date)

        print("\n" + "-" * 60)
        print(
            f"Observation {position}/"
            f"{len(observations)}"
        )
        print("-" * 60)

        print("Year:", year)
        print("Cube index:", cube_index)
        print("Manifest mid_date:", date)
        print("Expected file:", output_path)

        # ----------------------------------------------------
        # FILE CHECK
        # ----------------------------------------------------

        print("\n===== FILE CHECK =====")

        if not os.path.exists(output_path):

            print("File exists: FAIL")
            print(
                "STATUS: MISSING "
                "(not a data-validation failure)"
            )

            missing.append(
                {
                    "year": year,
                    "cube_index": cube_index,
                    "date": str(date),
                    "file": output_path,
                    "reason": "Expected output file does not exist",
                }
            )

            continue

        file_size_mb = (
            os.path.getsize(output_path)
            / (1024 * 1024)
        )

        print(f"File exists: PASS")
        print(f"File size: {file_size_mb:.2f} MB")

        if os.path.getsize(output_path) == 0:

            print("File size: FAIL")

            failed.append(
                {
                    "year": year,
                    "cube_index": cube_index,
                    "date": str(date),
                    "file": output_path,
                    "reason": "File is empty",
                }
            )

            continue

        # ----------------------------------------------------
        # OPEN DATASET
        # ----------------------------------------------------

        ds = None

        try:

            print("\n===== OPENING DATASET =====")

            ds = xr.open_dataset(output_path)

            print("Dataset opened successfully: PASS")

            # ------------------------------------------------
            # DATASET STRUCTURE
            # ------------------------------------------------

            print("\n===== DATASET STRUCTURE =====")
            print(ds)

            # ------------------------------------------------
            # VARIABLE CHECK
            # ------------------------------------------------

            print("\n===== VARIABLE CHECK =====")

            missing_variables = [
                variable
                for variable in EXPECTED_VARIABLES
                if variable not in ds.data_vars
            ]

            if missing_variables:
                raise ValueError(
                    f"Missing variables: {missing_variables}"
                )

            print("All expected variables present: PASS")

            # ------------------------------------------------
            # DIMENSION CHECK
            # ------------------------------------------------

            print("\n===== DIMENSION CHECK =====")

            print("Dimensions:")

            for dimension, size in ds.sizes.items():
                print(f"  {dimension}: {size}")

            if "x" not in ds.sizes:
                raise ValueError("Missing x dimension")

            if "y" not in ds.sizes:
                raise ValueError("Missing y dimension")

            expected_shape = (
                ds.sizes["y"],
                ds.sizes["x"],
            )

            print(
                f"Spatial dimensions: PASS "
                f"(y={expected_shape[0]}, "
                f"x={expected_shape[1]})"
            )

            # ------------------------------------------------
            # COORDINATE CHECK
            # ------------------------------------------------

            print("\n===== COORDINATE CHECK =====")

            for coordinate in ["x", "y"]:

                if coordinate not in ds.coords:
                    raise ValueError(
                        f"Missing coordinate: {coordinate}"
                    )

                values = ds[coordinate].values

                print(
                    f"{coordinate}: "
                    f"shape={values.shape}, "
                    f"dtype={values.dtype}"
                )

            if "mid_date" not in ds.coords:
                raise ValueError(
                    "mid_date coordinate missing"
                )

            print(
                "mid_date:",
                ds["mid_date"].values,
            )

            # ------------------------------------------------
            # DATE CHECK
            # ------------------------------------------------

            print("\n===== OBSERVATION DATE =====")

            dataset_date = pd.Timestamp(
                ds["mid_date"].values
            )

            print("Manifest mid_date:", date)
            print("Dataset mid_date:", dataset_date)

            date_difference = abs(
                (
                    dataset_date - date
                ).total_seconds()
            )

            print(
                "Date difference:",
                date_difference,
                "seconds",
            )

            if date_difference > 1:
                raise ValueError(
                    "Dataset mid_date does not match "
                    "manifest mid_date"
                )

            print("Observation date: PASS")

            # ------------------------------------------------
            # VARIABLE SHAPES
            # ------------------------------------------------

            print("\n===== VARIABLE SHAPES =====")

            for variable in EXPECTED_VARIABLES:

                shape = ds[variable].shape

                print(
                    f"{variable}: "
                    f"shape={shape}, "
                    f"dtype={ds[variable].dtype}"
                )

                if shape != expected_shape:
                    raise ValueError(
                        f"{variable} has unexpected "
                        f"shape: {shape}; "
                        f"expected {expected_shape}"
                    )

            print("Variable shapes: PASS")

            # ------------------------------------------------
            # VALID VELOCITY
            # ------------------------------------------------

            print("\n===== VALID VELOCITY =====")

            v = ds["v"]

            valid = np.isfinite(v.values)

            total_pixels = valid.size
            valid_pixels = int(valid.sum())

            valid_percentage = (
                valid_pixels / total_pixels * 100
            )

            print("Total pixels:", total_pixels)
            print(
                "Valid velocity pixels:",
                valid_pixels,
            )
            print(
                f"Valid percentage: "
                f"{valid_percentage:.2f}%"
            )

            if valid_pixels == 0:
                raise ValueError(
                    "No valid velocity pixels"
                )

            # ------------------------------------------------
            # VELOCITY STATISTICS
            # ------------------------------------------------

            print("\n===== VELOCITY STATISTICS =====")

            v_values = v.values[valid]

            v_min = float(np.min(v_values))
            v_max = float(np.max(v_values))
            v_mean = float(np.mean(v_values))
            v_median = float(np.median(v_values))
            v_std = float(np.std(v_values))

            print("v min:", v_min)
            print("v max:", v_max)
            print("v mean:", v_mean)
            print("v median:", v_median)
            print("v std:", v_std)

            # ------------------------------------------------
            # VELOCITY COMPONENTS
            # ------------------------------------------------

            print("\n===== VELOCITY COMPONENTS =====")

            vx_values = ds["vx"].values
            vy_values = ds["vy"].values

            vx_valid = vx_values[valid]
            vy_valid = vy_values[valid]

            print(
                "vx mean:",
                float(np.mean(vx_valid)),
            )
            print(
                "vx min:",
                float(np.min(vx_valid)),
            )
            print(
                "vx max:",
                float(np.max(vx_valid)),
            )

            print(
                "vy mean:",
                float(np.mean(vy_valid)),
            )
            print(
                "vy min:",
                float(np.min(vy_valid)),
            )
            print(
                "vy max:",
                float(np.max(vy_valid)),
            )

            # ------------------------------------------------
            # VELOCITY CONSISTENCY
            # ------------------------------------------------

            print(
                "\n===== VELOCITY CONSISTENCY ====="
            )

            calculated_v = np.sqrt(
                vx_values ** 2
                + vy_values ** 2
            )

            velocity_difference = np.abs(
                v.values - calculated_v
            )

            valid_difference = (
                velocity_difference[valid]
            )

            mean_difference = float(
                np.mean(valid_difference)
            )

            max_difference = float(
                np.max(valid_difference)
            )

            print(
                "Mean |v - sqrt(vx² + vy²)|:",
                mean_difference,
            )

            print(
                "Max |v - sqrt(vx² + vy²)|:",
                max_difference,
            )

            # ------------------------------------------------
            # VELOCITY ERROR
            # ------------------------------------------------

            print("\n===== VELOCITY ERROR =====")

            v_error_values = (
                ds["v_error"].values[valid]
            )

            print(
                "v_error min:",
                float(np.min(v_error_values)),
            )

            print(
                "v_error max:",
                float(np.max(v_error_values)),
            )

            print(
                "v_error mean:",
                float(np.mean(v_error_values)),
            )

            # ------------------------------------------------
            # SPATIAL EXTENT
            # ------------------------------------------------

            print("\n===== SPATIAL EXTENT =====")

            dataset_minx = float(
                ds["x"].min().item()
            )
            dataset_maxx = float(
                ds["x"].max().item()
            )
            dataset_miny = float(
                ds["y"].min().item()
            )
            dataset_maxy = float(
                ds["y"].max().item()
            )

            print("Dataset X:")
            print("  min:", dataset_minx)
            print("  max:", dataset_maxx)

            print("Dataset Y:")
            print("  min:", dataset_miny)
            print("  max:", dataset_maxy)

            # ------------------------------------------------
            # GLACIER OVERLAP
            # ------------------------------------------------

            print("\n===== GLACIER OVERLAP =====")

            x_overlap = (
                dataset_maxx >= minx
                and dataset_minx <= maxx
            )

            y_overlap = (
                dataset_maxy >= miny
                and dataset_miny <= maxy
            )

            print("X overlap:", x_overlap)
            print("Y overlap:", y_overlap)

            if not (x_overlap and y_overlap):
                raise ValueError(
                    "Dataset does not overlap glacier"
                )

            print(
                "Glacier/data spatial overlap: PASS"
            )

            # ------------------------------------------------
            # INTERP MASK
            # ------------------------------------------------

            print("\n===== INTERP MASK =====")

            interp_mask = ds["interp_mask"]

            print("interp_mask present: PASS")
            print(
                "interp_mask shape:",
                interp_mask.shape,
            )
            print(
                "interp_mask dtype:",
                interp_mask.dtype,
            )

            print(
                "NOTE: interp_mask is preserved "
                "but NOT interpreted in V1."
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            print(
                "\n===== OBSERVATION VALIDATION ====="
            )
            print("STATUS: PASS")

            validated.append(
                {
                    "year": year,
                    "cube_index": cube_index,
                    "date": str(date),
                    "file": output_path,
                    "valid_pixels": valid_pixels,
                    "valid_percentage": valid_percentage,
                    "v_mean": v_mean,
                }
            )

        except Exception as exc:

            print(
                "\n===== OBSERVATION VALIDATION ====="
            )
            print("STATUS: FAIL")
            print("ERROR:", exc)

            failed.append(
                {
                    "year": year,
                    "cube_index": cube_index,
                    "date": str(date),
                    "file": output_path,
                    "reason": str(exc),
                }
            )

        finally:

            if ds is not None:
                ds.close()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_header(
        "REPRESENTATIVE VALIDATION COMPLETE"
    )

    print("\n===== VALIDATION SUMMARY =====")

    print(
        "Requested:",
        len(observations),
    )

    print(
        "Validated successfully:",
        len(validated),
    )

    print(
        "Missing:",
        len(missing),
    )

    print(
        "Failed validation:",
        len(failed),
    )

    # --------------------------------------------------------
    # VALIDATED
    # --------------------------------------------------------

    if validated:

        print("\n===== VALIDATED =====")

        for item in validated:

            print(
                f"{item['year']} | "
                f"{item['date']} | "
                f"valid="
                f"{item['valid_percentage']:.2f}%"
            )

    # --------------------------------------------------------
    # MISSING
    # --------------------------------------------------------

    if missing:

        print("\n===== MISSING =====")

        for item in missing:

            print(
                f"{item['year']} | "
                f"{item['date']} | "
                f"{item['file']}"
            )

    # --------------------------------------------------------
    # FAILED
    # --------------------------------------------------------

    if failed:

        print("\n===== FAILED VALIDATION =====")

        for item in failed:

            print(
                f"{item['year']} | "
                f"{item['date']} | "
                f"{item['reason']}"
            )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    if (
        len(validated) == len(observations)
        and not missing
        and not failed
    ):

        print("\nV1 STATUS: PASS")
        print(
            "All selected representative observations "
            "validated successfully."
        )

    else:

        print("\nV1 STATUS: REVIEW")

        if missing:
            print(
                "Some selected representative observations "
                "are missing. This is expected if the "
                "downloader has not successfully retrieved them."
            )

        if failed:
            print(
                "Some downloaded observations failed "
                "data validation."
            )


if __name__ == "__main__":
    main()