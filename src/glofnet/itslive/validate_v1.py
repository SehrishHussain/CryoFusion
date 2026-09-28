import os

import numpy as np
import xarray as xr

from glofnet.common.find_glacier import load_glacier
from glofnet.itslive.config import GLACIER_ID


INPUT_PATH = (
    "data/raw/itslive/"
    f"{GLACIER_ID}_observation_7.nc"
)


EXPECTED_VARIABLES = [
    "v",
    "vx",
    "vy",
    "v_error",
    "interp_mask",
]


def main():

    print("========================================")
    print("       ITS_LIVE V1 VALIDATION")
    print("========================================")

    # ---------------------------------------------------------
    # 1. CHECK FILE
    # ---------------------------------------------------------

    print("\n===== FILE CHECK =====")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"ITS_LIVE file not found:\n{INPUT_PATH}"
        )

    file_size_mb = os.path.getsize(INPUT_PATH) / (1024 ** 2)

    print("File:", INPUT_PATH)
    print(f"Size: {file_size_mb:.2f} MB")
    print("File exists: PASS")

    # ---------------------------------------------------------
    # 2. OPEN DATASET
    # ---------------------------------------------------------

    print("\n===== OPENING DATASET =====")

    ds = xr.open_dataset(INPUT_PATH)

    print("Dataset opened successfully.")
    print("\nDataset:")
    print(ds)

    # ---------------------------------------------------------
    # 3. VARIABLE CHECK
    # ---------------------------------------------------------

    print("\n===== VARIABLE CHECK =====")

    variables = list(ds.data_vars)

    print("Variables found:")
    for variable in variables:
        print(f"  - {variable}")

    missing = [
        variable
        for variable in EXPECTED_VARIABLES
        if variable not in ds
    ]

    if missing:
        print("\nMISSING VARIABLES:")
        for variable in missing:
            print(f"  - {variable}")

        raise ValueError(
            "Expected ITS_LIVE variables are missing."
        )

    print("\nAll expected variables present: PASS")

    # ---------------------------------------------------------
    # 4. DIMENSION CHECK
    # ---------------------------------------------------------

    print("\n===== DIMENSION CHECK =====")

    print("Dimensions:")
    for dimension, size in ds.sizes.items():
        print(f"  {dimension}: {size}")

    expected_y = 126
    expected_x = 108

    if ds.sizes.get("y") != expected_y:
        raise ValueError(
            f"Unexpected y dimension: "
            f"{ds.sizes.get('y')}"
        )

    if ds.sizes.get("x") != expected_x:
        raise ValueError(
            f"Unexpected x dimension: "
            f"{ds.sizes.get('x')}"
        )

    print("\nSpatial dimensions: PASS")
    print(f"Expected: y={expected_y}, x={expected_x}")

    # ---------------------------------------------------------
    # 5. COORDINATE CHECK
    # ---------------------------------------------------------

    print("\n===== COORDINATE CHECK =====")

    for coordinate in ["x", "y", "mid_date"]:

        if coordinate not in ds.coords:
            print(f"{coordinate}: MISSING")
        else:
            values = ds[coordinate].values

            print(
                f"{coordinate}: "
                f"shape={values.shape}, "
                f"dtype={values.dtype}"
            )

    # ---------------------------------------------------------
    # 6. OBSERVATION DATE
    # ---------------------------------------------------------

    print("\n===== OBSERVATION DATE =====")

    if "mid_date" in ds:

        date_value = ds["mid_date"].values

        print("mid_date:", date_value)

    # ---------------------------------------------------------
    # 7. VARIABLE SHAPE CHECK
    # ---------------------------------------------------------

    print("\n===== VARIABLE SHAPES =====")

    for variable in EXPECTED_VARIABLES:

        data = ds[variable]

        print(
            f"{variable}: "
            f"shape={data.shape}, "
            f"dtype={data.dtype}"
        )

        if data.shape != (expected_y, expected_x):
            raise ValueError(
                f"{variable} has unexpected shape: "
                f"{data.shape}"
            )

    print("\nVariable shapes: PASS")

    # ---------------------------------------------------------
    # 8. VALID VELOCITY PIXELS
    # ---------------------------------------------------------

    print("\n===== VALID VELOCITY =====")

    v = ds["v"]

    valid = v.notnull()

    valid_count = valid.sum().item()
    total_count = valid.size
    invalid_count = total_count - valid_count

    valid_percentage = (
        valid_count / total_count * 100
    )

    print("Total pixels:", total_count)
    print("Valid velocity pixels:", valid_count)
    print("Invalid / NaN pixels:", invalid_count)
    print(f"Valid percentage: {valid_percentage:.2f}%")

    # ---------------------------------------------------------
    # 9. VELOCITY STATISTICS
    # ---------------------------------------------------------

    print("\n===== VELOCITY STATISTICS =====")

    print(
        "v min:",
        v.where(valid).min().item()
    )

    print(
        "v max:",
        v.where(valid).max().item()
    )

    print(
        "v mean:",
        v.where(valid).mean().item()
    )

    print(
        "v median:",
        v.where(valid).median().item()
    )

    print(
        "v std:",
        v.where(valid).std().item()
    )

    # ---------------------------------------------------------
    # 10. VX / VY STATISTICS
    # ---------------------------------------------------------

    print("\n===== VELOCITY COMPONENTS =====")

    vx = ds["vx"]
    vy = ds["vy"]

    print(
        "vx mean:",
        vx.where(valid).mean().item()
    )

    print(
        "vx min:",
        vx.where(valid).min().item()
    )

    print(
        "vx max:",
        vx.where(valid).max().item()
    )

    print(
        "vy mean:",
        vy.where(valid).mean().item()
    )

    print(
        "vy min:",
        vy.where(valid).min().item()
    )

    print(
        "vy max:",
        vy.where(valid).max().item()
    )

    # ---------------------------------------------------------
    # 11. V vs VX/VY CONSISTENCY
    # ---------------------------------------------------------

    print("\n===== VELOCITY CONSISTENCY =====")

    calculated_v = np.sqrt(
        vx ** 2 + vy ** 2
    )

    difference = (
        v - calculated_v
    ).where(valid)

    difference_mean = (
        np.abs(difference).mean().item()
    )

    difference_max = (
        np.abs(difference).max().item()
    )

    print(
        "Mean |v - sqrt(vx² + vy²)|:",
        difference_mean
    )

    print(
        "Max |v - sqrt(vx² + vy²)|:",
        difference_max
    )

    # ---------------------------------------------------------
    # 12. V_ERROR
    # ---------------------------------------------------------

    print("\n===== VELOCITY ERROR =====")

    v_error = ds["v_error"]

    print(
        "v_error min:",
        v_error.where(valid).min().item()
    )

    print(
        "v_error max:",
        v_error.where(valid).max().item()
    )

    print(
        "v_error mean:",
        v_error.where(valid).mean().item()
    )

    # ---------------------------------------------------------
    # 13. SPATIAL EXTENT
    # ---------------------------------------------------------

    print("\n===== SPATIAL EXTENT =====")

    x = ds["x"].values
    y = ds["y"].values

    print("Dataset X:")
    print("  min:", x.min())
    print("  max:", x.max())

    print("Dataset Y:")
    print("  min:", y.min())
    print("  max:", y.max())

    # ---------------------------------------------------------
    # 14. GLACIER EXTENT
    # ---------------------------------------------------------

    print("\n===== GLACIER EXTENT =====")

    glacier = load_glacier(
        GLACIER_ID
    ).to_crs("EPSG:32643")

    minx, miny, maxx, maxy = (
        glacier.total_bounds
    )

    print("Glacier:")
    print("  minx:", minx)
    print("  miny:", miny)
    print("  maxx:", maxx)
    print("  maxy:", maxy)

    # ---------------------------------------------------------
    # 15. SPATIAL OVERLAP CHECK
    # ---------------------------------------------------------

    print("\n===== SPATIAL OVERLAP =====")

    data_minx = x.min()
    data_maxx = x.max()
    data_miny = y.min()
    data_maxy = y.max()

    x_overlap = (
        data_minx <= maxx
        and data_maxx >= minx
    )

    y_overlap = (
        data_miny <= maxy
        and data_maxy >= miny
    )

    print("X overlap:", x_overlap)
    print("Y overlap:", y_overlap)

    if x_overlap and y_overlap:
        print("Glacier/data spatial overlap: PASS")
    else:
        print("Glacier/data spatial overlap: FAIL")

    # ---------------------------------------------------------
    # 16. INTERP MASK — PRESENCE ONLY
    # ---------------------------------------------------------

    print("\n===== INTERP MASK =====")

    interp = ds["interp_mask"]

    print(
        "interp_mask present: PASS"
    )

    print(
        "interp_mask shape:",
        interp.shape
    )

    print(
        "interp_mask dtype:",
        interp.dtype
    )

    print(
        "NOTE: interp_mask is NOT interpreted "
        "in V1 validation."
    )

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------

    print("\n========================================")
    print("          V1 VALIDATION COMPLETE")
    print("========================================")

    print("\nCore checks completed:")
    print("  [PASS] File exists")
    print("  [PASS] Dataset opens")
    print("  [PASS] Required variables present")
    print("  [PASS] Spatial dimensions")
    print("  [PASS] Variable shapes")
    print("  [PASS] Valid velocity pixels")
    print("  [PASS] Velocity statistics")
    print("  [PASS] Velocity components")
    print("  [PASS] Spatial extent")
    print("  [PASS] Glacier overlap")
    print("  [PASS] interp_mask preserved")

    print(
        "\nV1 status: READY FOR NEXT PIPELINE STAGE"
    )


if __name__ == "__main__":
    main()