from pathlib import Path

import duckdb


CSV_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "raw"
    / "itslive"
    / "velocity_data_raw"
    / "df_raw.csv"
)


def main():

    print("=" * 70)
    print("ITS_LIVE RAW DATA - KARAKORAM COVERAGE")
    print("=" * 70)

    con = duckdb.connect()

    # Shisper glacier region
    min_lon = 74.50
    max_lon = 74.75
    min_lat = 36.30
    max_lat = 36.55

    print("\nSearch region:")
    print(f"Longitude: {min_lon} → {max_lon}")
    print(f"Latitude : {min_lat} → {max_lat}")

    # ------------------------------------------------------------
    # Coverage summary
    # ------------------------------------------------------------

    query = """
        SELECT
            COUNT(*) AS observations,
            MIN(mid_date) AS earliest,
            MAX(mid_date) AS latest,
            MIN("v [m/yr]") AS min_velocity,
            MAX("v [m/yr]") AS max_velocity,
            AVG("v [m/yr]") AS mean_velocity
        FROM read_csv_auto(
            ?,
            sample_size=1000
        )
        WHERE
            lon BETWEEN ? AND ?
            AND lat BETWEEN ? AND ?
    """

    result = con.execute(
        query,
        [
            str(CSV_PATH),
            min_lon,
            max_lon,
            min_lat,
            max_lat,
        ],
    ).fetchone()

    print("\nCoverage")
    print("-" * 70)

    print("Observations :", result[0])
    print("Earliest     :", result[1])
    print("Latest       :", result[2])
    print("Min velocity :", result[3], "m/yr")
    print("Max velocity :", result[4], "m/yr")
    print("Mean velocity:", result[5], "m/yr")

    # ------------------------------------------------------------
    # Sample observations
    # ------------------------------------------------------------

    query = """
        SELECT
            mid_date,
            lon,
            lat,
            "v [m/yr]",
            "v_error [m/yr]",
            "vx [m/yr]",
            "vx_error [m/yr]",
            "vy [m/yr]",
            "vy_error [m/yr]",
            "date_dt [days]"
        FROM read_csv_auto(
            ?,
            sample_size=1000
        )
        WHERE
            lon BETWEEN ? AND ?
            AND lat BETWEEN ? AND ?
        ORDER BY mid_date
        LIMIT 20
    """

    rows = con.execute(
        query,
        [
            str(CSV_PATH),
            min_lon,
            max_lon,
            min_lat,
            max_lat,
        ],
    ).fetchall()

    print("\nSample observations")
    print("-" * 70)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()