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

    con = duckdb.connect()

    min_lon = 74.50
    max_lon = 74.75
    min_lat = 36.30
    max_lat = 36.55

    print("=" * 70)
    print("ITS_LIVE RAW DATA - TEMPORAL/SPATIAL ANALYSIS")
    print("=" * 70)

    query = """
        SELECT
            COUNT(*) AS observations,

            COUNT(DISTINCT CAST(mid_date AS DATE))
                AS unique_dates,

            COUNT(DISTINCT
                CONCAT(
                    CAST(lon AS VARCHAR),
                    '_',
                    CAST(lat AS VARCHAR)
                )
            ) AS unique_locations,

            COUNT(DISTINCT
                CONCAT(
                    CAST(lon AS VARCHAR),
                    '_',
                    CAST(lat AS VARCHAR),
                    '_',
                    CAST(mid_date AS DATE)
                )
            ) AS location_date_pairs

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

    print("\nResults")
    print("-" * 70)

    print("Observations       :", result[0])
    print("Unique dates       :", result[1])
    print("Unique locations   :", result[2])
    print("Location-date pairs:", result[3])

    # ------------------------------------------------------------
    # Observations by year
    # ------------------------------------------------------------

    query = """
        SELECT
            EXTRACT(YEAR FROM mid_date) AS year,
            COUNT(*) AS observations
        FROM read_csv_auto(
            ?,
            sample_size=1000
        )
        WHERE
            lon BETWEEN ? AND ?
            AND lat BETWEEN ? AND ?
        GROUP BY year
        ORDER BY year
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

    print("\nObservations by year")
    print("-" * 70)

    for year, count in rows:
        print(f"{int(year)} : {count:,}")


if __name__ == "__main__":
    main()