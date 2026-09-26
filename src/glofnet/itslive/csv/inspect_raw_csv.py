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
    print("ITS_LIVE RAW CSV INSPECTION")
    print("=" * 70)

    print("\nCSV:", CSV_PATH)
    print("Exists:", CSV_PATH.exists())
    print(f"Size: {CSV_PATH.stat().st_size / (1024**3):.2f} GB")

    con = duckdb.connect()

    print("\nReading schema...")

    schema = con.execute(
        """
        DESCRIBE
        SELECT *
        FROM read_csv_auto(
            ?,
            sample_size=1000
        )
        """,
        [str(CSV_PATH)],
    ).fetchall()

    print("\nColumns:")
    print("-" * 70)

    for column in schema:
        print(f"{column[0]:30} {column[1]}")

    print("\nFirst 3 rows:")
    print("-" * 70)

    rows = con.execute(
        """
        SELECT *
        FROM read_csv_auto(
            ?,
            sample_size=1000
        )
        LIMIT 3
        """,
        [str(CSV_PATH)],
    ).fetchall()

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()