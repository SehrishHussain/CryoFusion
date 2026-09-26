import duckdb


CATALOG = (
    "s3://its-live-data/test-space/stac/geoparquet/h3r2"
)


def main():

    con = duckdb.connect()

    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    print("Connected to geoparquet catalog.")

    query = f"""
        SELECT
            id,
            platform,
            datetime,
            start_datetime,
            end_datetime,
            percent_valid_pixels,
            bbox
        FROM read_parquet('{CATALOG}/**/*.parquet')
        WHERE
            bbox.xmax >= 70
            AND bbox.xmin <= 78
            AND bbox.ymax >= 34
            AND bbox.ymin <= 39
        LIMIT 10
    """

    print("\nSearching Karakoram...")
    
    rows = con.execute(query).fetchall()

    print(f"Found {len(rows)} records.")

    for row in rows:
        print()
        print("ID:", row[0])
        print("Platform:", row[1])
        print("Datetime:", row[2])
        print("Start:", row[3])
        print("End:", row[4])
        print("Quality:", row[5])
        print("BBox:", row[6])


if __name__ == "__main__":
    main()