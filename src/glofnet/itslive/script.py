from datetime import date

import itslive


bbox = [74.50, 36.35, 74.65, 36.48]

print("Starting streaming search...")

results = itslive.velocity_pairs.find_streaming(
    bbox=bbox,
    start=date(2026, 7, 1),
    end=date(2026, 8, 9),
    engine="duckdb",
)

count = 0

for url in results:
    print(url)
    count += 1

    if count >= 5:
        break

print(f"\nFound at least {count} pairs.")