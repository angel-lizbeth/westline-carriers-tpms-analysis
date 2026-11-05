import pandas as pd
from pathlib import Path

INPUT = Path(r"C:\Users\chand\Desktop\Prepared_AI_Package\TPMS_CSV_Analysis\tyresense_all_vehicles_combined.csv")
OUT = INPUT.parent / "TPMS_Profile"
OUT.mkdir(exist_ok=True)

CHUNK_SIZE = 100_000
header = pd.read_csv(INPUT, nrows=0).columns.tolist()

# Actual headers use a space, e.g. "1LO PSI", "1LO TEMP", "1LO BATT".
metrics = ("PSI", "TEMP", "BATT")
sensor_cols = {m: [c for c in header if c.endswith(" " + m)] for m in metrics}
pressure_cols = sensor_cols["PSI"]
temp_cols = sensor_cols["TEMP"]
batt_cols = sensor_cols["BATT"]

if not pressure_cols:
    raise SystemExit("No PSI columns found. Please check the CSV headers.")

print("Detected sensor columns:")
for metric in metrics:
    print(f"{metric}: {len(sensor_cols[metric])}")
print("Reading CSV in chunks...")

coverage = {}
stats = {
    c: {"count": 0, "sum": 0.0, "min": float("inf"), "max": float("-inf")}
    for c in pressure_cols + temp_cols + batt_cols
}
row_count = 0
time_min = None
time_max = None

for chunk_no, chunk in enumerate(pd.read_csv(INPUT, chunksize=CHUNK_SIZE, low_memory=False), start=1):
    row_count += len(chunk)

    if "Vehicle" in chunk.columns and "Source_File" in chunk.columns:
        for key, value in chunk.groupby(["Vehicle", "Source_File"], dropna=False).size().items():
            coverage[key] = coverage.get(key, 0) + int(value)

    if "Time" in chunk.columns:
        times = pd.to_datetime(chunk["Time"], errors="coerce")
        if times.notna().any():
            lo, hi = times.min(), times.max()
            time_min = lo if time_min is None or lo < time_min else time_min
            time_max = hi if time_max is None or hi > time_max else time_max

    for col in stats:
        vals = pd.to_numeric(chunk[col], errors="coerce").dropna()
        if not vals.empty:
            st = stats[col]
            st["count"] += int(vals.count())
            st["sum"] += float(vals.sum())
            st["min"] = min(st["min"], float(vals.min()))
            st["max"] = max(st["max"], float(vals.max()))

    print(f"Processed chunk {chunk_no}: {row_count:,} rows")

coverage_df = pd.DataFrame([
    {"Vehicle": key[0], "Source_File": key[1], "Rows": value}
    for key, value in coverage.items()
]).sort_values(["Vehicle", "Source_File"])
coverage_df.to_csv(OUT / "coverage_by_vehicle_source.csv", index=False)

summary_rows = []
for col, st in stats.items():
    summary_rows.append({
        "Field": col,
        "Non_Null_Count": st["count"],
        "Mean": st["sum"] / st["count"] if st["count"] else None,
        "Min": st["min"] if st["count"] else None,
        "Max": st["max"] if st["count"] else None,
    })
pd.DataFrame(summary_rows).to_csv(OUT / "sensor_field_summary.csv", index=False)

availability_rows = []
for metric in metrics:
    for col in sensor_cols[metric]:
        availability_rows.append({
            "Metric": metric,
            "Slot_Field": col,
            "Non_Null_Count": stats[col]["count"],
            "Share_of_All_Rows": stats[col]["count"] / row_count if row_count else 0,
        })
pd.DataFrame(availability_rows).to_csv(OUT / "sensor_availability_by_slot.csv", index=False)

with open(OUT / "profile_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Rows processed: {row_count:,}\n")
    f.write(f"Time minimum: {time_min}\n")
    f.write(f"Time maximum: {time_max}\n")
    for metric in metrics:
        f.write(f"{metric} fields detected: {len(sensor_cols[metric])}\n")
    f.write(f"Output folder: {OUT}\n")

print("\nFinished successfully.")
print("Output folder:", OUT)
print("Please share profile_summary.txt and sensor_availability_by_slot.csv.")
