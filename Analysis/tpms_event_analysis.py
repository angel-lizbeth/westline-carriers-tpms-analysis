import pandas as pd
from pathlib import Path

INPUT = Path(r"C:\Users\chand\Desktop\Prepared_AI_Package\TPMS_CSV_Analysis\tyresense_all_vehicles_combined.csv")
OUT = INPUT.parent / "TPMS_Event_Analysis"
OUT.mkdir(exist_ok=True)
CHUNK_SIZE = 50_000

# Actual 14 fitted positions; axle 6 and 7 are excluded as unoccupied.
slots = {
    "1LO": 115, "1RO": 115,
    "2LO": 105, "2RO": 105,
    "3LO": 105, "3RO": 105,
    "4LI": 100, "4LO": 100, "4RI": 100, "4RO": 100,
    "5LI": 100, "5LO": 100, "5RI": 100, "5RO": 100,
}
metric_cols = {slot: (f"{slot} PSI", f"{slot} TEMP", f"{slot} BATT") for slot in slots}
required = ["Time", "Vehicle"] + [c for trio in metric_cols.values() for c in trio]
header = pd.read_csv(INPUT, nrows=0).columns.tolist()
missing = [c for c in required if c not in header]
if missing:
    raise SystemExit("Missing expected columns: " + ", ".join(missing))

daily_parts = []
pair_parts = []

# Dual pairs: each inner/outer pair on the same side of axles 4 and 5.
pairs = [
    ("4L", "4LI", "4LO"), ("4R", "4RI", "4RO"),
    ("5L", "5LI", "5LO"), ("5R", "5RI", "5RO"),
]

print("Analyzing valid fitted slots by vehicle and date...")
for chunk_no, chunk in enumerate(pd.read_csv(INPUT, usecols=required, chunksize=CHUNK_SIZE, low_memory=False), start=1):
    chunk["Time"] = pd.to_datetime(chunk["Time"], errors="coerce")
    chunk = chunk.dropna(subset=["Time", "Vehicle"])
    chunk["Date"] = chunk["Time"].dt.date

    # Per slot daily distribution and share below cold target.
    for slot, target in slots.items():
        psi_col, temp_col, batt_col = metric_cols[slot]
        psi = pd.to_numeric(chunk[psi_col], errors="coerce")
        temp = pd.to_numeric(chunk[temp_col], errors="coerce")
        batt = pd.to_numeric(chunk[batt_col], errors="coerce")
        valid = psi.notna()
        if not valid.any():
            continue
        work = chunk.loc[valid, ["Vehicle", "Date"]].copy()
        work["PSI"] = psi.loc[valid]
        work["TEMP"] = temp.loc[valid]
        work["BATT"] = batt.loc[valid]
        work["Below_Cold_Target"] = (work["PSI"] < target).astype("int8")
        grouped = work.groupby(["Vehicle", "Date"], dropna=False)
        agg = grouped.agg(
            Reading_Count=("PSI", "count"),
            PSI_Min=("PSI", "min"),
            PSI_P10=("PSI", lambda x: x.quantile(0.10)),
            PSI_Median=("PSI", "median"),
            PSI_Max=("PSI", "max"),
            TEMP_Median=("TEMP", "median"),
            BATT_Min=("BATT", "min"),
            Below_Target_Count=("Below_Cold_Target", "sum"),
            Below_Target_Share=("Below_Cold_Target", "mean"),
        ).reset_index()
        agg["Slot"] = slot
        agg["Cold_Target_PSI"] = target
        daily_parts.append(agg)

    # Timestamp-level inner/outer pressure difference for duals.
    for pair_name, inner_slot, outer_slot in pairs:
        inner = pd.to_numeric(chunk[f"{inner_slot} PSI"], errors="coerce")
        outer = pd.to_numeric(chunk[f"{outer_slot} PSI"], errors="coerce")
        valid = inner.notna() & outer.notna()
        if valid.any():
            work = chunk.loc[valid, ["Vehicle", "Date"]].copy()
            work["Abs_Difference_PSI"] = (inner.loc[valid] - outer.loc[valid]).abs()
            g = work.groupby(["Vehicle", "Date"])["Abs_Difference_PSI"]
            agg = g.agg(Paired_Count="count", Median_Abs_Diff_PSI="median", P95_Abs_Diff_PSI=lambda x: x.quantile(0.95), Max_Abs_Diff_PSI="max").reset_index()
            agg["Dual_Pair"] = pair_name
            pair_parts.append(agg)

    print(f"Processed chunk {chunk_no}: {min(chunk_no * CHUNK_SIZE, 559179):,} rows")

daily = pd.concat(daily_parts, ignore_index=True)
daily = daily.groupby(["Vehicle", "Date", "Slot", "Cold_Target_PSI"], as_index=False).agg(
    Reading_Count=("Reading_Count", "sum"),
    PSI_Min=("PSI_Min", "min"),
    PSI_P10=("PSI_P10", "min"),
    PSI_Median=("PSI_Median", "mean"),
    PSI_Max=("PSI_Max", "max"),
    TEMP_Median=("TEMP_Median", "mean"),
    BATT_Min=("BATT_Min", "min"),
    Below_Target_Count=("Below_Target_Count", "sum"),
    Below_Target_Share=("Below_Target_Share", "mean"),
)
daily.to_csv(OUT / "daily_pressure_profiles.csv", index=False)

pairs_df = pd.concat(pair_parts, ignore_index=True)
pairs_df = pairs_df.groupby(["Vehicle", "Date", "Dual_Pair"], as_index=False).agg(
    Paired_Count=("Paired_Count", "sum"),
    Median_Abs_Diff_PSI=("Median_Abs_Diff_PSI", "mean"),
    P95_Abs_Diff_PSI=("P95_Abs_Diff_PSI", "max"),
    Max_Abs_Diff_PSI=("Max_Abs_Diff_PSI", "max"),
)
pairs_df.to_csv(OUT / "daily_dual_pressure_differences.csv", index=False)

# Candidate lows are deliberately flagged for review, not labeled as faults.
candidates = daily[(daily["PSI_P10"] < daily["Cold_Target_PSI"]) | (daily["Below_Target_Share"] >= 0.10)].copy()
candidates.sort_values(["Vehicle", "Slot", "Date"]).to_csv(OUT / "candidate_low_pressure_days.csv", index=False)

print("\nDONE.")
print("Files saved in:", OUT)
print("Share daily_pressure_profiles.csv, candidate_low_pressure_days.csv, and daily_dual_pressure_differences.csv.")
print("Note: targets are cold-pressure references; candidate flags require contextual review.")
