"""Reproduces the headline checks in the findings from the four daily CSVs
(run from the folder containing them). Daily tables are chunk-averaged, so
figures are approximate and used for comparing trucks only."""
import pandas as pd
d = pd.read_csv('daily_pressure_profiles.csv', parse_dates=['Date'])
p = pd.read_csv('daily_dual_pressure_differences.csv', parse_dates=['Date'])
fix = {'KA25B4471': 'KA25AB4471'}          # one truck, two spellings
d['V'] = d.Vehicle.replace(fix); p['V'] = p.Vehicle.replace(fix)
d['ratio'] = d.PSI_Median / d.Cold_Target_PSI

# 1. Sunday (idle, ~28C) as the cold baseline, by truck
sun = d[d.Date.dt.dayofweek == 6].groupby('V').agg(ratio=('ratio', 'mean'), temp=('TEMP_Median', 'mean'))
print('Sunday cold ratio by truck\n', sun.round(3))

# 2. Weekday vs Sunday ratio by truck
d['sun'] = d.Date.dt.dayofweek == 6
print(d.groupby(['V', 'sun']).ratio.mean().unstack().round(3))

# 3. Dual-pair gap series (sawtooth) for the two trucks discussed
for v, pr in [('KA22AD7730', '5R'), ('KA25AB4471', '4L')]:
    x = p[(p.V == v) & (p.Dual_Pair == pr)].groupby('Date').Median_Abs_Diff_PSI.mean().round(1)
    print(v, pr, x.to_dict())

# 4. Weekly median sensor temperature by truck (monsoon onset)
d['wk'] = d.Date.dt.isocalendar().week
print(d.groupby(['wk', 'V']).TEMP_Median.mean().unstack().round(1))

# 5. KA25AC1182 axle-5 right, 8-14 July
x = d[(d.V == 'KA25AC1182') & d.Slot.isin(['5RI', '5RO']) & d.Date.between('2026-07-08', '2026-07-14')]
print(x[['Date', 'Slot', 'PSI_Min', 'PSI_Median', 'PSI_Max', 'BATT_Min']].to_string(index=False))
