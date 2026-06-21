"""
EDA and Feature Engineering for Traffic Event Data (ASTRAM Bengaluru)
=====================================================================
Dataset: ~8,173 traffic events (Jan–Apr 2024)
Run: py -3 eda_feature_engineering.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_PATH = Path(r"c:\Users\Khushi\Downloads\Traffic\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")
OUTPUT_DIR = Path(r"c:\Users\Khushi\Downloads\Traffic\eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Global plot style
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "font.family": "DejaVu Sans", "font.size": 11})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SECTION 1: Loading Data")
print("═"*60)

df = pd.read_csv(CSV_PATH, low_memory=False)

print(f"Raw shape: {df.shape}")
print(f"Columns : {list(df.columns)}\n")

# ── 1a. Parse start_datetime: comes as UTC string with timezone offset
# Replace string 'NULL' with NaN first, then parse
df['start_datetime'] = df['start_datetime'].replace('NULL', np.nan)
df['start_datetime_utc'] = pd.to_datetime(df['start_datetime'], utc=True, errors='coerce')

# Convert UTC → IST (UTC+5:30) using tz_convert
df['start_datetime_ist'] = df['start_datetime_utc'].dt.tz_convert('Asia/Kolkata')

# ── 1b. Parse closed_datetime and resolved_datetime similarly
for col in ['closed_datetime', 'resolved_datetime']:
    df[col] = df[col].replace('NULL', np.nan)
    df[col + '_utc'] = pd.to_datetime(df[col], utc=True, errors='coerce')

print(f"Datetime parse success: {df['start_datetime_ist'].notna().sum()} / {len(df)} rows")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TEMPORAL FEATURES (IST-based)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SECTION 2: Temporal Feature Engineering")
print("═"*60)

# ── 2a. Basic time decomposition from IST datetime
df['hour_of_day']  = df['start_datetime_ist'].dt.hour          # 0–23 IST
df['day_of_week']  = df['start_datetime_ist'].dt.dayofweek     # 0=Mon, 6=Sun
df['day_name']     = df['start_datetime_ist'].dt.day_name()    # 'Monday' etc.
df['month']        = df['start_datetime_ist'].dt.month         # 1–12
df['month_name']   = df['start_datetime_ist'].dt.month_name()  # 'January' etc.
df['date']         = df['start_datetime_ist'].dt.date          # for daily aggregations

# ── 2b. Weekend flag (Saturday=5, Sunday=6)
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# ── 2c. Peak hour flag
# IST peak hours: 19:00–22:00 (evening rush) and 04:00–06:00 (early morning fleet)
def is_peak(hour):
    return int((19 <= hour <= 22) or (4 <= hour <= 6))

df['is_peak_hour'] = df['hour_of_day'].apply(is_peak)

print(f"Peak hour events   : {df['is_peak_hour'].sum()} ({df['is_peak_hour'].mean()*100:.1f}%)")
print(f"Weekend events     : {df['is_weekend'].sum()} ({df['is_weekend'].mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NULL HANDLING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SECTION 3: Null Handling")
print("═"*60)

# ── Replace 'NULL' strings with NaN across the whole df for clean null counting
df.replace('NULL', np.nan, inplace=True)

# ── 3a. Fill 'zone' using police_station as a proxy
# Logic: for each police_station, find the most frequent non-null zone (mode)
# then impute that zone for rows where zone is null but police_station is known

# Build mapping: police_station → most common zone
ps_zone_map = (
    df.dropna(subset=['zone', 'police_station'])
      .groupby('police_station')['zone']
      .agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
      .to_dict()
)

null_zone_before = df['zone'].isna().sum()

df['zone'] = df.apply(
    lambda row: ps_zone_map.get(row['police_station'], row['zone'])
                if pd.isna(row['zone']) and pd.notna(row['police_station'])
                else row['zone'],
    axis=1
)

null_zone_after = df['zone'].isna().sum()
print(f"Zone nulls: {null_zone_before} → {null_zone_after} (filled {null_zone_before - null_zone_after} using police_station proxy)")

# ── 3b. Fill 'junction' with literal "Unknown"
null_junc_before = df['junction'].isna().sum()
df['junction'] = df['junction'].fillna('Unknown')
print(f"Junction nulls filled with 'Unknown': {null_junc_before} rows")

# ── 3c. Fill corridor nulls with 'Non-corridor' (consistent with existing data)
df['corridor'] = df['corridor'].fillna('Non-corridor')

# ── 3d. Standardise priority and requires_road_closure
df['priority'] = df['priority'].fillna('Low')
df['requires_road_closure'] = df['requires_road_closure'].astype(str).str.upper().str.strip()
df['requires_road_closure'] = df['requires_road_closure'].map({'TRUE': True, 'FALSE': False}).fillna(False)
df['requires_road_closure'] = df['requires_road_closure'].astype(bool)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VEHICLE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SECTION 4: Vehicle Feature Engineering")
print("═"*60)

# ── 4a. Normalise veh_type: lowercase + strip whitespace
df['veh_type'] = df['veh_type'].astype(str).str.lower().str.strip()
df['veh_type'] = df['veh_type'].replace({'nan': 'unknown', '': 'unknown', '0': 'unknown'})

# ── 4b. is_heavy_vehicle flag
# Heavy / lane-blocking vehicles that cause secondary cascades
HEAVY_TYPES = {'heavy_vehicle', 'truck', 'bmtc_bus', 'ksrtc_bus', 'private_bus'}
df['is_heavy_vehicle'] = df['veh_type'].isin(HEAVY_TYPES).astype(int)

# ── 4c. veh_category: grouped functional category
def categorise_vehicle(vt):
    if vt in {'bmtc_bus', 'ksrtc_bus'}:
        return 'public_transit'
    elif vt in {'heavy_vehicle', 'truck', 'lcv'}:
        return 'freight'
    elif vt in {'private_car', 'private_bus', 'taxi', 'auto'}:
        return 'private'
    else:
        return 'unknown'

df['veh_category'] = df['veh_type'].apply(categorise_vehicle)

print("Vehicle category distribution:")
print(df['veh_category'].value_counts().to_string())
print(f"\nHeavy vehicles: {df['is_heavy_vehicle'].sum()} ({df['is_heavy_vehicle'].mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RESOLUTION TIME
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SECTION 5: Resolution Time")
print("═"*60)

# ── 5a. resolution_minutes = (closed_datetime_utc - start_datetime_utc) in minutes
# Using total_seconds() (not .seconds) to handle durations > 1 hour correctly
df['resolution_minutes'] = (
    (df['closed_datetime_utc'] - df['start_datetime_utc'])
    .dt.total_seconds() / 60
)

# ── 5b. Cap at 99th percentile to remove extreme outliers (e.g., Debris: 19,190 min)
p99 = df['resolution_minutes'].quantile(0.99)
df['resolution_minutes_capped'] = df['resolution_minutes'].clip(upper=p99)

# ── 5c. Flag: is the event still open (no resolution time available)?
df['is_open'] = df['resolution_minutes'].isna().astype(int)

valid_res = df['resolution_minutes'].dropna()
print(f"Resolution time available for {len(valid_res)} events")
print(f"  Mean  : {valid_res.mean():.0f} min")
print(f"  Median: {valid_res.median():.0f} min")
print(f"  99th %: {p99:.0f} min  (cap applied)")
print(f"  Open events (no close time): {df['is_open'].sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CORRIDOR RISK SCORE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SECTION 6: Corridor Risk Score")
print("═"*60)

# ── corridor_risk_score: count of High-priority events per corridor, normalised 0→1
# Computed on the FULL dataset as a static lookup (no train/test leakage concern for EDA)
corridor_high = (
    df[df['priority'] == 'High']
    .groupby('corridor')['id']
    .count()
    .rename('high_priority_count')
)

# Min-max normalisation
min_val = corridor_high.min()
max_val = corridor_high.max()
corridor_risk = ((corridor_high - min_val) / (max_val - min_val)).rename('corridor_risk_score')

# Merge back onto main df
df = df.merge(corridor_risk.reset_index(), on='corridor', how='left')
df['corridor_risk_score'] = df['corridor_risk_score'].fillna(0.0)

print("Top 10 corridors by risk score:")
print(corridor_risk.sort_values(ascending=False).head(10).round(3).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TARGET VARIABLE: congestion_severity
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SECTION 7: Target Variable — congestion_severity")
print("═"*60)

# ── Three-class ordinal target
# CRITICAL: High priority AND road closure required  → most severe
# MEDIUM  : High priority OR road closure required   → moderate
# LOW     : neither                                  → baseline

def assign_severity(row):
    high     = row['priority'] == 'High'
    closure  = row['requires_road_closure'] == True
    if high and closure:
        return 'high'
    elif high or closure:
        return 'medium'
    else:
        return 'low'

df['congestion_severity'] = df.apply(assign_severity, axis=1)

print("Target class distribution:")
print(df['congestion_severity'].value_counts())
print(df['congestion_severity'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — EVENT CAUSE: MERGE RARE CLASSES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SECTION 8: Event Cause — Merging Rare Classes")
print("═"*60)

# ── Standardise event_cause first
df['event_cause'] = df['event_cause'].astype(str).str.lower().str.strip()

# ── Drop true noise rows
df = df[df['event_cause'] != 'test_demo'].copy()

# ── Merge low-frequency civic causes into a single 'civic_event' class
CIVIC_CAUSES = {'public_event', 'procession', 'vip_movement', 'protest'}
df['event_cause_grouped'] = df['event_cause'].apply(
    lambda x: 'civic_event' if x in CIVIC_CAUSES else x
)

print("Original event_cause counts:")
print(df['event_cause'].value_counts().to_string())
print("\nGrouped event_cause_grouped counts:")
print(df['event_cause_grouped'].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PRINT SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SECTION 9: Summary Statistics")
print("═"*60)

print(f"\nFinal df.shape: {df.shape}")

engineered_features = [
    'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'is_peak_hour',
    'is_heavy_vehicle', 'veh_category', 'resolution_minutes_capped',
    'corridor_risk_score', 'congestion_severity', 'event_cause_grouped',
    'is_open'
]
print(f"\nEngineered features ({len(engineered_features)}):")
for f in engineered_features:
    print(f"  • {f}")

print(f"\nTarget class distribution (congestion_severity):")
print(df['congestion_severity'].value_counts().to_string())

print(f"\nNull counts for key columns (after cleaning):")
key_cols = ['zone', 'junction', 'corridor', 'priority', 'veh_type',
            'resolution_minutes', 'event_cause_grouped', 'start_datetime_ist']
for col in key_cols:
    n = df[col].isna().sum()
    pct = n / len(df) * 100
    print(f"  {col:<30} {n:>5} ({pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SECTION 10: Generating Plots")
print("═"*60)

# ─────────────────────────────────────────────
# PLOT 1: Hourly Distribution of Events (IST)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

hourly = df['hour_of_day'].value_counts().sort_index()
colors = ['#e74c3c' if is_peak(h) else '#3498db' for h in hourly.index]

bars = ax.bar(hourly.index, hourly.values, color=colors, edgecolor='white', linewidth=0.5, width=0.7)

# Annotate bars
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 5, str(int(h)),
            ha='center', va='bottom', fontsize=7.5, color='#2c3e50')

# Peak hour shading
ax.axvspan(3.5, 6.5, alpha=0.10, color='orange', label='Peak: 04–06 IST')
ax.axvspan(18.5, 22.5, alpha=0.10, color='red', label='Peak: 19–22 IST')

ax.set_xlabel('Hour of Day (IST)', fontsize=12)
ax.set_ylabel('Number of Events', fontsize=12)
ax.set_title('Hourly Distribution of Traffic Events (IST)', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 24))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=9)

# Custom legend for bar colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', label='Peak Hour'),
    Patch(facecolor='#3498db', label='Off-Peak Hour'),
]
ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][2:], fontsize=9)

plt.tight_layout()
out = OUTPUT_DIR / "plot1_hourly_distribution.png"
plt.savefig(out, bbox_inches='tight')
plt.show()
print(f"  ✓ Saved: {out}")


# ─────────────────────────────────────────────
# PLOT 2: Top 10 Corridors by Event Count
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

top_corridors = (
    df['corridor']
    .value_counts()
    .drop('Non-corridor', errors='ignore')  # exclude the catch-all bucket
    .head(10)
)

# Color by corridor risk score
corridor_colors = []
for c in top_corridors.index:
    score = df[df['corridor'] == c]['corridor_risk_score'].iloc[0] if len(df[df['corridor'] == c]) > 0 else 0
    corridor_colors.append(score)

# Normalize colors to colormap
import matplotlib.cm as cm
norm = plt.Normalize(min(corridor_colors), max(corridor_colors))
mapped_colors = [cm.RdYlGn_r(norm(v)) for v in corridor_colors]

bars = ax.barh(top_corridors.index[::-1], top_corridors.values[::-1],
               color=mapped_colors[::-1], edgecolor='white', linewidth=0.5)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 5, bar.get_y() + bar.get_height()/2,
            str(int(w)), va='center', fontsize=9, fontweight='bold', color='#2c3e50')

ax.set_xlabel('Number of Events', fontsize=12)
ax.set_title('Top 10 Corridors by Event Count\n(Color = Risk Score: red=high, green=low)',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, top_corridors.max() * 1.12)

sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Corridor Risk Score', shrink=0.7)

plt.tight_layout()
out = OUTPUT_DIR / "plot2_top_corridors.png"
plt.savefig(out, bbox_inches='tight')
plt.show()
print(f"  ✓ Saved: {out}")


# ─────────────────────────────────────────────
# PLOT 3: Event Cause Distribution — Pie Chart
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

cause_counts = df['event_cause_grouped'].value_counts()

# ── Left: Pie chart
PALETTE = [
    '#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6',
    '#1abc9c','#e67e22','#34495e','#e91e63','#00bcd4'
]
explode = [0.05] * len(cause_counts)

wedges, texts, autotexts = axes[0].pie(
    cause_counts.values,
    labels=cause_counts.index,
    autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
    startangle=140,
    colors=PALETTE[:len(cause_counts)],
    explode=explode,
    pctdistance=0.82,
    labeldistance=1.12,
    textprops={'fontsize': 9}
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_fontweight('bold')

axes[0].set_title('Event Cause Distribution (Grouped)', fontsize=13, fontweight='bold')

# ── Right: Horizontal bar for readability
cause_pct = (cause_counts / cause_counts.sum() * 100).round(1)
colors_bar = PALETTE[:len(cause_counts)]
axes[1].barh(cause_counts.index[::-1], cause_counts.values[::-1],
             color=colors_bar[::-1], edgecolor='white')
for i, (val, pct) in enumerate(zip(cause_counts.values[::-1], cause_pct.values[::-1])):
    axes[1].text(val + 30, i, f'{val:,}  ({pct}%)', va='center', fontsize=9)
axes[1].set_xlabel('Count', fontsize=11)
axes[1].set_title('Event Cause — Count & Percentage', fontsize=13, fontweight='bold')
axes[1].set_xlim(0, cause_counts.max() * 1.25)

plt.suptitle('Traffic Event Cause Analysis', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
out = OUTPUT_DIR / "plot3_event_cause.png"
plt.savefig(out, bbox_inches='tight')
plt.show()
print(f"  ✓ Saved: {out}")


# ─────────────────────────────────────────────
# PLOT 4: Heatmap — Hour of Day vs Day of Week
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

day_order  = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
hour_order = list(range(24))

# ── Left: Total event count heatmap
pivot_count = (
    df.groupby(['day_name', 'hour_of_day'])
      .size()
      .unstack(fill_value=0)
      .reindex(index=day_order, columns=hour_order, fill_value=0)
)

sns.heatmap(
    pivot_count,
    ax=axes[0],
    cmap='YlOrRd',
    annot=True,
    fmt='d',
    annot_kws={'size': 6.5},
    linewidths=0.3,
    linecolor='#ecf0f1',
    cbar_kws={'label': 'Event Count', 'shrink': 0.8}
)
axes[0].set_title('Event Count: Hour × Day of Week (IST)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Hour of Day (IST)', fontsize=11)
axes[0].set_ylabel('Day of Week', fontsize=11)
axes[0].set_xticklabels([f"{h:02d}" for h in hour_order], rotation=0, fontsize=7)
axes[0].set_yticklabels(day_order, rotation=0)

# ── Right: High-severity events only (congestion_severity == 'high')
pivot_high = (
    df[df['congestion_severity'] == 'high']
      .groupby(['day_name', 'hour_of_day'])
      .size()
      .unstack(fill_value=0)
      .reindex(index=day_order, columns=hour_order, fill_value=0)
)

sns.heatmap(
    pivot_high,
    ax=axes[1],
    cmap='PuRd',
    annot=True,
    fmt='d',
    annot_kws={'size': 6.5},
    linewidths=0.3,
    linecolor='#ecf0f1',
    cbar_kws={'label': 'High Severity Count', 'shrink': 0.8}
)
axes[1].set_title('HIGH Severity Events: Hour × Day of Week (IST)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Hour of Day (IST)', fontsize=11)
axes[1].set_ylabel('')
axes[1].set_xticklabels([f"{h:02d}" for h in hour_order], rotation=0, fontsize=7)
axes[1].set_yticklabels(day_order, rotation=0)

plt.suptitle('Traffic Event Density — Temporal Heatmaps', fontsize=15, fontweight='bold')
plt.tight_layout()
out = OUTPUT_DIR / "plot4_heatmap_hour_vs_day.png"
plt.savefig(out, bbox_inches='tight')
plt.show()
print(f"  ✓ Saved: {out}")


# ─────────────────────────────────────────────
# BONUS PLOT 5: Resolution Time Distribution by Event Cause
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

# Filter to reasonable resolution times (non-null, non-outlier)
res_df = df[df['resolution_minutes_capped'].notna()].copy()
res_df = res_df[res_df['event_cause_grouped'] != 'others']

# Compute median per cause for ordering
order = (res_df.groupby('event_cause_grouped')['resolution_minutes_capped']
               .median()
               .sort_values(ascending=False)
               .index.tolist())

sns.boxplot(
    data=res_df,
    x='event_cause_grouped',
    y='resolution_minutes_capped',
    order=order,
    palette='coolwarm',
    width=0.6,
    linewidth=1.2,
    fliersize=2,
    ax=ax
)

ax.set_xlabel('Event Cause (Grouped)', fontsize=12)
ax.set_ylabel('Resolution Time (minutes, capped at P99)', fontsize=12)
ax.set_title('Resolution Time Distribution by Event Cause', fontsize=14, fontweight='bold')
ax.set_xticklabels(order, rotation=30, ha='right')

# Add median labels
medians = res_df.groupby('event_cause_grouped')['resolution_minutes_capped'].median()
for i, cause in enumerate(order):
    med = medians[cause]
    ax.text(i, med + 5, f'{med:.0f}m', ha='center', va='bottom',
            fontsize=8, color='#2c3e50', fontweight='bold')

plt.tight_layout()
out = OUTPUT_DIR / "plot5_resolution_by_cause.png"
plt.savefig(out, bbox_inches='tight')
plt.show()
print(f"  ✓ Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — EXPORT ENGINEERED DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════
# Drop raw intermediate columns before saving
cols_to_drop = ['start_datetime', 'closed_datetime', 'resolved_datetime',
                'start_datetime_utc', 'closed_datetime_utc', 'resolved_datetime_utc',
                'day_name', 'month_name', 'date']
df_export = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

out_csv = OUTPUT_DIR / "traffic_events_engineered.csv"
df_export.to_csv(out_csv, index=False)

print("\n" + "═"*60)
print("  DONE")
print("═"*60)
print(f"Final engineered CSV : {out_csv}")
print(f"Plots saved to       : {OUTPUT_DIR}")
print(f"Final df shape       : {df_export.shape}")
print(f"Engineered features  : {engineered_features}")
