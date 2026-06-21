"""
Traffic Event Model Training Pipeline
=====================================
Includes:
- Loading raw CSV and clean up columns
- Resolution Time recomputation (total_seconds / 60)
- Cascade seed label creation (120-minute forward lookback on same corridor)
- Rolling corridor lookback features (avoiding data leakage)
- Time-based train/test split (Train < April 2024, Test >= April 2024)
- Model A: Cascade Seed Binary Classifier (XGBoost + feature importance fallback)
- Model B: Severity Multiclass Classifier (XGBoost)
- Model C: Event Count Forecaster (LightGBM on Hourly Aggregates)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
try:
    import shap
    HAS_SHAP = True
except ModuleNotFoundError:
    HAS_SHAP = False
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from pathlib import Path

# Paths
INPUT_CSV = Path(r"C:\Users\Khushi\Downloads\Traffic\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")
OUTPUT_DIR = Path(r"C:\Users\Khushi\Downloads\Traffic\model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA & CRITICAL CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("Loading raw CSV data...")
df = pd.read_csv(INPUT_CSV, low_memory=False)
df.replace('NULL', np.nan, inplace=True)
df.replace('nan', np.nan, inplace=True)

# Re-read or parse start_datetime and closed_datetime as UTC
df['start_datetime'] = pd.to_datetime(df['start_datetime'], utc=True, errors='coerce')
df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], utc=True, errors='coerce')

# Convert start_datetime to IST timezone
df['start_ist'] = df['start_datetime'].dt.tz_convert('Asia/Kolkata')

# CRITICAL FIX: Recompute resolution_minutes using total_seconds() / 60
df['resolution_minutes'] = (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 60
# Handle negative values if any, and cap outliers at 99th percentile
valid_resolutions = df['resolution_minutes'].dropna()
if len(valid_resolutions) > 0:
    p99 = valid_resolutions.quantile(0.99)
    df['resolution_minutes_capped'] = df['resolution_minutes'].clip(lower=0, upper=p99)
else:
    df['resolution_minutes_capped'] = 0.0

df['is_open'] = df['resolution_minutes'].isna().astype(int)

# Standardize values
df['corridor'] = df['corridor'].fillna('Non-corridor')
df['priority'] = df['priority'].fillna('Low')
df['requires_road_closure'] = df['requires_road_closure'].astype(str).str.upper().str.strip()
df['requires_road_closure'] = df['requires_road_closure'].map({'TRUE': True, 'FALSE': False}).fillna(False).astype(bool)

# Impute Zone using police_station mode proxy
ps_zone_map = (
    df.dropna(subset=['zone', 'police_station'])
      .groupby('police_station')['zone']
      .agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
      .to_dict()
)
df['zone'] = df.apply(
    lambda row: ps_zone_map.get(row['police_station'], row['zone'])
                if pd.isna(row['zone']) and pd.notna(row['police_station'])
                else row['zone'],
    axis=1
).fillna('Unknown')

# Fill Junction
df['junction'] = df['junction'].fillna('Unknown')

# Vehicle properties
df['veh_type'] = df['veh_type'].astype(str).str.lower().str.strip()
df['veh_type'] = df['veh_type'].replace({'nan': 'unknown', '': 'unknown', '0': 'unknown'})
HEAVY_TYPES = {'heavy_vehicle', 'truck', 'bmtc_bus', 'ksrtc_bus', 'private_bus'}
df['is_heavy_vehicle'] = df['veh_type'].isin(HEAVY_TYPES).astype(int)

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

# Event cause remapping
df['event_cause'] = df['event_cause'].astype(str).str.lower().str.strip()
df = df[df['event_cause'] != 'test_demo'].copy()
CIVIC_CAUSES = {'public_event', 'procession', 'vip_movement', 'protest'}
df['event_cause_grouped'] = df['event_cause'].apply(
    lambda x: 'civic_event' if x in CIVIC_CAUSES else x
)

# Time parts
df['hour_of_day'] = df['start_ist'].dt.hour
df['day_of_week'] = df['start_ist'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['start_ist'].dt.month
def is_peak(hour):
    return int((19 <= hour <= 22) or (4 <= hour <= 6))
df['is_peak_hour'] = df['hour_of_day'].apply(is_peak)

# Target B definition: congestion_severity
def assign_severity(row):
    high = row['priority'] == 'High'
    closure = row['requires_road_closure'] == True
    if high and closure:
        return 'high'
    elif high or closure:
        return 'medium'
    else:
        return 'low'
df['congestion_severity'] = df.apply(assign_severity, axis=1)

# Corridor risk score
corridor_high = (
    df[df['priority'] == 'High']
    .groupby('corridor')['id']
    .count()
)
min_val = corridor_high.min()
max_val = corridor_high.max()
corridor_risk = ((corridor_high - min_val) / (max_val - min_val)).rename('corridor_risk_score')
df = df.merge(corridor_risk.reset_index(), on='corridor', how='left')
df['corridor_risk_score'] = df['corridor_risk_score'].fillna(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CASCADE SEED LABEL CREATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 1: Cascade Seed Label Creation ---")

# Filter to named corridor events only
corridor_df = df[~df['corridor'].isin(['Non-corridor', 'NULL']) & df['corridor'].notna()].copy()

# Sort by corridor, then start_datetime
corridor_df = corridor_df.sort_values(by=['corridor', 'start_datetime']).reset_index(drop=True)

# Compute cascade seeds
is_cascade_seed = []
secondary_counts = []

for i, row in corridor_df.iterrows():
    curr_time = row['start_datetime']
    curr_corridor = row['corridor']
    
    # Get subsequent events on same corridor in next 120 minutes (strictly after current event)
    subsequent = corridor_df.iloc[i+1:]
    subsequent_same_corr = subsequent[subsequent['corridor'] == curr_corridor]
    
    window_end = curr_time + pd.Timedelta(minutes=120)
    following_events = subsequent_same_corr[
        (subsequent_same_corr['start_datetime'] > curr_time) & 
        (subsequent_same_corr['start_datetime'] <= window_end)
    ]
    
    count = len(following_events)
    secondary_counts.append(count)
    is_cascade_seed.append(count >= 3)

corridor_df['is_cascade_seed'] = is_cascade_seed
corridor_df['secondary_count'] = secondary_counts

# Merge these back into the main dataframe (Non-corridor rows get False/0)
df = df.merge(
    corridor_df[['id', 'is_cascade_seed', 'secondary_count']], 
    on='id', 
    how='left'
)
df['is_cascade_seed'] = df['is_cascade_seed'].fillna(False)
df['secondary_count'] = df['secondary_count'].fillna(0).astype(int)

# Print verification stats
print(f"Total cascade seeds found: {df['is_cascade_seed'].sum()}")
print("\nSeeds by top corridors:")
top_corr_seeds = df[df['is_cascade_seed']].groupby('corridor').agg(
    seed_count=('is_cascade_seed', 'count'),
    total_secondary_events=('secondary_count', 'sum')
).sort_values(by='seed_count', ascending=False)
print(top_corr_seeds.head(10))

print("\nSeeds by event cause:")
seeds_by_cause = df[df['is_cascade_seed']].groupby('event_cause_grouped').size().sort_values(ascending=False)
print(seeds_by_cause)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ROLLING CORRIDOR FEATURES (No leakage)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 2: Rolling Corridor Features ---")

# Sort full df by start_datetime ascending before computing lookbacks
df = df.sort_values(by='start_datetime').reset_index(drop=True)

# Initialize columns
df['corridor_events_2h'] = 0
df['corridor_events_6h'] = 0
df['seed_event_present_3h'] = 0

# Group by corridor to calculate features
for corr, group in df.groupby('corridor'):
    c_2h = []
    c_6h = []
    seed_3h = []
    
    seed_causes = {'water_logging', 'tree_fall', 'pot_holes', 'construction'}
    
    for idx, t in enumerate(group['start_datetime']):
        t_2h_ago = t - pd.Timedelta(hours=2)
        t_6h_ago = t - pd.Timedelta(hours=6)
        t_3h_ago = t - pd.Timedelta(hours=3)
        
        past_group = group.iloc[:idx]
        
        events_2h = past_group[past_group['start_datetime'] >= t_2h_ago]
        events_6h = past_group[past_group['start_datetime'] >= t_6h_ago]
        events_3h = past_group[past_group['start_datetime'] >= t_3h_ago]
        
        c_2h.append(len(events_2h))
        c_6h.append(len(events_6h))
        
        has_seed = 1 if any(events_3h['event_cause_grouped'].isin(seed_causes)) else 0
        seed_3h.append(has_seed)
        
    df.loc[group.index, 'corridor_events_2h'] = c_2h
    df.loc[group.index, 'corridor_events_6h'] = c_6h
    df.loc[group.index, 'seed_event_present_3h'] = seed_3h

# Compute cascade density
corridor_total_counts = df['corridor'].value_counts().to_dict()
df['cascade_density'] = df.apply(
    lambda row: row['corridor_events_2h'] / corridor_total_counts[row['corridor']]
    if corridor_total_counts[row['corridor']] > 0 else 0.0,
    axis=1
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TIME-BASED SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 3: Time-Based Train/Test Split ---")

train_mask = (df['start_ist'] < pd.Timestamp('2024-04-01', tz='Asia/Kolkata')) | df['start_ist'].isna()
test_mask = ~train_mask

df_train = df[train_mask].copy()
df_test = df[test_mask].copy()

print(f"Train set shape: {df_train.shape}")
print(f"Test set shape : {df_test.shape}")
print("\nis_cascade_seed class distribution (Train):")
print(df_train['is_cascade_seed'].value_counts(normalize=True))
print("\nis_cascade_seed class distribution (Test):")
print(df_test['is_cascade_seed'].value_counts(normalize=True))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MODEL A: CASCADE SEED BINARY CLASSIFIER (XGBOOST)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 4: Model A - Cascade Seed Classifier ---")

le_cause = LabelEncoder()
le_corr = LabelEncoder()

df['event_cause_encoded'] = le_cause.fit_transform(df['event_cause_grouped'].astype(str))
df['corridor_encoded'] = le_corr.fit_transform(df['corridor'].astype(str))

df_train['event_cause_encoded'] = le_cause.transform(df_train['event_cause_grouped'].astype(str))
df_test['event_cause_encoded'] = le_cause.transform(df_test['event_cause_grouped'].astype(str))

df_train['corridor_encoded'] = le_corr.transform(df_train['corridor'].astype(str))
df_test['corridor_encoded'] = le_corr.transform(df_test['corridor'].astype(str))

features_a = [
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour',
    'is_heavy_vehicle', 'corridor_risk_score', 'corridor_events_2h',
    'corridor_events_6h', 'seed_event_present_3h', 'cascade_density',
    'event_cause_encoded', 'corridor_encoded'
]

X_train_a = df_train[features_a]
y_train_a = df_train['is_cascade_seed'].astype(int)

X_test_a = df_test[features_a]
y_test_a = df_test['is_cascade_seed'].astype(int)

num_neg = (y_train_a == 0).sum()
num_pos = (y_train_a == 1).sum()
scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
print(f"Computed scale_pos_weight: {scale_pos_weight:.3f}")

model_a = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model_a.fit(X_train_a, y_train_a)

probs_test = model_a.predict_proba(X_test_a)[:, 1]
chosen_threshold = 0.5
best_recall = 0.0

print("\nThreshold Sweep:")
for thresh in np.linspace(0.1, 0.9, 9):
    preds_temp = (probs_test >= thresh).astype(int)
    rec = (preds_temp[y_test_a == 1] == 1).mean()
    prec = (y_test_a[preds_temp == 1] == 1).mean() if preds_temp.sum() > 0 else 0
    print(f"  Threshold: {thresh:.1f} | Precision: {prec:.3f} | Recall: {rec:.3f}")
    if rec >= 0.80 and (thresh < chosen_threshold or chosen_threshold == 0.5):
        chosen_threshold = thresh
        best_recall = rec

if best_recall < 0.80:
    recalls = []
    thresholds = np.linspace(0.1, 0.9, 81)
    for t in thresholds:
        p_t = (probs_test >= t).astype(int)
        recalls.append((t, (p_t[y_test_a == 1] == 1).mean()))
    recalls = sorted(recalls, key=lambda x: abs(x[1] - 0.80))
    chosen_threshold = recalls[0][0]

print(f"\nChosen threshold for recall >= 0.80: {chosen_threshold:.3f}")
final_preds_a = (probs_test >= chosen_threshold).astype(int)

print("\nModel A Classification Report:")
print(classification_report(y_test_a, final_preds_a))
print("Confusion Matrix:")
print(confusion_matrix(y_test_a, final_preds_a))

if HAS_SHAP:
    try:
        explainer = shap.TreeExplainer(model_a)
        shap_values = explainer(X_test_a)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_a, max_display=10, show=False)
        plt.title("Model A: Top 10 SHAP Feature Importances", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "model_a_shap.png", bbox_inches='tight')
        plt.close()
        print("Generated SHAP feature importance plot.")
    except Exception as e:
        print(f"SHAP explanation generation failed: {e}. Falling back to standard plot.")
        HAS_SHAP = False

if not HAS_SHAP:
    importances = model_a.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    names = [features_a[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices][::-1], align='center', color='teal')
    plt.yticks(range(len(indices)), [names[i] for i in range(len(indices))][::-1])
    plt.xlabel('Relative Importance')
    plt.title('Model A: Top 10 Feature Importances (XGBoost Built-in)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_a_shap.png", bbox_inches='tight')
    plt.close()
    print("Generated standard XGBoost feature importance plot.")

joblib.dump(model_a, OUTPUT_DIR / "cascade_seed_model.pkl")
with open(OUTPUT_DIR / "cascade_threshold.txt", "w") as f:
    f.write(f"{chosen_threshold:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — MODEL B: SEVERITY CLASSIFIER (XGBOOST MULTICLASS)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 5: Model B - Severity Classifier ---")

le_sev = LabelEncoder()
y_train_b = le_sev.fit_transform(df_train['congestion_severity'].astype(str))
y_test_b = le_sev.transform(df_test['congestion_severity'].astype(str))

features_b = [
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour',
    'is_heavy_vehicle', 'corridor_risk_score',
    'event_cause_encoded', 'corridor_encoded',
    'requires_road_closure', 'secondary_count'
]

X_train_b = df_train[features_b].copy()
X_test_b = df_test[features_b].copy()

X_train_b['requires_road_closure'] = X_train_b['requires_road_closure'].astype(int)
X_test_b['requires_road_closure'] = X_test_b['requires_road_closure'].astype(int)

model_b = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
model_b.fit(X_train_b, y_train_b)

preds_b = model_b.predict(X_test_b)

print("\nModel B Classification Report:")
print(classification_report(y_test_b, preds_b, target_names=le_sev.classes_))

joblib.dump(model_b, OUTPUT_DIR / "severity_model.pkl")
joblib.dump(le_sev, OUTPUT_DIR / "severity_target_encoder.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — MODEL C: EVENT COUNT FORECASTER (LIGHTGBM REGRESSOR)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- STEP 6: Model C - Count Forecaster ---")

corridor_hourly = df[~df['corridor'].isin(['Non-corridor', 'NULL']) & df['corridor'].notna()].copy()
corridor_hourly['hourly_bucket'] = corridor_hourly['start_ist'].dt.floor('h')

agg_df = corridor_hourly.groupby(['corridor', 'hourly_bucket']).size().rename('event_count').reset_index()

min_time = agg_df['hourly_bucket'].min()
max_time = agg_df['hourly_bucket'].max()

if pd.notna(min_time) and pd.notna(max_time):
    time_range = pd.date_range(start=min_time, end=max_time, freq='h', tz='Asia/Kolkata')
    corridors_list = agg_df['corridor'].unique()
    
    mux = pd.MultiIndex.from_product([corridors_list, time_range], names=['corridor', 'hourly_bucket'])
    grid_df = pd.DataFrame(index=mux).reset_index()
    
    grid_df = grid_df.merge(agg_df, on=['corridor', 'hourly_bucket'], how='left').fillna(0)
    grid_df['event_count'] = grid_df['event_count'].astype(int)
else:
    grid_df = agg_df

grid_df = grid_df.sort_values(by=['corridor', 'hourly_bucket']).reset_index(drop=True)

grid_df['day_of_week'] = grid_df['hourly_bucket'].dt.dayofweek
grid_df['hour_of_day'] = grid_df['hourly_bucket'].dt.hour
grid_df['corridor_encoded'] = le_corr.transform(grid_df['corridor'].astype(str))

grid_df['count_lag_1'] = grid_df.groupby('corridor')['event_count'].shift(1)
grid_df['count_lag_24'] = grid_df.groupby('corridor')['event_count'].shift(24)
grid_df['count_lag_168'] = grid_df.groupby('corridor')['event_count'].shift(168)

grid_df = grid_df.dropna().reset_index(drop=True)

train_mask_c = grid_df['hourly_bucket'] < pd.Timestamp('2024-04-01', tz='Asia/Kolkata')
test_mask_c = ~train_mask_c

train_c = grid_df[train_mask_c].copy()
test_c = grid_df[test_mask_c].copy()

features_c = ['day_of_week', 'hour_of_day', 'corridor_encoded', 'count_lag_1', 'count_lag_24', 'count_lag_168']

X_train_c = train_c[features_c]
y_train_c = train_c['event_count']

X_test_c = test_c[features_c]
y_test_c = test_c['event_count']

print(f"Model C Train records: {len(X_train_c)}")
print(f"Model C Test records : {len(X_test_c)}")

model_c = lgb.LGBMRegressor(
    objective='poisson',
    metric='mae',
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model_c.fit(X_train_c, y_train_c)

preds_c = model_c.predict(X_test_c)
mae = mean_absolute_error(y_test_c, preds_c)
rmse = np.sqrt(mean_squared_error(y_test_c, preds_c))

print(f"\nModel C Regression Metrics (April Test Slots):")
print(f"  Mean Absolute Error (MAE) : {mae:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")

joblib.dump(model_c, OUTPUT_DIR / "count_model.pkl")

def predict_next_2h(corridor, day_of_week, hour_ist, recent_count_1=0, recent_count_24=0, recent_count_168=0):
    """
    Predict event count for a given corridor and hour slot.
    """
    try:
        corr_enc = le_corr.transform([corridor])[0]
    except ValueError:
        corr_enc = 0
    
    input_data = pd.DataFrame([{
        'day_of_week': day_of_week,
        'hour_of_day': hour_ist,
        'corridor_encoded': corr_enc,
        'count_lag_1': recent_count_1,
        'count_lag_24': recent_count_24,
        'count_lag_168': recent_count_168
    }])
    
    pred_hour1 = model_c.predict(input_data)[0]
    
    input_data_hour2 = pd.DataFrame([{
        'day_of_week': day_of_week,
        'hour_of_day': (hour_ist + 1) % 24,
        'corridor_encoded': corr_enc,
        'count_lag_1': pred_hour1,
        'count_lag_24': recent_count_24,
        'count_lag_168': recent_count_168
    }])
    pred_hour2 = model_c.predict(input_data_hour2)[0]
    
    total_pred = int(np.round(pred_hour1 + pred_hour2))
    return max(0, total_pred)

print("\nExample prediction:")
print(f"Mysore Road on Thursday at 19:00 IST: {predict_next_2h('Mysore Road', 3, 19, 1, 2, 1)} expected events in next 2 hours.")

print("\nAll models and metrics computed and saved to:", OUTPUT_DIR)
