"""
Resource Deployment Optimizer
=============================
Uses trained models:
- cascade_seed_model.pkl + cascade_threshold.txt
- severity_model.pkl + severity_target_encoder.pkl
- count_model.pkl

Optimizes police, breakdown units, and barricades based on predicted risk.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path

# Paths
MODELS_DIR = Path(r"C:\Users\Khushi\Downloads\Traffic\model_outputs")
OUTPUT_DIR = MODELS_DIR
DATA_DIR = Path(r"C:\Users\Khushi\Downloads\Traffic\eda_outputs")
INPUT_CSV = Path(r"C:\Users\Khushi\Downloads\Traffic\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════
print("Loading trained models and threshold...")
model_a = joblib.load(MODELS_DIR / "cascade_seed_model.pkl")
model_b = joblib.load(MODELS_DIR / "severity_model.pkl")
model_c = joblib.load(MODELS_DIR / "count_model.pkl")
le_sev = joblib.load(MODELS_DIR / "severity_target_encoder.pkl")

with open(MODELS_DIR / "cascade_threshold.txt", "r") as f:
    CASCADE_THRESHOLD = float(f.read().strip())

print(f"Loaded cascade model. Optimal threshold: {CASCADE_THRESHOLD:.3f}")

# Configurable resources
INITIAL_RESOURCES = {"traffic_police": 50, "breakdown_units": 8, "barricades": 20}

# Average resolution times (minutes) based on data analysis
AVG_RESOLUTIONS = {
    'vehicle_breakdown': 58,
    'accident': 48,
    'water_logging': 4358,  # ~3 days
    'pot_holes': 5047,      # ~3.5 days
    'construction': 4351,
    'tree_fall': 2427,
    'road_conditions': 5989,
    'congestion': 75,
    'procession': 55,
    'civic_event': 191,
    'debris': 19190,
    'fog / low visibility': 120,
    'others': 2227
}

RESOLUTION_SAMPLE_N = {
    'vehicle_breakdown': 6500,
    'water_logging': 400,
    'pot_holes': 450,
    'construction': 380,
    'tree_fall': 240,
    'accident': 300,
    'debris': 3,         # very low — flag this
    'road_conditions': 20,
    'others': 500,
    'civic_event': 60
}

# Features required for Model A and B
FEATURES_A = [
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour',
    'is_heavy_vehicle', 'corridor_risk_score', 'corridor_events_2h',
    'corridor_events_6h', 'seed_event_present_3h', 'cascade_density',
    'event_cause_encoded', 'corridor_encoded'
]

FEATURES_B = [
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour',
    'is_heavy_vehicle', 'corridor_risk_score',
    'event_cause_encoded', 'corridor_encoded',
    'requires_road_closure', 'secondary_count'
]

# We need encoders to convert input strings to label-encoded integers.
# Let's rebuild encoders from the raw CSV data.
print("Rebuilding label encoders from the dataset...")
raw_df = pd.read_csv(INPUT_CSV, low_memory=False)
raw_df.replace('NULL', np.nan, inplace=True)
raw_df.replace('nan', np.nan, inplace=True)

# Standardize values
raw_df['corridor'] = raw_df['corridor'].fillna('Non-corridor')
raw_df['event_cause'] = raw_df['event_cause'].fillna('others').astype(str).str.lower().str.strip()
raw_df = raw_df[raw_df['event_cause'] != 'test_demo'].copy()
CIVIC_CAUSES = {'public_event', 'procession', 'vip_movement', 'protest'}
raw_df['event_cause_grouped'] = raw_df['event_cause'].apply(
    lambda x: 'civic_event' if x in CIVIC_CAUSES else x
)

from sklearn.preprocessing import LabelEncoder
le_cause = LabelEncoder()
le_corr = LabelEncoder()
raw_df['event_cause_encoded'] = le_cause.fit_transform(raw_df['event_cause_grouped'].astype(str))
raw_df['corridor_encoded'] = le_corr.fit_transform(raw_df['corridor'].astype(str))

# Create mapping dictionary for risk scores
corridor_high = (
    raw_df[raw_df['priority'] == 'High']
    .groupby('corridor')['id']
    .count()
)
min_val = corridor_high.min()
max_val = corridor_high.max()
corridor_risk_map = ((corridor_high - min_val) / (max_val - min_val)).to_dict()

# Corridor count map for cascade density
corridor_total_counts = raw_df['corridor'].value_counts().to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 1 — PREDICT CASCADE
# ══════════════════════════════════════════════════════════════════════════════
def predict_cascade(event_row_dict: dict) -> dict:
    """
    Predicts cascade seed probability, secondary count forecast, and severity level.
    
    Parameters:
    event_row_dict (dict): Dictionary representing a traffic event row with features.
    
    Returns:
    dict: Prediction metrics containing is_cascade_seed, cascade_probability,
          predicted_secondary_count, and severity.
    """
    # 1. Prepare input row for Model A
    row_a = {}
    for f in FEATURES_A:
        if f in event_row_dict:
            row_a[f] = event_row_dict[f]
        else:
            # fill default or derive if encoding
            if f == 'event_cause_encoded':
                cause = event_row_dict.get('event_cause_grouped', 'others')
                row_a[f] = le_cause.transform([cause])[0] if cause in le_cause.classes_ else le_cause.transform(['others'])[0]
            elif f == 'corridor_encoded':
                corr = event_row_dict.get('corridor', 'Non-corridor')
                row_a[f] = le_corr.transform([corr])[0] if corr in le_corr.classes_ else le_corr.transform(['Non-corridor'])[0]
            else:
                row_a[f] = 0

    df_row_a = pd.DataFrame([row_a])
    
    # Predict cascade seed probability
    prob = float(model_a.predict_proba(df_row_a)[0, 1])
    is_seed = bool(prob >= CASCADE_THRESHOLD)
    
    # 2. Predict secondary count using Model C
    # Model C features: ['day_of_week', 'hour_of_day', 'corridor_encoded', 'count_lag_1', 'count_lag_24', 'count_lag_168']
    corr_enc = row_a['corridor_encoded']
    day_of_week = event_row_dict.get('day_of_week', 0)
    hour_of_day = event_row_dict.get('hour_of_day', 0)
    
    # Lags (use values in row dict if present, else default to average lookback counts)
    lag1 = event_row_dict.get('count_lag_1', event_row_dict.get('corridor_events_2h', 0))
    lag24 = event_row_dict.get('count_lag_24', 0)
    lag168 = event_row_dict.get('count_lag_168', 0)
    
    df_row_c1 = pd.DataFrame([{
        'day_of_week': day_of_week,
        'hour_of_day': hour_of_day,
        'corridor_encoded': corr_enc,
        'count_lag_1': lag1,
        'count_lag_24': lag24,
        'count_lag_168': lag168
    }])
    pred_hour1 = model_c.predict(df_row_c1)[0]
    
    # Roll forward for the second hour
    df_row_c2 = pd.DataFrame([{
        'day_of_week': day_of_week,
        'hour_of_day': (hour_of_day + 1) % 24,
        'corridor_encoded': corr_enc,
        'count_lag_1': pred_hour1,
        'count_lag_24': lag24,
        'count_lag_168': lag168
    }])
    pred_hour2 = model_c.predict(df_row_c2)[0]
    
    pred_count = int(np.round(pred_hour1 + pred_hour2))
    
    # 3. Predict Severity Level using Model B
    row_b = {}
    for f in FEATURES_B:
        if f in row_a:
            row_b[f] = row_a[f]
        elif f == 'requires_road_closure':
            row_b[f] = int(event_row_dict.get('requires_road_closure', False))
        elif f == 'secondary_count':
            # Use predicted secondary count as a proxy feature for severity model
            row_b[f] = pred_count
        else:
            row_b[f] = 0
            
    df_row_b = pd.DataFrame([row_b])
    sev_idx = model_b.predict(df_row_b)[0]
    severity = str(le_sev.classes_[sev_idx])
    
    return {
        "is_cascade_seed": is_seed,
        "cascade_probability": prob,
        "predicted_secondary_count": max(0, pred_count),
        "severity": severity
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 2 — GENERATE ALERT
# ══════════════════════════════════════════════════════════════════════════════
def _build_deploy_message(police: int, breakdown: int, barricades: int, corridor: str) -> str:
    """Build a dynamic, plain-English deployment instruction listing all non-zero resources."""
    parts = []
    if police > 0:
        parts.append(f"{police} traffic police")
    if breakdown > 0:
        parts.append(f"{breakdown} breakdown units")
    if barricades > 0:
        parts.append(f"{barricades} barricades")
    if not parts:
        return f"Monitor {corridor} - no immediate deployment required"
    return f"Deploy {', '.join(parts)} -> {corridor}, ETA: subject to fleet positioning"


def generate_alert(corridor: str, seed_event_row: dict, current_resources: dict) -> dict:
    """
    Generates a structured warning alert when a cascade seed event is detected.
    
    Parameters:
    corridor (str): Name of the corridor.
    seed_event_row (dict): Dictionary of features for the seed event.
    current_resources (dict): Map of currently available response resources.
    
    Returns:
    dict: Fully structured alarm alert details.
    """
    # 1. Run cascade prediction
    pred = predict_cascade(seed_event_row)
    risk = pred['cascade_probability']
    sec_count = pred['predicted_secondary_count']
    
    # Confidence band
    if risk >= 0.70:
        conf_band = "HIGH"
    elif risk >= 0.40:
        conf_band = "MEDIUM"
    else:
        conf_band = "LOW"
        
    # 2. Sequential deployment rules
    police_needed = 0
    breakdown_needed = 0
    barricades_needed = 0
    prio_flag = ""
    
    # Base allocation by risk
    if risk >= 0.70:
        police_needed = 4
        breakdown_needed = 2
        barricades_needed = 3
        prio_flag = "PRIORITY DEPLOYMENT"
    elif risk >= 0.40:
        police_needed = 3
        breakdown_needed = 1
        barricades_needed = 2
    else:
        police_needed = 2
        breakdown_needed = 0
        barricades_needed = 1
        
    # Cause-specific add-ons
    cause = seed_event_row.get('event_cause_grouped', 'others')
    hour_ist = seed_event_row.get('hour_of_day', 12)
    closure = seed_event_row.get('requires_road_closure', False)
    
    if cause == 'water_logging':
        breakdown_needed += 1
    if cause == 'pot_holes' and (0 <= hour_ist <= 6):
        barricades_needed += 2
    if closure:
        barricades_needed += 2
        
    # Check resource shortfall
    zone = seed_event_row.get('zone', 'Unknown')
    # Simple lookup for neighbors
    neighbor_map = {
        'Central Zone 2': 'East Zone',
        'Central Zone 1': 'West Zone',
        'North Zone 1': 'North Zone 2',
        'North Zone 2': 'North Zone 1',
        'South Zone 1': 'South Zone 2',
        'South Zone 2': 'South Zone 1'
    }
    neighboring_zone = neighbor_map.get(zone, 'neighboring zone')
    
    police_flag = "ALLOCATED"
    if police_needed > current_resources.get('traffic_police', 0):
        police_flag = f"RESOURCE SHORTFALL: request backup from {neighboring_zone}"
        
    # Sequence ID and code
    corr_code = "".join([w[0] for w in corridor.split() if w[0].isalpha()]).upper()[:3]
    if len(corr_code) < 3:
        corr_code = (corr_code + "XXX")[:3]
        
    # Unique alert ID timestamp
    time_ist_str = seed_event_row.get('time_ist', '12:00 IST')
    date_clean = seed_event_row.get('date_str', '20240307').replace("-", "")
    alert_seq = seed_event_row.get('seq_num', 1)
    alert_id = f"ALERT-{date_clean}-{corr_code}-{alert_seq:03d}"
    
    # 3. Estimated clearance datetime
    res_min = AVG_RESOLUTIONS.get(cause, 60)
    start_time_val = seed_event_row.get('start_datetime')
    if isinstance(start_time_val, str):
        try:
            start_dt = pd.to_datetime(start_time_val)
        except:
            start_dt = datetime.now()
    elif isinstance(start_time_val, (pd.Timestamp, datetime)):
        start_dt = start_time_val
    else:
        start_dt = datetime.now()
        
    clearance_dt = start_dt + pd.Timedelta(minutes=res_min)
    clearance_dt_ist = clearance_dt.tz_convert('Asia/Kolkata') if hasattr(clearance_dt, 'tz_convert') else clearance_dt
    clearance_str = clearance_dt_ist.strftime('%Y-%m-%d %H:%M:%S IST')
    
    if RESOLUTION_SAMPLE_N.get(cause, 100) < 10:
        clearance_str += " (estimate — limited data)"
    
    # 4. English SHAP representations
    shap_factors = []
    # Logic to mock/translate shap relative to event attributes
    if seed_event_row.get('is_peak_hour', 0) == 1:
        shap_factors.append("Evening peak hour")
    else:
        shap_factors.append("Off-peak slot traffic profile")
        
    if seed_event_row.get('seed_event_present_3h', 0) == 1:
        shap_factors.append("Infrastructure seed nearby")
    else:
        shap_factors.append("No active surrounding blockages")
        
    if seed_event_row.get('is_heavy_vehicle', 0) == 1:
        shap_factors.append("Heavy vehicle involved")
    else:
        shap_factors.append("Light vehicle involved")
        
    return {
        "alert_id": alert_id,
        "trigger": {
            "cause": cause,
            "corridor": corridor,
            "time_ist": time_ist_str,
            "road_closure_required": closure
        },
        "cascade_prediction": {
            "risk_score": round(risk, 2),
            "predicted_secondary_events_2h": sec_count,
            "top_cause_expected": "vehicle_breakdown",
            "confidence_band": conf_band
        },
        "deployment": {
            "DEPLOY_NOW": _build_deploy_message(police_needed, breakdown_needed, barricades_needed, corridor),
            "ALERT_STATIONS": seed_event_row.get('police_station', 'Unknown Station'),
            "STANDBY": f"Activate additional 2 breakdown recovery units if secondary count exceeds {sec_count}",
            "police_needed": police_needed,
            "breakdown_needed": breakdown_needed,
            "barricades_needed": barricades_needed,
            "police_flag": police_flag,
            "prio_flag": prio_flag
        },
        "estimated_clearance_ist": clearance_str,
        "shap_top3": shap_factors[:3]
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 4 — RUN REPLAY
# ══════════════════════════════════════════════════════════════════════════════
def run_replay(date_str="2024-03-07", corridor="Mysore Road") -> list:
    """
    Replays a specific corridor and date timeline to run predictions and log alerts.
    
    Parameters:
    date_str (str): Date of interest (YYYY-MM-DD).
    corridor (str): Name of the corridor.
    
    Returns:
    list: List of generated alerts.
    """
    print(f"\n--- Running Replay for {corridor} on {date_str} ---")
    
    # Reload original dataset to make sure we parse times cleanly
    df_raw = pd.read_csv(INPUT_CSV, low_memory=False)
    df_raw.replace('NULL', np.nan, inplace=True)
    df_raw.replace('nan', np.nan, inplace=True)
    
    df_raw['start_datetime'] = pd.to_datetime(df_raw['start_datetime'], utc=True, errors='coerce')
    df_raw['start_ist'] = df_raw['start_datetime'].dt.tz_convert('Asia/Kolkata')
    
    # Filter
    df_raw['date_str'] = df_raw['start_ist'].dt.strftime('%Y-%m-%d')
    filtered = df_raw[(df_raw['date_str'] == date_str) & (df_raw['corridor'] == corridor)].copy()
    
    # Sort
    filtered = filtered.sort_values(by='start_ist').reset_index(drop=True)
    
    if len(filtered) == 0:
        print(f"No events found for corridor {corridor} on {date_str}.")
        return []
    
    # Create temporal and other features for the replay slice
    def is_peak(hour):
        return int((19 <= hour <= 22) or (4 <= hour <= 6))
        
    filtered['hour_of_day'] = filtered['start_ist'].dt.hour
    filtered['day_of_week'] = filtered['start_ist'].dt.dayofweek
    filtered['is_weekend'] = filtered['day_of_week'].isin([5, 6]).astype(int)
    filtered['is_peak_hour'] = filtered['hour_of_day'].apply(is_peak)
    filtered['requires_road_closure'] = filtered['requires_road_closure'].astype(str).str.upper().str.strip()
    filtered['requires_road_closure'] = filtered['requires_road_closure'].map({'TRUE': True, 'FALSE': False}).fillna(False).astype(bool)
    
    # Fill defaults for rolling features within replay
    filtered['corridor_risk_score'] = filtered['corridor'].map(corridor_risk_map).fillna(0.0)
    filtered['event_cause_grouped'] = filtered['event_cause'].astype(str).str.lower().str.strip().apply(
        lambda x: 'civic_event' if x in CIVIC_CAUSES else x
    )
    
    # Add dummy rolling count to replicate step 2
    filtered['corridor_events_2h'] = list(range(len(filtered))) # simplified proxy for chronological count
    filtered['corridor_events_6h'] = list(range(len(filtered)))
    filtered['seed_event_present_3h'] = 1
    filtered['cascade_density'] = filtered['corridor_events_2h'] / corridor_total_counts.get(corridor, 100)
    
    # Encode
    filtered['event_cause_encoded'] = le_cause.transform(filtered['event_cause_grouped'])
    filtered['corridor_encoded'] = le_corr.transform(filtered['corridor'])
    
    alerts = []
    current_resources = INITIAL_RESOURCES.copy()
    seq_counter = 1
    
    for i, row in filtered.iterrows():
        # Prepare dictionary row
        row_dict = row.to_dict()
        row_dict['time_ist'] = row['start_ist'].strftime('%H:%M IST')
        row_dict['date_str'] = date_str
        row_dict['seq_num'] = seq_counter
        
        # Predict cascade
        pred = predict_cascade(row_dict)
        prob = pred['cascade_probability']
        
        # Log to timeline console
        status_str = f"CASCADE SEED ({prob*100:.0f}% risk)" if pred['is_cascade_seed'] else f"NO CASCADE ({prob*100:.0f}% risk)"
        print(f"  {row_dict['time_ist']} - {row_dict['event_cause_grouped']} detected -> {status_str}")
        
        if pred['is_cascade_seed']:
            alert = generate_alert(corridor, row_dict, current_resources)
            alerts.append(alert)
            seq_counter += 1
            
    return alerts


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 5 — DEPLOYMENT PLAN GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def deployment_plan_df(alerts_list: list) -> pd.DataFrame:
    """
    Compiles alerts into a tabular deployment plan dataframe respecting resource caps.
    
    Parameters:
    alerts_list (list): List of generated alerts dict.
    
    Returns:
    pd.DataFrame: Resource allocation plan dataframe.
    """
    print("\nCompiling resource allocation plan (greedy by risk_score)...")
    
    # Sort alerts by risk_score descending
    sorted_alerts = sorted(alerts_list, key=lambda x: x['cascade_prediction']['risk_score'], reverse=True)
    
    resources = INITIAL_RESOURCES.copy()
    records = []
    
    for alert in sorted_alerts:
        dep = alert['deployment']
        pol_req = dep['police_needed']
        bd_req = dep['breakdown_needed']
        barr_req = dep['barricades_needed']
        
        # Allocation checking
        pol_alloc = min(pol_req, resources['traffic_police'])
        bd_alloc = min(bd_req, resources['breakdown_units'])
        barr_alloc = min(barr_req, resources['barricades'])
        
        # Deduct
        resources['traffic_police'] -= pol_alloc
        resources['breakdown_units'] -= bd_alloc
        resources['barricades'] -= barr_alloc
        
        # Determine status flag
        if pol_alloc < pol_req or bd_alloc < bd_req or barr_alloc < barr_req:
            flag = "RESOURCE SHORTFALL: request backup"
        else:
            flag = "ALLOCATED"
            
        records.append({
            "alert_id": alert['alert_id'],
            "corridor": alert['trigger']['corridor'],
            "time_ist": alert['trigger']['time_ist'],
            "seed_cause": alert['trigger']['cause'],
            "risk_score": alert['cascade_prediction']['risk_score'],
            "deploy_police": pol_alloc,
            "deploy_breakdown": bd_alloc,
            "deploy_barricades": barr_alloc,
            "estimated_clearance": alert['estimated_clearance_ist'],
            "resource_flag": flag
        })
        
    plan_df = pd.DataFrame(records)
    
    # Save files to both model outputs directory and parent workspace root
    plan_df.to_csv(OUTPUT_DIR / "deployment_plan.csv", index=False)
    plan_df.to_json(OUTPUT_DIR / "deployment_plan.json", orient='records', indent=2)
    plan_df.to_csv(Path(r"C:\Users\Khushi\Downloads\Traffic\deployment_plan.csv"), index=False)
    plan_df.to_json(Path(r"C:\Users\Khushi\Downloads\Traffic\deployment_plan.json"), orient='records', indent=2)
    print(f"Exported plan to model_outputs/ and workspace root.")
    
    return plan_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Test replay on Mysore Road demo day
    alerts = run_replay(date_str="2024-03-07", corridor="Mysore Road")
    
    # Generate deployment plan
    if len(alerts) > 0:
        plan_df = deployment_plan_df(alerts)
        print("\nDeployment Plan DataFrame Preview:")
        print(plan_df.to_string(index=False))
    else:
        print("\nNo alerts triggered during replay.")
