import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# Set page configuration
st.set_page_config(layout="wide", page_title="ASTRAM Cascade Intelligence", page_icon="🚦")

# Custom CSS for dark sidebar, hide default streamlit elements, custom metrics, and custom alerts
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stSidebar"] {
    font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #0f1117 !important;
}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3, 
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
}
/* Style sidebar button to be premium and readable */
section[data-testid="stSidebar"] button {
    background-color: #3b82f6 !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 6px !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] button:hover {
    background-color: #2563eb !important;
    border-color: #2563eb !important;
}
section[data-testid="stSidebar"] button p {
    color: #ffffff !important;
    font-weight: 600 !important;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="metric-container"] {
    background-color: white !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}
.cascade-alert {
    background-color: #991b1b !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 1rem 1.5rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin: 1rem 0 !important;
}
.warning-alert {
    background-color: #78350f !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 1rem 1.5rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin: 1rem 0 !important;
}
.safe-banner {
    background-color: #065f46 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 1rem 1.5rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin: 1rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Load Prediction System (cached resource)
@st.cache_resource
def load_prediction_system():
    import resource_optimizer
    return resource_optimizer

ro = load_prediction_system()

# Session state — alert history log
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []

# Load Dataset (cached data)
@st.cache_data
def load_data():
    df = pd.read_csv(Path(__file__).parent / "eda_outputs" / "traffic_events_engineered.csv")
    df['start_datetime_ist'] = pd.to_datetime(df['start_datetime_ist'])
    df = df.sort_values(by=['corridor', 'start_datetime_ist']).reset_index(drop=True)

    # Fast vectorized cascade seed computation (replaces O(n²) iterrows loop)
    # Self-join on corridor, then filter to events within 130-minute window ahead
    corridor_mask = df['corridor'].notna() & (df['corridor'] != 'Non-corridor')
    df_corr = df[corridor_mask][['corridor', 'start_datetime_ist']].copy()
    df_corr['_idx'] = df_corr.index

    merged = df_corr.merge(df_corr, on='corridor', suffixes=('_seed', '_follow'))
    window = pd.Timedelta(minutes=130)
    merged = merged[
        (merged['start_datetime_ist_follow'] > merged['start_datetime_ist_seed']) &
        (merged['start_datetime_ist_follow'] <= merged['start_datetime_ist_seed'] + window)
    ]
    sec_count_map = merged.groupby('_idx_seed').size()

    df['sec_count'] = df.index.map(sec_count_map).fillna(0).astype(int)
    df.loc[~corridor_mask, 'sec_count'] = 0
    df['is_cascade_seed'] = df['sec_count'] >= 4
    return df

df = load_data()

# Navigation + live mode + alert log
st.sidebar.title("ASTRAM OPS")
page = st.sidebar.radio("Navigation", ["Command Overview", "Cascade Alert System", "March 7, 2024 Replay (The Proof)"])

# Live mode auto-refresh
live_mode = st.sidebar.toggle("🔴 Live Mode (auto-refresh 30s)", value=False)
if live_mode:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30000, key="live_mode_refresh")

# Alert history log in sidebar
if st.session_state.alert_log:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚨 Alert History (this session)**")
    for entry in reversed(st.session_state.alert_log[-5:]):
        colour = "🔴" if entry['tier'] == 'RED' else "🟠"
        st.sidebar.caption(f"{colour} {entry['time'].strftime('%H:%M')} — {entry['corridor']} ({entry['risk']})")
    if st.sidebar.button("Clear log"):
        st.session_state.alert_log = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — COMMAND OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Command Overview":
    st.title("Bengaluru Traffic Command — Live Overview")
    
    # System Risk Headline Banner
    st.markdown("""
    <div style="background-color: #fef2f2; border-left: 5px solid #ef4444; padding: 1.25rem; border-radius: 8px; margin-bottom: 1.5rem; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);">
        <h4 style="margin: 0 0 0.5rem 0; color: #991b1b; font-size: 1.15rem; font-weight: 700;">🚨 SYSTEM RISK HEADLINE: Largest Cascade Identified</h4>
        <p style="margin: 0; color: #7f1d1d; font-size: 1rem; line-height: 1.5;">
            The largest single cascade in the dataset occurred on <b>Bannerghatta Road on Dec 16, 2023</b>: 
            <b>30 secondary events</b> were triggered within 2 hours from a single <b>pot_holes</b> seed event.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Baseline Metrics Section
    st.subheader("System Baseline Metrics")
    base_col1, base_col2, base_col3, base_col4, base_col5, base_col6 = st.columns(6)
    with base_col1:
        st.metric("Cascade seeds found", "176", help="Total historical cascade seed events identified across all corridors")
    with base_col2:
        st.metric("Secondary events", "1,499", help="Total downstream incidents triggered by cascade seeds")
    with base_col3:
        st.metric("Worst single cascade", "30", help="Maximum downstream incidents from a single seed (Bannerghatta Road, Dec 16)")
    with base_col4:
        st.metric("Avg secondaries / seed", "8.5", help="Average number of secondary events triggered per seed event")
    with base_col5:
        st.metric("Water-logging resolution", "4,358 min", help="Average resolution time for water-logging events (~3 days)")
    with base_col6:
        st.metric("Affected corridors", "14", help="Number of corridors with active cascade patterns")
        
    st.markdown("---")
    
    # Filters
    st.sidebar.subheader("Filters")
    min_date = df['start_datetime_ist'].min().date()
    max_date = df['start_datetime_ist'].max().date()
    
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    elif isinstance(date_range, tuple) and len(date_range) == 1:
        start_date = date_range[0]
        end_date = max_date
    else:
        start_date, end_date = min_date, max_date
        
    corridors_list = sorted([c for c in df['corridor'].dropna().unique() if c != 'Non-corridor'])
    selected_corridors = st.sidebar.multiselect("Corridors", options=corridors_list, default=[])
    
    causes_list = sorted(df['event_cause_grouped'].dropna().unique())
    selected_causes = st.sidebar.multiselect("Incident Causes", options=causes_list, default=[])
    
    # Filter DataFrame
    filtered_df = df.copy()
    filtered_df = filtered_df[
        (filtered_df['start_datetime_ist'].dt.date >= start_date) &
        (filtered_df['start_datetime_ist'].dt.date <= end_date)
    ]
    if selected_corridors:
        filtered_df = filtered_df[filtered_df['corridor'].isin(selected_corridors)]
    if selected_causes:
        filtered_df = filtered_df[filtered_df['event_cause_grouped'].isin(selected_causes)]
        
    # Filtered Period Metrics Row
    st.subheader("Filtered Period Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Events in period", len(filtered_df), help="Events matching current date/corridor/cause filters")
    with col2:
        st.metric("Cascade alerts identified", int(filtered_df['is_cascade_seed'].sum()), help="Cascade seeds detected in filtered period")
    with col3:
        st.metric("Corridors at risk", filtered_df[filtered_df['is_cascade_seed']]['corridor'].nunique(), help="Distinct corridors with cascade activity in filtered period")
    with col4:
        # Filter out negative and extreme outlier resolution times
        valid_res = filtered_df['resolution_minutes'].dropna()
        valid_res = valid_res[(valid_res > 0) & (valid_res <= 43200)]  # Cap at 30 days
        avg_res = valid_res.mean()
        avg_res_str = f"{avg_res:.0f} min" if not pd.isna(avg_res) else "N/A"
        st.metric("Avg resolution time", avg_res_str, help="Mean resolution time (excluding data errors and extreme outliers)")
        
    st.write("")
    
    # Columns for Chart and Map
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.subheader("Temporal Breakdown")
        # Hourly Breakdown Plot
        hourly_counts = filtered_df.groupby('hour_of_day').size().reset_index(name='count')
        hourly_counts = hourly_counts.set_index('hour_of_day').reindex(range(24), fill_value=0).reset_index()
        
        # Color peak hours (4-6 and 19-22 IST) red — matches is_peak_hour model definition
        colors = ['#ef4444' if (4 <= h <= 6) or (19 <= h <= 22) else '#3b82f6' for h in hourly_counts['hour_of_day']]
        
        fig = go.Figure(data=[go.Bar(
            x=hourly_counts['hour_of_day'],
            y=hourly_counts['count'],
            marker_color=colors
        )])
        fig.update_layout(
            title="When Bengaluru Breaks Down",
            xaxis_title="Hour (IST)",
            yaxis_title="Event count",
            template="plotly_white",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("🔴 Red bars = model-defined operational shift windows (04:00–06:00, 19:00–22:00 IST). Actual reporting peaks at 02:00 and 11:00 IST reflect overnight monitoring patterns.")
        
    with right_col:
        st.subheader("Live Operations Map")
        map_source = filtered_df.dropna(subset=['latitude', 'longitude']).copy()
        map_truncated = len(map_source) > 500
        map_data = map_source.head(500)
        
        if not map_data.empty:
            map_data['severity_display'] = map_data['congestion_severity'].str.upper()
            fig_map = px.scatter_mapbox(
                map_data,
                lat='latitude',
                lon='longitude',
                color='congestion_severity',
                color_discrete_map={'high': '#ef4444', 'medium': '#f59e0b', 'low': '#10b981'},
                size=map_data['corridor_risk_score'].fillna(0).clip(lower=0.1) * 15 + 5,
                hover_name='corridor',
                hover_data={'event_cause': True, 'severity_display': True, 'corridor_risk_score': ':.2f', 'latitude': False, 'longitude': False},
                zoom=10.5,
                center=dict(lat=12.9716, lon=77.5946),
                height=400
            )
            fig_map.update_layout(
                mapbox_style='open-street-map',
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
                legend=dict(
                    title="Severity",
                    yanchor="top",
                    y=0.98,
                    xanchor="left",
                    x=0.02,
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="#e2e8f0",
                    borderwidth=1
                )
            )
            st.plotly_chart(fig_map, width='stretch')
        else:
            st.info("No geocoded incidents to map for this period.")
            
        if map_truncated:
            st.caption(f"Showing 500 of {len(map_source)} events. Apply filters to narrow down.")
        
    # Top Corridors Expander
    with st.expander("Top cascade corridors", expanded=True):
        exp_col1, exp_col2 = st.columns([3, 2])
        
        with exp_col1:
            summary_df = df[df['is_cascade_seed']].groupby('corridor').agg(
                seeds_count=('is_cascade_seed', 'count'),
                secondary_events_count=('sec_count', 'sum')
            ).sort_values(by='seeds_count', ascending=False).reset_index()
            summary_df.columns = ["Corridor", "Cascade Seeds Count", "Secondary Events Triggered"]
            st.dataframe(summary_df, width='stretch', hide_index=True)
            
        with exp_col2:
            st.markdown("""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 1.25rem; border-radius: 8px; height: 100%;">
                <h5 style="margin-top: 0; color: #0f172a; font-weight: 700; font-size: 1.05rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; margin-bottom: 0.75rem;">💡 Operational Intelligence</h5>
                <ul style="margin: 0; padding-left: 1.15rem; color: #334155; font-size: 0.9rem; line-height: 1.6;">
                    <li><b>Hosur Road</b>: 52 seeds, 558 secondary events — <i>the most cascade-prone corridor</i>.</li>
                    <li><b>Bannerghatta Road</b>: 39 seeds, 527 secondary events — <i>worst per-seed severity (13.5 events/seed)</i>.</li>
                    <li><b>Mysore Road</b>: 26 seeds, 152 secondary events — <i>your March 7 demo corridor</i>.</li>
                    <li><b>pot_holes</b>: #1 cascade trigger (91 seeds → 1,024 secondary events).</li>
                    <li><b>water_logging</b>: #2 cascade trigger (61 seeds → 378 secondary events, avg resolution ~3 days).</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CASCADE ALERT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Cascade Alert System":
    st.title("Cascade Alert System")
    
    st.sidebar.subheader("Incident Parameter Matrix")
    named_corridors = sorted([c for c in df['corridor'].dropna().unique() if c != 'Non-corridor'])
    corridor_selected = st.sidebar.selectbox("Corridor Under Evaluation", options=named_corridors)
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_selected = st.sidebar.selectbox("Day of Week", options=days)
    day_of_week_num = days.index(day_selected)
    
    hour_ist = st.sidebar.slider("Hour of Day (IST)", min_value=0, max_value=23, value=12)
    
    causes = sorted(df['event_cause_grouped'].dropna().unique())
    event_cause = st.sidebar.selectbox("Reported Incident Cause", options=causes)
    
    # Run Risk Assessment automatically on input changes
    with st.spinner("Assessing risk..."):
        # Lookup typical zone and station for realistic prediction
        corr_data = df[df['corridor'] == corridor_selected]
        if not corr_data.empty:
            typical_zone = corr_data['zone'].dropna().mode().iloc[0] if not corr_data['zone'].dropna().empty else 'Unknown'
            typical_ps = corr_data['police_station'].dropna().mode().iloc[0] if not corr_data['police_station'].dropna().empty else 'Unknown Station'
            typical_risk = corr_data['corridor_risk_score'].mean()
        else:
            typical_zone = 'Unknown'
            typical_ps = 'Unknown Station'
            typical_risk = 0.0
            
        # Build input features — compute rolling corridor counts from real historical trends for selected slot
        corr_hist = df[df['corridor'] == corridor_selected]
        
        # Filter to events that occurred on the same day of week, in the preceding 2 and 6 hour windows
        hist_2h = corr_hist[
            (corr_hist['day_of_week'] == day_of_week_num) &
            (corr_hist['hour_of_day'] >= max(0, hour_ist - 2)) &
            (corr_hist['hour_of_day'] <= hour_ist)
        ]
        hist_6h = corr_hist[
            (corr_hist['day_of_week'] == day_of_week_num) &
            (corr_hist['hour_of_day'] >= max(0, hour_ist - 6)) &
            (corr_hist['hour_of_day'] <= hour_ist)
        ]
        
        # Calculate typical active events in these buckets
        if not hist_2h.empty:
            dyn_events_2h = int(np.ceil(hist_2h.groupby(hist_2h['start_ist'].dt.date).size().mean()))
        else:
            dyn_events_2h = 0
            
        if not hist_6h.empty:
            dyn_events_6h = int(np.ceil(hist_6h.groupby(hist_6h['start_ist'].dt.date).size().mean()))
        else:
            dyn_events_6h = 0

        event_row_dict = {
            'hour_of_day': hour_ist,
            'day_of_week': day_of_week_num,
            'is_weekend': int(day_of_week_num in [5, 6]),
            'is_peak_hour': int((19 <= hour_ist <= 22) or (4 <= hour_ist <= 6)),
            'is_heavy_vehicle': 1 if event_cause in ['vehicle_breakdown', 'construction', 'accident'] else 0,
            'corridor_risk_score': typical_risk,
            'corridor_events_2h': dyn_events_2h,
            'corridor_events_6h': dyn_events_6h,
            'seed_event_present_3h': 1 if event_cause in ['water_logging', 'pot_holes', 'tree_fall', 'construction'] else 0,
            'cascade_density': ro.corridor_total_counts.get(corridor_selected, 50) / 8173.0,
            'event_cause_grouped': event_cause,
            'corridor': corridor_selected,
            'requires_road_closure': int(event_cause in ['water_logging', 'tree_fall', 'accident']),
            'zone': typical_zone,
            'police_station': typical_ps,
            'time_ist': f"{hour_ist:02d}:00 IST",
            'date_str': datetime.now().strftime('%Y%m%d'),
            'seq_num': 1
        }
        
        # Predict
        import time
        t0 = time.time()
        pred = ro.predict_cascade(event_row_dict)
        latency_ms = round((time.time() - t0) * 1000, 1)
        alert_details = ro.generate_alert(corridor_selected, event_row_dict, ro.INITIAL_RESOURCES)
        
    risk = pred['cascade_probability']
    st.caption(f"⚡ Prediction computed in {latency_ms}ms")

    # Sidebar assessment indicator for instant visibility
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Assessment Status**")
    if risk >= 0.60:
        st.sidebar.markdown('<div style="background-color:#991b1b;color:white;padding:10px;border-radius:6px;font-weight:bold;text-align:center;font-size:0.9rem;">🔴 HIGH CASCADE RISK ({:.0f}%)</div>'.format(risk*100), unsafe_allow_html=True)
    elif risk >= 0.30:
        st.sidebar.markdown('<div style="background-color:#78350f;color:white;padding:10px;border-radius:6px;font-weight:bold;text-align:center;font-size:0.9rem;">🟠 MODERATE RISK ({:.0f}%)</div>'.format(risk*100), unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div style="background-color:#065f46;color:white;padding:10px;border-radius:6px;font-weight:bold;text-align:center;font-size:0.9rem;">🟢 LOW RISK ({:.0f}%)</div>'.format(risk*100), unsafe_allow_html=True)

    # ── Risk Banner — always first visible element ──────────────────────────
    risk = pred.get("cascade_probability", 0)
    if risk >= 0.60:
        st.markdown('<div class="cascade-alert">🔴 RED — HIGH CASCADE RISK ({:.0f}%) — Deploy immediately</div>'.format(risk*100), unsafe_allow_html=True)
        tier = "RED"
    elif risk >= 0.30:
        st.markdown('<div class="warning-alert">🟠 AMBER — MODERATE RISK ({:.0f}%) — Pre-position resources</div>'.format(risk*100), unsafe_allow_html=True)
        tier = "AMBER"
    else:
        st.markdown('<div class="safe-banner">🟢 GREEN — LOW RISK ({:.0f}%) — Monitor only</div>'.format(risk*100), unsafe_allow_html=True)
        tier = "GREEN"

    # Append RED/AMBER to session alert log
    if tier in ("RED", "AMBER"):
        st.session_state.alert_log.append({
            'time': datetime.now(),
            'corridor': corridor_selected,
            'risk': f"{risk*100:.0f}%",
            'tier': tier,
            'cause': event_cause,
        })

    # ── Main layout columns: Left for Risk assessment, Right for Explainability ──────────────────────────
    left_chart_col, right_chart_col = st.columns([1, 1])
    
    with left_chart_col:
        # Donut / Gauge Chart for Cascade Risk
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Cascade Risk Level", 'font': {'size': 18, 'color': '#1e293b', 'weight': 'bold'}},
            number={'suffix': "%", 'font': {'size': 36, 'color': '#1e293b', 'weight': 'bold'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': '#0f172a'}, # Dark slate pointer bar
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 30], 'color': '#10b981'}, # Green
                    {'range': [30, 60], 'color': '#f59e0b'}, # Amber
                    {'range': [60, 100], 'color': '#ef4444'} # Red
                ],
            }
        ))
        fig_gauge.update_layout(
            height=240,
            margin=dict(l=30, r=30, t=50, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, width='stretch')
        
        # Human readable summary
        st.markdown(f"""
        <div style="background-color:#f8fafc; border-left: 4px solid #3b82f6; padding: 12px; border-radius: 6px; margin-top: 5px;">
            <p style="margin:0; font-weight:600; color:#1e293b; font-size:0.9rem;">🔍 Risk Assessment Summary</p>
            <p style="margin:4px 0 0 0; font-size:0.85rem; color:#475569; line-height:1.4;">
                An incident of type <b>{event_cause}</b> on <b>{corridor_selected}</b> yields a 
                <b>{risk*100:.0f}%</b> risk of secondary cascade events within the next 2 hours. 
                Recommended Response Level: <span style="font-weight:700; color:{'#ef4444' if tier=='RED' else '#d97706' if tier=='AMBER' else '#059669'};">{tier}</span>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with right_chart_col:
        # Horizontal Bar Chart — using true SHAP feature contributions
        shap_labels = alert_details['shap_top3']
        shap_raw_values = alert_details.get('shap_values3', [0.33, 0.33, 0.33])
        sum_vals = sum(shap_raw_values)
        shap_values = [round(v / sum_vals, 2) for v in shap_raw_values] if sum_vals > 0 else [0.33, 0.33, 0.33]
        
        fig_shap = go.Figure(go.Bar(
            x=shap_values[::-1],
            y=shap_labels[::-1],
            orientation='h',
            marker_color=['#ef4444', '#f59e0b', '#3b82f6'][::-1]
        ))
        fig_shap.update_layout(
            title="Primary Risk Drivers (SHAP Explainability)",
            xaxis_title="Relative Contribution Weight",
            template="plotly_white",
            height=240,
            margin=dict(l=20, r=20, t=50, b=10)
        )
        st.plotly_chart(fig_shap, width='stretch')
        st.caption("Factors derived from dynamic model feature values.")
        
    st.write("")
    
    # ── Visual Resource Dispatch Grid ──────────────────────────
    st.subheader("Response Resource Action Dashboard")
    dep = alert_details['deployment']
    
    # Tier-specific overrides for AMBER and GREEN
    if tier == "RED":
        police_disp   = dep['police_needed']
        bd_disp       = dep['breakdown_needed']
        barr_disp     = dep['barricades_needed']
        action_disp   = dep['DEPLOY_NOW']
        status_disp   = dep['police_flag']
        status_color  = "#fee2e2" # Light Red
        text_color    = "#991b1b"
    elif tier == "AMBER":
        police_disp   = max(dep['police_needed'], 2)
        bd_disp       = max(dep['breakdown_needed'], 1)
        barr_disp     = max(dep['barricades_needed'], 1)
        action_disp   = f"Pre-position {police_disp} traffic police and {bd_disp} breakdown unit(s) near {corridor_selected}"
        status_disp   = "STANDBY — Deploy on confirmation"
        status_color  = "#fef3c7" # Light Orange
        text_color    = "#78350f"
    else:  # GREEN
        police_disp   = 1
        bd_disp       = 1 if event_cause in ['accident', 'vehicle_breakdown', 'tree_fall'] else 0
        barr_disp     = 1 if event_cause in ['accident', 'water_logging', 'tree_fall'] else 0
        action_disp   = f"Dispatch 1 patrol officer to {corridor_selected} — routine monitoring"
        status_disp   = "ROUTINE MONITORING"
        status_color  = "#dcfce7" # Light Green
        text_color    = "#065f46"
        
    # Grid of Cards
    rc_col1, rc_col2, rc_col3 = st.columns(3)
    
    with rc_col1:
        st.markdown(f"""
        <div style="background-color:#f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center;">
            <p style="margin: 0; font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase;">👮 Traffic Police</p>
            <p style="margin: 8px 0; font-size: 2.2rem; font-weight: 700; color: #0f172a;">{police_disp}</p>
            <p style="margin: 0; font-size: 0.8rem; color: #475569; font-weight: 500;">Officers Dispatched</p>
        </div>
        """, unsafe_allow_html=True)
        
    with rc_col2:
        st.markdown(f"""
        <div style="background-color:#f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center;">
            <p style="margin: 0; font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase;">🚒 Breakdown Recovery</p>
            <p style="margin: 8px 0; font-size: 2.2rem; font-weight: 700; color: #0f172a;">{bd_disp}</p>
            <p style="margin: 0; font-size: 0.8rem; color: #475569; font-weight: 500;">Towing Units Dispatched</p>
        </div>
        """, unsafe_allow_html=True)
        
    with rc_col3:
        st.markdown(f"""
        <div style="background-color:#f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center;">
            <p style="margin: 0; font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase;">🚧 Barricades</p>
            <p style="margin: 8px 0; font-size: 2.2rem; font-weight: 700; color: #0f172a;">{barr_disp}</p>
            <p style="margin: 0; font-size: 0.8rem; color: #475569; font-weight: 500;">Blockage Gates Dispatched</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Dispatch Details Card
    st.markdown(f"""
    <div style="background-color:#f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 12px;">
            <span style="font-weight: 600; color: #334155; font-size: 1rem;">📋 Operational Dispatch Details</span>
            <span style="background-color: {status_color}; color: {text_color}; font-size: 0.8rem; font-weight: 700; padding: 4px 8px; border-radius: 4px; text-transform: uppercase;">{status_disp}</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div>
                <p style="margin: 0; font-size: 0.85rem; color: #64748b;">🎯 Recommended Action</p>
                <p style="margin: 4px 0 0 0; font-size: 0.95rem; font-weight: 600; color: #0f172a;">{action_disp}</p>
            </div>
            <div>
                <p style="margin: 0; font-size: 0.85rem; color: #64748b;">📡 Dispatch Station</p>
                <p style="margin: 4px 0 0 0; font-size: 0.95rem; font-weight: 600; color: #0f172a;">{dep['ALERT_STATIONS']}</p>
            </div>
            <div>
                <p style="margin: 0; font-size: 0.85rem; color: #64748b;">⏳ Estimated Clearance Time</p>
                <p style="margin: 4px 0 0 0; font-size: 0.95rem; font-weight: 600; color: #0f172a;">{alert_details['estimated_clearance_ist']}</p>
            </div>
            <div>
                <p style="margin: 0; font-size: 0.85rem; color: #64748b;">📈 Expected Secondary Count</p>
                <p style="margin: 4px 0 0 0; font-size: 0.95rem; font-weight: 600; color: #0f172a;">{pred['predicted_secondary_count']} events (next 2 hours)</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MARCH 7, 2024 REPLAY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "March 7, 2024 Replay (The Proof)":
    st.title("Cascade Replay — Mysore Road, March 7 2024")
    st.markdown("""
    On March 7, 2024, a **water_logging** event appeared on Mysore Road at **10:51 IST**. 
    Within **76 minutes, 8 events followed** — 6 requiring road closure. Here is the chronological playback.
    """)
    
    # Load Mysore Road data for replay
    @st.cache_data
    def load_replay_data():
        _data_path = Path(__file__).parent / "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
        df_raw = pd.read_csv(_data_path)
        df_raw['start_datetime'] = pd.to_datetime(df_raw['start_datetime'], utc=True, errors='coerce')
        df_raw['start_ist'] = df_raw['start_datetime'].dt.tz_convert('Asia/Kolkata')
        df_raw['date_str'] = df_raw['start_ist'].dt.strftime('%Y-%m-%d')
        
        # Filter Mysore Road March 7
        filtered = df_raw[(df_raw['date_str'] == '2024-03-07') & (df_raw['corridor'] == 'Mysore Road')].copy()
        filtered = filtered.sort_values(by='start_ist').reset_index(drop=True)
        
        # Group causes
        CIVIC_CAUSES = {'public_event', 'procession', 'vip_movement', 'protest'}
        filtered['event_cause_grouped'] = filtered['event_cause'].astype(str).str.lower().str.strip().apply(
            lambda x: 'civic_event' if x in CIVIC_CAUSES else x
        )
        filtered['hour_of_day'] = filtered['start_ist'].dt.hour + filtered['start_ist'].dt.minute / 60.0
        return filtered
        
    replay_data = load_replay_data()
    
    # Slider with helpful hint
    st.info("⬅ Drag the slider right to replay the Mysore Road timeline hour by hour. The cascade seed fires at hour 11.")
    selected_hour = st.slider("Replay up to hour (IST)", min_value=0, max_value=24, value=0)
    
    # Filter
    replay_filtered = replay_data[replay_data['start_ist'].dt.hour <= selected_hour]
    
    # Timeline plot
    if not replay_filtered.empty:
        fig_timeline = px.bar(
            replay_filtered,
            x='hour_of_day',
            y=[1]*len(replay_filtered),
            color='event_cause_grouped',
            labels={'hour_of_day': 'Hour of Day (IST)', 'y': 'Incident Count'},
            template='plotly_white',
            title="Chronological Incident Timeline"
        )
        fig_timeline.update_layout(
            xaxis=dict(range=[0, 24], dtick=1),
            yaxis=dict(dtick=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        # Add seed indicator line
        if selected_hour >= 11:
            fig_timeline.add_vline(
                x=10.85, line_width=2, line_dash="dash", line_color="red",
                annotation_text="CASCADE SEED DETECTED", annotation_position="top right"
            )
        st.plotly_chart(fig_timeline, width='stretch')
    else:
        st.info("No incidents reported on Mysore Road prior to this hour.")
        
    # Events table
    st.subheader("Reported Incident Stream")
    if not replay_filtered.empty:
        df_disp = replay_filtered.copy()
        df_disp['start_ist'] = df_disp['start_ist'].dt.strftime('%H:%M:%S')
        df_disp = df_disp[['start_ist', 'event_cause_grouped', 'priority', 'requires_road_closure']]
        st.dataframe(df_disp, width='stretch', hide_index=True)
    else:
        st.write("Incident stream is empty.")
        
    # Replay alerts and plan compilation
    @st.cache_data
    def compile_replay_results():
        alerts = ro.run_replay("2024-03-07", "Mysore Road")
        plan_df = ro.deployment_plan_df(alerts)
        return alerts, plan_df
        
    alerts, plan_df = compile_replay_results()
    
    # If seed is activated (hour passes 11)
    if selected_hour >= 11:
        st.write("")
        st.markdown('<div class="cascade-alert">⚠ HIGH CASCADE RISK DETECTED — 99% — Pre-emptive Dispatch Active</div>', unsafe_allow_html=True)
        
        st.subheader("Operational Deployment Plan")
        # Filter deployment plan based on alert time
        plan_df['hour_ist'] = plan_df['time_ist'].apply(lambda x: int(x.split(':')[0]))
        plan_filtered = plan_df[plan_df['hour_ist'] <= selected_hour]
        
        st.dataframe(
            plan_filtered.drop(columns=['hour_ist']),
            width='stretch',
            hide_index=True
        )
        
        # Calculate resources used
        used_police = int(plan_filtered['deploy_police'].sum())
        used_bd = int(plan_filtered['deploy_breakdown'].sum())
        used_barr = int(plan_filtered['deploy_barricades'].sum())
        
        # Cap resource metrics
        used_police = min(used_police, 50)
        used_bd = min(used_bd, 8)
        used_barr = min(used_barr, 20)
    else:
        used_police = 0
        used_bd = 0
        used_barr = 0
        
    # Progress bars — always visible regardless of slider position
    st.subheader("Resource Deployment Capacity")
    if selected_hour < 11:
        st.caption("⏳ Awaiting cascade trigger — move slider past hour 11 to activate deployment")
        st.info("ℹ️ **Status: NOT YET DEPLOYED** (All resources are currently held in reserve: 50/50 Police, 8/8 Breakdown units, 20/20 Barricades available)")
    else:
        st.progress(max(used_police / 50.0, 0.001), text=f"Officers deployed: {used_police}/50")
        st.progress(max(used_bd / 8.0, 0.001),     text=f"Breakdown units deployed: {used_bd}/8")
        st.progress(max(used_barr / 20.0, 0.001),  text=f"Barricades deployed: {used_barr}/20")
    
    # Download deployment plan button — generate from in-memory plan_df
    st.write("")
    csv_bytes = plan_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download deployment plan (CSV)",
        data=csv_bytes,
        file_name="deployment_plan.csv",
        mime="text/csv"
    )
