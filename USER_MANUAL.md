
# 🚦 ASTRAM Cascade Intelligence System
## Complete User Manual — Operations Edition

> **Built for:** Bengaluru Traffic Operations Officers, Duty Commanders, and Hackathon Evaluators  
> **Dashboard:** Streamlit web app · 3 pages · Real-time cascade prediction  
> **Data:** 8,173 verified ASTRAM platform events · Nov 2023 – Apr 2024 · 14 named corridors

---

## ⚡ Quick Start (30 seconds)

```bash
# 1. Navigate to the project folder
cd C:\Users\Khushi\Downloads\Traffic

# 2. Launch the dashboard
streamlit run app.py

# 3. Open your browser at:
http://localhost:8501
```

> The app loads all three ML models automatically on startup. First load takes ~5–10 seconds.  
> After that, all predictions run in under 50ms.

---

## 🗺️ App Structure at a Glance

```
ASTRAM OPS (Sidebar)
│
├── 📊  Page 1 — Command Overview
│         System-wide metrics, hourly breakdown chart, live map, top corridors
│
├── 🚨  Page 2 — Cascade Alert System
│         Real-time per-incident risk scoring, resource deployment card
│
└── 🎬  Page 3 — March 7, 2024 Replay (The Proof)
          Step-by-step replay of a real cascade event on Mysore Road
```

Navigate between pages using the **radio buttons in the left sidebar** under "Navigation."

---

---

# 📊 PAGE 1 — Command Overview

> **When to use:** Start here. This page gives you the system-wide picture — which corridors are hot, when events cluster, and where resources are being consumed.

---

## 1.1 · System Risk Headline Banner

At the very top, a red banner highlights the **largest single cascade** ever identified in the dataset:

| Field | Value |
|---|---|
| **Corridor** | Bannerghatta Road |
| **Date** | Dec 16, 2023 |
| **Trigger** | pot_holes |
| **Secondary events** | 30 in 2 hours |

This is your reference point — the worst-case the system is designed to prevent.

---

## 1.2 · System Baseline Metrics (6 cards)

| Metric Card | What It Means |
|---|---|
| **Cascade seeds found** | 176 — total historical events that triggered a chain reaction |
| **Secondary events** | 1,499 — downstream incidents directly caused by seeds |
| **Worst single cascade** | 30 — the maximum downstream impact from one seed event |
| **Avg secondaries / seed** | 8.5 — on average, each seed spawns 8–9 follow-on incidents |
| **Water-logging resolution** | 4,358 min (~3 days) — the longest-running event type |
| **Affected corridors** | 14 — named ASTRAM corridors actively modelled |

---

## 1.3 · Sidebar Filters

Located in the **left sidebar** under the navigation buttons. All charts and metrics on this page update live as you adjust filters.

| Filter | How to Use |
|---|---|
| **Date Range** | Click either date to open a calendar picker. Drag to set a range. Default: full dataset. |
| **Corridors** | Multi-select dropdown. Leave blank to show all corridors. Type to search. |
| **Incident Causes** | Multi-select dropdown. E.g., select "water_logging" to see only waterlogging events. |

> 💡 **Tip:** To isolate a single corridor and cause combination (e.g., pot_holes on Hosur Road), select both filters simultaneously.

---

## 1.4 · Filtered Period Metrics (4 cards)

These update based on your sidebar filters.

| Card | Description |
|---|---|
| **Events in period** | Total events matching your date + corridor + cause selection |
| **Cascade alerts identified** | How many of those events were cascade seeds |
| **Corridors at risk** | Number of distinct corridors with cascade activity in the filtered period |
| **Avg resolution time** | Mean time to resolve events (negative values and data errors excluded) |

---

## 1.5 · "When Bengaluru Breaks Down" — Hourly Chart

A bar chart showing how many events occurred at each hour of the day (IST).

| Bar Color | Meaning |
|---|---|
| 🔴 **Red** | Model-defined operational shift windows: **04:00–06:00** and **19:00–22:00 IST** |
| 🔵 **Blue** | All other hours |

> ⚠️ **Interpret carefully:** The tallest bars are at **02:00 IST** (839 events) and **11:00 IST** (671 events). These represent when events are *reported*, not necessarily when they occur — overnight monitoring staff log many infrastructure events (potholes, road conditions) during the 01:00–03:00 shift.

---

## 1.6 · Live Operations Map

An interactive Plotly WebGL Map showing individual event locations as colored markers.

| Marker Color | Severity Level |
|---|---|
| 🔴 Red | HIGH — Priority event with road closure |
| 🟡 Amber | MEDIUM — High priority OR road closure |
| 🟢 Green | LOW — Neither high priority nor closure |

**Marker size** scales with the **corridor risk score** (larger = higher-risk corridor).

**Click any marker** to see a popup with: Corridor name · Incident cause · Risk score · Severity level.

---

## 1.7 · Top Cascade Corridors Table

Expanding panel at the bottom. Shows each corridor's total cascade seeds and cumulative secondary events triggered.

**Key numbers to know:**

| Corridor | Seeds | Secondary Events | Notes |
|---|---|---|---|
| Hosur Road | 52 | 558 | Most seeds — highest frequency |
| Bannerghatta Road | 39 | 527 | Worst per-seed ratio (13.5 events/seed) |
| Mysore Road | 26 | 152 | Demo corridor — March 7 replay |
| ORR East 1 | 16 | — | Fourth highest |

---

---

# 🚨 PAGE 2 — Cascade Alert System

> **When to use:** When an officer has just received an incoming incident report and wants to know: *"Is this going to cascade? What do I deploy?"*

---

## 2.1 · How to Run a Risk Assessment

All inputs are in the **left sidebar** under "Incident Parameter Matrix."

**Step 1 — Select the corridor** where the incident was reported.  
*(Only named ASTRAM corridors are available. Non-corridor events use separate fallback logic.)*

**Step 2 — Select the day of week** (the current day, or the day of the event).

**Step 3 — Set the hour (IST)** using the slider.  
*(0 = midnight, 12 = noon, 23 = 11 PM)*

**Step 4 — Select the reported incident cause** from the dropdown.

> ✅ The prediction runs automatically — no "Submit" button needed. Results update instantly whenever any input changes.

---

## 2.2 · Reading the Risk Banner

The large colored banner at the top of the main area shows the tier result.

| Banner | Risk Score | What It Means | Action |
|---|---|---|---|
| 🔴 **RED — HIGH CASCADE RISK** | ≥ 60% | High probability of chain reaction | **Deploy resources immediately** |
| 🟠 **AMBER — MODERATE RISK** | 30–59% | Elevated risk, not yet critical | **Pre-position resources** — put units on standby |
| 🟢 **GREEN — LOW RISK** | < 30% | Unlikely to cascade | **Monitor only** — log and watch |

> 💡 **Risk score** is the raw model output from Model A (XGBoost cascade seed classifier), expressed as a probability percentage.

---

## 2.3 · Prediction Latency Caption

Directly below the banner, a small caption shows:

```
⚡ Prediction computed in Xms
```

This is the live inference time for the 3-model pipeline. Typical values: **10–50ms**. This confirms the system is capable of real-time deployment — no batch processing delays.

---

## 2.4 · Semicircular Risk Gauge (Left Column)

A custom, high-contrast Plotly Gauge visualizing the cascade risk probability percentage. It is segmented into distinct Green (0-30%), Amber (30-60%), and Red (60-100%) bands with a dark slate pointer indicator. It is accompanied by a dynamic, plain-English **Risk Assessment Summary card** describing the incident profile and severity recommendations.

---

## 2.5 · Primary Risk Drivers (Right Column)

A horizontal bar chart visualizing the top 3 contributing factors to the cascade probability, derived using true mathematical **SHAP Feature Contributions** computed directly from the XGBoost booster model. 

* **X-Axis**: Relative Contribution Weight
* **Y-Axis**: Feature labels (e.g. Corridor local vulnerability, recent event rate, traffic peak flags)

---

## 2.6 · Response Resource Action Dashboard

This dashboard displays the pre-emptive resource recommendations in a clean, visual grid:

* **👮 Traffic Police Card**: Officers recommended for direct dispatch.
* **🚒 Breakdown Recovery Card**: Heavy towing units required to clear blockages.
* **🚧 Barricades Card**: Blockage gates to redirect traffic flow.

It is accompanied by a detailed **Operational Dispatch Details** panel detailing:
* **Recommended Action**: Clear text instructions.
* **Dispatch Station**: Duty station location.
* **Estimated Clearance**: Projected duration formatted with relative strings (e.g. `clears in ~48 min`).
* **Expected Secondary Count**: Expected follow-on events in the next 2 hours.

---

## 2.7 · Live Operations Control Console

When **User Mode** is toggled to **Ops Dispatcher**, an interactive controls console is rendered:

* **🚨 Confirm Immediate Dispatch**: Triggers an interactive green toast notification and updates the session history log with a `(DISPATCHED)` status.
* **⏳ Order Pre-position Standby**: Triggers a standby warning banner and logs a `(STANDBY)` status.

---

## 2.8 · Resource Allocation Rules

The system deploys resources in tiers based on the risk score:

| Risk Score | Police | Breakdown Units | Barricades | Priority Flag |
|---|---|---|---|---|
| ≥ 70% | 4 | 2 | 3 | PRIORITY DEPLOYMENT |
| 40–69% | 3 | 1 | 2 | — |
| < 40% | 2 | 0 | 1 | — |

**Cause-specific add-ons applied automatically:**

| Cause | Extra Resources |
|---|---|
| water_logging | +1 breakdown unit |
| pot_holes (between 00:00–06:00) | +2 barricades |
| Road closure required | +2 barricades |

---

---

# 🎬 PAGE 3 — March 7, 2024 Replay (The Proof)

> **When to use:** For demonstrations, judge presentations, or understanding how the system behaves during a real-world cascade. This is your "show-don't-tell" page.

---

## 3.1 · The Scenario

On **March 7, 2024**, a `water_logging` event was reported on **Mysore Road at 10:51 IST**.

- Within **76 minutes**, **8 additional events** followed on the same corridor
- **6 of those 8** required road closure
- Without the system, each event would have been dispatched reactively — 30+ minutes after it was reported
- With ASTRAM Cascade Intelligence, a deployment alert would have fired **at 10:51 IST** — the moment the first event was logged

---

## 3.2 · Using the Replay Slider

The **"Replay up to hour (IST)"** slider steps through the day hour by hour.

| Slider Position | What Happens |
|---|---|
| **0–10** | Shows events reported before the water_logging seed. No cascade alert yet. |
| **11+** | 🚨 CASCADE SEED DETECTED line appears on the timeline. Alert banner activates. Deployment plan loads. |

> 💡 **Demo tip:** Set the slider to 10, then slowly drag it to 11 to show the moment the cascade alert fires. This is the most impactful moment in the demo.

---

## 3.3 · Chronological Incident Timeline

A bar chart showing events plotted against the time of day they were reported.

- Each bar represents **one incident**
- Bars are **colour-coded by event cause**
- A **red dashed vertical line** appears at hour 10.85 (approx. 10:51 IST) when the slider passes 11 — this marks the cascade seed detection point

---

## 3.4 · Reported Incident Stream Table

Below the chart, a scrollable table shows every event in chronological order with:

| Column | Meaning |
|---|---|
| **start_ist** | Time the event was reported (HH:MM:SS) |
| **event_cause_grouped** | Standardised event cause |
| **priority** | High / Medium / Low |
| **requires_road_closure** | TRUE / FALSE |

---

## 3.5 · Operational Deployment Plan

Appears when the slider is at 11 or beyond.

This is the **full output of the resource optimizer** for March 7 — the actual plan the system would have sent to the duty officer.

**Key stats from the actual March 7 plan:**

| Metric | Value |
|---|---|
| Total alerts fired | 11 of 16 events |
| Fully allocated | 3 alerts |
| Resource shortfall | 8 alerts → backup requested from neighboring zones |
| Highest risk score | 0.99 (water_logging at 10:51 IST) |
| Lowest risk score | 0.10 (vehicle_breakdown at 05:37 IST) |

> **Why so many shortfalls?** The resource pool (50 police, 8 breakdown units, 20 barricades) is a single-corridor daily budget for demonstration. In production, this connects to the city-wide fleet management system across all zones.

---

## 3.6 · Resource Deployment Capacity Bars

Three progress bars show cumulative resource consumption as the slider advances:

```
Officers deployed:       ████████░░░░░░  X/50
Breakdown units:         ████████████░░  X/8
Barricades:              ████████████░░  X/20
```

Watch these bars fill as you advance the slider — this visually demonstrates how quickly breakdown units are exhausted by the 3rd high-risk alert.

---

## 3.7 · Download Deployment Plan

A **"Download deployment plan (CSV)"** button at the bottom exports the full 11-alert plan as a CSV file.

Use this to:
- Show judges a concrete, exportable operational output
- Demonstrate the system produces machine-readable results (not just charts)
- Share with evaluators for offline review

---

---

# 🧠 Understanding the Models

| Model | Type | Task | Algorithm |
|---|---|---|---|
| **Model A** | Binary classifier | Will this event trigger a cascade? | XGBoost |
| **Model B** | Multiclass classifier | What severity level? (high/medium/low) | XGBoost |
| **Model C** | Count regressor | How many secondary events in the next 2 hours? | LightGBM (Poisson) |

**Training data:** January – March 2024 (named corridor events only)  
**Test data:** April 2024 (strict chronological split — no future leakage)  
**Cascade seed label:** An event is a seed if 3 or more additional events appear on the same corridor within 120 minutes

---

# ⚠️ Known Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| Non-corridor events excluded | 38% of events (3,145) not modelled | Rule-based fallback planned for Phase 2 |
| Threshold = 0.10 (high recall) | ~80% of alerts are false positives | Use RED/AMBER/GREEN tiers — only auto-deploy on RED |
| Debris clearance from 3 data points | Shows "(estimate — limited data)" | Do not use for operational planning |
| Resource caps are demo values | Shortfalls appear rapidly in replay | In production, connect to real fleet management |
| Static corridor risk score | Computed over full dataset — slightly future-aware | Treat as infrastructure property, not prediction |

---

# 🔑 Glossary

| Term | Definition |
|---|---|
| **Cascade seed** | An event that is followed by 3 or more events on the same corridor within 120 minutes |
| **Secondary event** | An event triggered (or aggravated) by a cascade seed |
| **Corridor risk score** | 0–1 score based on historical high-priority event density on that corridor (1 = highest risk) |
| **Cascade density** | Ratio of events in the past 2h on this corridor to the corridor's total historical event count |
| **seed_event_present_3h** | Whether a seed-type cause (water_logging, pot_holes, construction, tree_fall) appeared on this corridor in the past 3 hours |
| **ETA** | Estimated time of arrival for deployed units — subject to fleet positioning |
| **RESOURCE SHORTFALL** | The requested units exceed what's available in the current resource pool |
| **IST** | Indian Standard Time (UTC+5:30) — all times in the dashboard are IST |

---

*ASTRAM Cascade Intelligence System · Built on 8,173 verified ASTRAM platform events · Demo corridor: Mysore Road, March 7, 2024*
