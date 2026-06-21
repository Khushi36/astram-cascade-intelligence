# 🚦 ASTRAM Cascade Intelligence System
### Pre-emptive Traffic Deployment Engine for Bengaluru

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 The Problem

On **December 16, 2023**, a single pothole was reported on Bannerghatta Road at the start of the evening commute. Within two hours, **30 vehicles broke down** at that location — each one blocking a lane, each one requiring a separate dispatch call, each one making the corridor worse for every vehicle behind it.

This is a **cascade event** — and today's traffic management systems treat every incident as a standalone event, dispatching resources only *after* a corridor collapses.

---

## 💡 The Solution

The **ASTRAM Cascade Intelligence System** analyses every incoming traffic event from the ASTRAM platform and detects whether the *first* incident on a corridor is an early warning of a chain reaction.

When a high-risk pattern is matched, the system fires a **pre-emptive deployment alert within 30 seconds** — not a chart, not a forecast, but a specific instruction:

> *"Deploy 8 traffic police, 2 breakdown recovery units, and 4 barricades to Hosur Road — estimated clearance: 45 minutes."*

---

## 📊 Key Findings (8,173 Real ASTRAM Events · Nov 2023 – Apr 2024)

| Metric | Value |
|---|---|
| Cascade seed events identified | **176** seeds → **1,499** secondary events |
| Corridors analysed | **14** named Bengaluru corridors |
| Highest-risk corridor | **Hosur Road** — 52 cascade triggers, 558 secondary events |
| Response improvement | Reactive 30+ min dispatch → **Pre-emptive alert in seconds** |

---

## 🏗️ System Architecture

```
ASTRAM Event Feed
       │
       ▼
┌─────────────────────────┐
│  EDA & Feature          │  eda_feature_engineering.py
│  Engineering            │  → Corridor risk scores, peak-hour flags,
└──────────┬──────────────┘    temporal features
           │
           ▼
┌─────────────────────────┐
│  ML Pipeline            │  model_pipeline.py
│  ┌──────────────────┐   │  → Cascade Seed Classifier (XGBoost)
│  │ Cascade Detector │   │  → Severity Estimator (XGBoost)
│  └──────────────────┘   │  → Incident Count Predictor (XGBoost / Poisson)
│  ┌──────────────────┐   │
│  │ Severity Model   │   │
│  └──────────────────┘   │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Resource Optimizer     │  resource_optimizer.py
│                         │  → Generates actionable deployment plan
└──────────┬──────────────┘    (officers, equipment, time-to-clear)
           │
           ▼
┌─────────────────────────┐
│  Streamlit Dashboard    │  app.py
│                         │  → Duty-officer alert cards
└─────────────────────────┘    → Live corridor risk view
```

---

## 📁 Project Structure

```
Traffic/
├── app.py                        # Streamlit dashboard (main entry point)
├── model_pipeline.py             # ML training & evaluation pipeline
├── eda_feature_engineering.py    # EDA, feature engineering & visualisations
├── resource_optimizer.py         # Deployment plan generator
├── analyze.py                    # Post-hoc analysis utilities
├── concept_note.md               # Project concept and background
├── USER_MANUAL.md                # Full operational user manual
├── deployment_plan.json          # Sample generated deployment plan
├── deployment_plan.csv           # Sample generated deployment plan (CSV)
├── eda_outputs/                  # Generated EDA charts (PNGs)
│   ├── plot1_hourly_distribution.png
│   ├── plot2_top_corridors.png
│   ├── plot3_event_cause.png
│   ├── plot4_heatmap_hour_vs_day.png
│   └── plot5_resolution_by_cause.png
└── model_outputs/                # Trained model artefacts
    ├── cascade_seed_model.pkl
    ├── severity_model.pkl
    ├── count_model.pkl
    ├── cascade_threshold.txt
    ├── severity_target_encoder.pkl
    └── model_a_shap.png
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Khushi36/astram-cascade-intelligence.git
cd astram-cascade-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit dashboard
```bash
streamlit run app.py
```

### 4. Re-train models (optional)
```bash
python eda_feature_engineering.py   # Generates engineered dataset
python model_pipeline.py            # Trains and saves model artefacts
```

---

## 🧠 ML Models

| Model | Algorithm | Purpose |
|---|---|---|
| Cascade Seed Classifier | XGBoost | Predicts if an event will trigger a cascade |
| Severity Estimator | XGBoost | Estimates cascade severity level |
| Incident Count Predictor | XGBoost (Poisson) | Predicts number of follow-on incidents |

Key features used: corridor ID, incident type, hour of day, day of week, peak-hour flag, historical cascade frequency, rolling incident rate.

---

## 📋 Requirements

- Python 3.9+
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- shap
- matplotlib
- seaborn
- plotly

---

## 📖 Documentation

- [`concept_note.md`](concept_note.md) — Problem statement and system overview
- [`USER_MANUAL.md`](USER_MANUAL.md) — Full operational guide for duty officers

---

## 🗺️ Roadmap (Next 90 Days)

- [ ] Live connection to real-time ASTRAM event feed
- [ ] WhatsApp push alerts with one-tap confirmation
- [ ] Field feedback loop (actual clearance times → model retraining)
- [ ] Expansion to additional cities beyond Bengaluru

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built on 8,173 verified ASTRAM platform events. Demo corridor: Mysore Road, March 7, 2024 — a waterlogging event at 10:51 IST triggered 8 incidents in 76 minutes, 6 requiring road closure. The system would have fired a pre-deployment alert at 10:51.*
