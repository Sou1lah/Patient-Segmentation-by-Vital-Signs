"""
Real-time Patient Segmentation Dashboard (synthetic data)

Run:
    pip install streamlit scikit-learn pandas numpy plotly
    streamlit run apps/realtime_dashboard.py

What it does:
- Generates synthetic patients (50-100) with realistic vital ranges.
- Simulates real-time streaming by updating vitals every `interval` seconds (1-2s default).
- Scales features and applies KMeans clustering dynamically to the current snapshot.
- Displays a Streamlit dashboard with:
  * Live table of patient vitals and cluster assignment (color-coded)
  * Scatter plot (BP vs HR) colored by cluster
  * Cluster-level alerts when vitals are outside normal ranges

Notes:
- This is a simulation (synthetic data). It demonstrates how streaming data might be connected to a real-time clustering dashboard in a hospital.
- The clustering runs on the current patient snapshot each tick (fits KMeans on entire cohort).

Author: GitHub Copilot (Raptor mini (Preview))
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import streamlit as st
from collections import deque

# -----------------------------
# Synthetic data generation
# -----------------------------

@dataclass
class Patient:
    id: int
    systolic_bp: float
    heart_rate: float
    temperature: float
    spo2: float

    def to_dict(self):
        return {'id': self.id, 'Systolic_BP': self.systolic_bp, 'Heart_Rate': self.heart_rate, 'Temperature': self.temperature, 'SpO2': self.spo2}


def generate_patients(n: int = 75, seed: int = 42) -> pd.DataFrame:
    """Generate baseline patient vitals for `n` patients."""
    rng = np.random.default_rng(seed)
    systolic = rng.uniform(90, 140, size=n)  # mmHg
    hr = rng.uniform(60, 120, size=n)  # bpm
    temp = rng.uniform(36.0, 39.0, size=n)  # C
    spo2 = rng.uniform(92, 100, size=n)  # % SpO2
    patients = [Patient(i, float(systolic[i]), float(hr[i]), float(temp[i]), float(spo2[i])) for i in range(n)]
    return pd.DataFrame([p.to_dict() for p in patients])


def apply_random_fluctuations(df: pd.DataFrame, magnitude: float = 1.0, spike_prob: float = 0.01, rng=None) -> pd.DataFrame:
    """Apply small time-step fluctuations to each vital to simulate streaming updates.

    magnitude: controls the standard deviation of changes.
    spike_prob: small probability to create transient abnormal spike.
    """
    if rng is None:
        rng = np.random.default_rng()
    df2 = df.copy()
    # small Gaussian noise
    df2['Systolic_BP'] = df2['Systolic_BP'] + rng.normal(0, 0.8 * magnitude, size=len(df2))
    df2['Heart_Rate'] = df2['Heart_Rate'] + rng.normal(0, 0.6 * magnitude, size=len(df2))
    df2['Temperature'] = df2['Temperature'] + rng.normal(0, 0.02 * magnitude, size=len(df2))
    if 'SpO2' in df2.columns:
        df2['SpO2'] = df2['SpO2'] + rng.normal(0, 0.4 * magnitude, size=len(df2))

    # occasional spike or dip to mimic events
    spikes = rng.random(size=len(df2)) < spike_prob
    if spikes.any():
        df2.loc[spikes, 'Heart_Rate'] += rng.uniform(-15, 25, size=spikes.sum())
        df2.loc[spikes, 'Systolic_BP'] += rng.uniform(-20, 20, size=spikes.sum())
        if 'SpO2' in df2.columns:
            df2.loc[spikes, 'SpO2'] += rng.uniform(-6, 2, size=spikes.sum())

    # keep values in plausible physiological bounds
    df2['Systolic_BP'] = df2['Systolic_BP'].clip(60, 220)
    df2['Heart_Rate'] = df2['Heart_Rate'].clip(30, 220)
    df2['Temperature'] = df2['Temperature'].clip(34.0, 42.0)
    if 'SpO2' in df2.columns:
        df2['SpO2'] = df2['SpO2'].clip(60, 100)

    return df2


# -----------------------------
# Clustering utilities
# -----------------------------

def cluster_snapshot(df: pd.DataFrame, n_clusters: int = 3) -> Tuple[np.ndarray, dict]:
    """Scale features, fit KMeans on snapshot, and return labels and diagnostics."""
    features = ['Systolic_BP', 'Heart_Rate', 'Temperature']
    X = df[features].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled).astype(int)
    diagnostics = {}
    if n_clusters >= 2 and len(df) >= n_clusters:
        try:
            diagnostics['silhouette'] = float(silhouette_score(X_scaled, labels))
        except Exception:
            diagnostics['silhouette'] = float('nan')
    else:
        diagnostics['silhouette'] = float('nan')
    diagnostics['centroids_scaled'] = model.cluster_centers_.tolist()
    diagnostics['inertia'] = float(model.inertia_)
    return labels, diagnostics


# -----------------------------
# Alert rules
# -----------------------------

NORMAL_RANGES = {
    'Systolic_BP': (90, 140),
    'Heart_Rate': (60, 100),
    'Temperature': (36.0, 37.5),
    'SpO2': (92, 100)
}


def vital_alert(row: pd.Series) -> str:
    """Return alert text based on a single patient row."""
    alerts = []
    if row['Systolic_BP'] < NORMAL_RANGES['Systolic_BP'][0] or row['Systolic_BP'] > NORMAL_RANGES['Systolic_BP'][1]:
        alerts.append('BP')
    if row['Heart_Rate'] < NORMAL_RANGES['Heart_Rate'][0] or row['Heart_Rate'] > NORMAL_RANGES['Heart_Rate'][1]:
        alerts.append('HR')
    if row['Temperature'] < NORMAL_RANGES['Temperature'][0] or row['Temperature'] > NORMAL_RANGES['Temperature'][1]:
        alerts.append('Temp')
    if 'SpO2' in row.index and (row['SpO2'] < NORMAL_RANGES['SpO2'][0] or row['SpO2'] > NORMAL_RANGES['SpO2'][1]):
        alerts.append('SpO2')
    return ','.join(alerts) if alerts else ''


# -----------------------------
# Streamlit app
# -----------------------------

st.set_page_config(page_title='Patient Segmentation Dashboard', layout='wide')
st.title('Patient Segmentation — Real-time Simulation')

# --- Theme CSS: default and monitor presets ---
if 'theme' in globals() and theme == 'Monitor':
    st.markdown(
        """<style>
        /* Base monitor background */
        .stApp { background: linear-gradient(#030405,#081219); color: #e6eef8; }
        .monitor-bg { background: #0b1220; color: #e6eef8; padding: 12px 16px; border-radius: 6px; }
        .patient-card { background: linear-gradient(90deg,#08101a,#0f2430); padding: 10px; border-radius:8px; margin:6px; color:#e6eef8; box-shadow: 0 2px 8px rgba(0,0,0,0.6); }
        .big-num { font-size: 28px; font-weight:800; color:#ffffff; }
        .big-hr { font-size:48px; font-weight:900; color:#ffffff; }
        .small-label { font-size:12px; color:#a6b7c9 }
        .label-spo2 { color:#2cff6d; font-weight:700; }
        .label-hr { color:#ffffff; font-weight:800; }
        .label-bp { color:#ffd166; font-weight:700; }
        .label-temp { color:#ff8b4b; font-weight:700; }
        .alert-badge { background: #ff5c5c; color: white; padding: 4px 8px; border-radius: 4px; font-weight:700; }
        .ok-badge { background: #19b76e; color: white; padding: 4px 8px; border-radius: 4px; font-weight:700; }
        hr.monitor-sep { border:none; border-top:1px solid rgba(255,255,255,0.06); margin:6px 0; }
        .metric-big { font-size: 22px; font-weight:800; color:#dfeffd; }
        /* tweak Plotly text colors */
        .plotly .main-svg { color: #e6eef8; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """<style>
        .patient-card { background: linear-gradient(90deg,#fff,#f6f8fb); padding: 10px; border-radius:8px; margin:6px; }
        .big-num { font-size: 28px; font-weight:700; }
        .small-label { font-size:12px; color:#333 }
        .alert-badge { background: #d9534f; color: white; padding: 4px 8px; border-radius: 4px; font-weight:700; }
        .ok-badge { background: #2ecc71; color: white; padding: 4px 8px; border-radius: 4px; font-weight:700; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_patient_cards(df: pd.DataFrame, n_cols: int = 4, show_count: int = 12):
    """Render a grid of patient cards styled like a monitor.

    - df: patient dataframe including columns ['id','Systolic_BP','Heart_Rate','Temperature','SpO2','Cluster','Alert']
    - n_cols: number of columns in the grid
    - show_count: maximum number of patient cards to show
    """
    # order by alerts first then by cluster
    dfc = df.copy().sort_values(['Alert', 'Cluster'], ascending=[False, True]).head(show_count)
    rows = math.ceil(len(dfc) / n_cols)
    i = 0
    for r in range(rows):
        cols = st.columns(n_cols)
        for c in cols:
            if i >= len(dfc):
                c.write("")
            else:
                row = dfc.iloc[i]
                # determine badge
                badge_html = f"<div class=\"alert-badge\">ALERT</div>" if row['Alert'] else f"<div class=\"ok-badge\">OK</div>"
                # color-coded vitals for monitor style
                bp_html = f"<div class='small-label label-bp'>BP</div><div class='big-num'>{row['Systolic_BP']:.0f} mmHg</div>"
                hr_html = f"<div class='small-label label-hr'>HR</div><div class='big-hr'>{row['Heart_Rate']:.0f}</div>"
                temp_html = f"<div class='small-label label-temp'>Temp</div><div class='big-num'>{row['Temperature']:.1f}°C</div>"
                spo2_html = f"<div class='small-label label-spo2'>SpO2</div><div class='big-num' style='color:#2cff6d'>{row.get('SpO2',0):.0f}%</div>"
                content = f"<div class=\"patient-card\">\n  <div style='display:flex;justify-content:space-between;align-items:center;'>\n    <div>\n      <div class='small-label'>Patient</div>\n      <div class='big-num'>ID {int(row['id'])}</div>\n    </div>\n    <div style='text-align:right;'>\n      {badge_html}\n    </div>\n  </div>\n  <hr class='monitor-sep'>\n  <div style='display:flex;justify-content:space-between;align-items:center;'>\n    <div style='width:25%;'>{bp_html}</div>\n    <div style='width:30%;text-align:center;'>{hr_html}</div>\n    <div style='width:20%;text-align:right;'>{spo2_html}</div>\n    <div style='width:20%;text-align:right;'>{temp_html}</div>\n  </div>\n  <div style='margin-top:6px;'><span class='small-label'>Cluster</span> <strong>{int(row['Cluster'])}</strong></div>\n</div>"
                c.markdown(content, unsafe_allow_html=True)
            i += 1

# Sidebar controls
with st.sidebar:
    st.header('Simulation controls')
    theme = st.selectbox('Theme', ['Default','Monitor'], index=1)
    n_patients = st.number_input('Number of patients', min_value=50, max_value=200, value=75, step=5)
    n_clusters = st.number_input('K (clusters)', min_value=2, max_value=10, value=3, step=1)
    interval_slider = st.slider('Update interval (seconds)', min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    force_one_sec = st.checkbox('Update every second (lock)', value=True)
    # when locked, force interval to 1.0s; otherwise use slider value
    interval = 1.0 if force_one_sec else float(interval_slider)
    batch_size = st.selectbox('Update batch size', options=[1, 3, 5, 10], index=1)
    magnitude = st.slider('Fluctuation magnitude', 0.1, 3.0, 1.0, step=0.1)
    history_len = st.slider('History length (points)', min_value=10, max_value=240, value=60, step=10)
    seed = st.number_input('Random seed', value=42)
    start = st.button('Start streaming')
    stop = st.button('Stop streaming')
    st.markdown('---')
    st.markdown('**Notes**: Streaming fits KMeans on current snapshot; results are for demo only.')

# Initialize session state
if 'df' not in st.session_state or st.session_state.get('n_patients') != n_patients or st.session_state.get('seed') != seed:
    st.session_state.df = generate_patients(int(n_patients), int(seed))
    st.session_state.n_patients = n_patients
    st.session_state.seed = seed
    st.session_state.running = False
    st.session_state.rng = np.random.default_rng(seed)
    # initialize HR history buffer for each patient
    st.session_state.hr_history = {}
    for _, row in st.session_state.df.iterrows():
        st.session_state.hr_history[int(row['id'])] = deque([float(row['Heart_Rate'])] * int(history_len), maxlen=int(history_len))
    st.session_state.history_len = int(history_len)
else:
    # if only history_len changed, resize existing buffers
    if st.session_state.get('history_len') != int(history_len):
        old_len = st.session_state.get('history_len', int(history_len))
        new_len = int(history_len)
        for pid, buf in st.session_state.hr_history.items():
            if new_len > old_len:
                # pad with last value
                last = buf[-1] if len(buf) > 0 else 60.0
                for _ in range(new_len - old_len):
                    buf.append(last)
                buf = deque(buf, maxlen=new_len)
                st.session_state.hr_history[pid] = buf
            else:
                # truncate
                st.session_state.hr_history[pid] = deque(list(buf)[-new_len:], maxlen=new_len)
        st.session_state.history_len = new_len

# Start/stop control
if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# Main layout: left controls and table, right visualization
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader('Patients')
    # container for live table
    table_container = st.empty()
    metrics_container = st.empty()

with col2:
    st.subheader('Visualizations')
    scatter_container = st.empty()
    cluster_metrics = st.empty()

# Simulation loop: update data while running
if st.session_state.running:
    # Run a timed loop; this will block until stopped (ok for simple demo)
    st.info('Streaming started — click Stop to end the simulation')
    try:
        while st.session_state.running:
            # Update a random subset (batch) of patients to simulate streaming arrival
            idx = st.session_state.rng.choice(st.session_state.df.index, size=batch_size, replace=False)
            df_new = st.session_state.df.copy()
            df_new.loc[idx, ['Systolic_BP', 'Heart_Rate', 'Temperature']] = \
                apply_random_fluctuations(df_new.loc[idx, ['Systolic_BP', 'Heart_Rate', 'Temperature']], magnitude=magnitude, rng=st.session_state.rng)[['Systolic_BP', 'Heart_Rate', 'Temperature']]
            # Update session state
            st.session_state.df.loc[idx, ['Systolic_BP', 'Heart_Rate', 'Temperature']] = df_new.loc[idx, ['Systolic_BP', 'Heart_Rate', 'Temperature']]

            # Clustering on the current snapshot
            labels, diag = cluster_snapshot(st.session_state.df, n_clusters=int(n_clusters))
            st.session_state.df['Cluster'] = labels
            st.session_state.df['Alert'] = st.session_state.df.apply(vital_alert, axis=1)

            # Table: highlight alerts and cluster color
            df_display = st.session_state.df.copy()
            df_display['Cluster'] = df_display['Cluster'].astype(int)
            df_display = df_display.sort_values('Cluster')

            # Compute alert/safe counts and show metrics
            total = len(st.session_state.df)
            n_alerts = int((st.session_state.df['Alert'] != '').sum())
            n_safe = int(total - n_alerts)
            pct_safe = 100.0 * n_safe / total if total > 0 else 0.0
            with metrics_container.container():
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric('Cluster count', int(n_clusters))
                col_b.metric('Patients SAFE', f"{n_safe} ({pct_safe:.0f}%)", delta=f"-{n_alerts} alerts")
                col_c.metric('Silhouette', f"{diag.get('silhouette', float('nan')):.3f}")
                # show average HR as a fast glance (match monitor theme)
                avg_hr = int(st.session_state.df['Heart_Rate'].mean()) if len(st.session_state.df) else 0
                col_d.metric('Avg HR', f"{avg_hr} bpm")

            # Render patient cards (monitor style)
            render_patient_cards(st.session_state.df, n_cols=3, show_count=12)

            # Scatter plot: BP vs HR colored by cluster (dark monitor style)
            # choose a neon-like palette suitable for monitor theme
            palette = ['#2cff6d', '#ff5c5c', '#ffd166', '#6bcBff', '#b983ff', '#ff7ab6', '#60d394', '#ffb86b', '#9be7ff', '#f3f48c']
            fig = px.scatter(st.session_state.df, x='Systolic_BP', y='Heart_Rate', color='Cluster', hover_data=['id', 'Temperature', 'SpO2', 'Alert'], color_discrete_sequence=palette, template='plotly_dark')
            fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='#000')))
            bg = '#08101a' if (('theme' in globals() and theme == 'Monitor')) else '#ffffff'
            fig.update_layout(paper_bgcolor=bg, plot_bgcolor=bg, title='Systolic BP vs Heart Rate (live)', xaxis_title='Systolic BP (mmHg)', yaxis_title='Heart Rate (bpm)', legend=dict(bgcolor='rgba(0,0,0,0.1)'))

            # Mark points with alerts with larger size and red border
            mask_alert = st.session_state.df['Alert'] != ''
            if mask_alert.any():
                # overlay alert points
                alerts = st.session_state.df[mask_alert]
                fig.add_scatter(x=alerts['Systolic_BP'], y=alerts['Heart_Rate'], mode='markers', marker=dict(symbol='x', size=16, color='red'), name='Alerts')

            scatter_container.plotly_chart(fig, use_container_width=True)

            # Cluster-level metrics
            cluster_stats = st.session_state.df.groupby('Cluster')[['Systolic_BP', 'Heart_Rate', 'Temperature']].mean().round(2).reset_index()
            cluster_metrics.dataframe(cluster_stats)

            # Sleep for the interval
            time.sleep(float(interval))
    except Exception as e:
        st.error(f'Streaming error: {e}')
else:
    st.info('Simulation stopped. Click Start streaming to begin.')
    # Show a static snapshot
    df_snapshot = st.session_state.df.copy()
    if 'Cluster' not in df_snapshot.columns:
        labels, _ = cluster_snapshot(df_snapshot, n_clusters=int(n_clusters))
        df_snapshot['Cluster'] = labels
    df_snapshot['Alert'] = df_snapshot.apply(vital_alert, axis=1)
    # Basic visualization
    # Compute safe/alert counts
    total = len(df_snapshot)
    n_alerts = int((df_snapshot['Alert'] != '').sum())
    n_safe = int(total - n_alerts)
    pct_safe = 100.0 * n_safe / total if total > 0 else 0.0
    col_a, col_b, col_c, col_d = metrics_container.columns(4)
    col_a.metric('Cluster count', int(n_clusters))
    col_b.metric('Patients SAFE', f"{n_safe} ({pct_safe:.0f}%)", delta=f"-{n_alerts} alerts")
    col_c.metric('Silhouette', "N/A")
    avg_hr = int(df_snapshot['Heart_Rate'].mean()) if len(df_snapshot) else 0
    col_d.metric('Avg HR', f"{avg_hr} bpm")

    palette = ['#2cff6d', '#ff5c5c', '#ffd166', '#6bcBff', '#b983ff', '#ff7ab6', '#60d394', '#ffb86b', '#9be7ff', '#f3f48c']
    fig = px.scatter(df_snapshot, x='Systolic_BP', y='Heart_Rate', color='Cluster', hover_data=['id', 'Temperature', 'SpO2', 'Alert'], color_discrete_sequence=palette, template='plotly_dark')
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='#000')))
    bg = '#08101a' if (('theme' in globals() and theme == 'Monitor')) else '#ffffff'
    fig.update_layout(paper_bgcolor=bg, plot_bgcolor=bg, title='Systolic BP vs Heart Rate (snapshot)', xaxis_title='Systolic BP (mmHg)', yaxis_title='Heart Rate (bpm)')
    scatter_container.plotly_chart(fig, use_container_width=True)
    # render cards in static mode
    render_patient_cards(df_snapshot, n_cols=3, show_count=12)
    cluster_stats = df_snapshot.groupby('Cluster')[['Systolic_BP','Heart_Rate','Temperature']].mean().round(2).reset_index()
    cluster_metrics.dataframe(cluster_stats)

# Footer explanation
st.markdown('---')
st.markdown('**Simulation details:** The app generates synthetic patients and updates a small batch every tick to simulate streaming. Each tick re-fits KMeans on the current snapshot and updates cluster labels. Alerts are generated when vitals fall outside normal ranges. This simulates a hospital real-time dashboard for demonstration purposes.')
