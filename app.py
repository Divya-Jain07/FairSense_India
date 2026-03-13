import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

st.set_page_config(page_title="FareSense India", page_icon="🚕", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: sans-serif;
    background-color: #DCE0E8 !important;
    color: #27363F !important;
}
section[data-testid="stAppViewContainer"] { background-color: #DCE0E8 !important; }
.stApp { background-color: #DCE0E8 !important; }

/* labels */
label, .stSlider label { color: #27363F !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; font-weight: 600 !important; }

/* selectbox */
div[data-testid="stSelectbox"] > div > div {
    background-color: #ffffff !important;
    border: 0.5px solid #b8c4cc !important;
    color: #27363F !important;
    border-radius: 6px !important;
}

/* slider track */
div[data-testid="stSlider"] > div > div { background: #b8c4cc !important; }
div[data-testid="stSlider"] > div > div > div { background: #27363F !important; }

.header-card {
    padding: 36px 0 24px 0;
    border-bottom: 2px solid #27363F;
    margin-bottom: 24px;
    text-align: center;
}
.app-title {
    font-size: 32px;
    font-weight: 700;
    color: #27363F;
    letter-spacing: -0.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.app-title span { color: #6B212C; }
.app-sub { font-size: 14px; color: #685652; letter-spacing: 1.5px; text-transform: uppercase; font-weight: 500; }

.section-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #27363F;
    margin-bottom: 12px;
    margin-top: 4px;
}

div[data-testid="stButton"] > button { cursor: pointer !important; }

.veh-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; margin-bottom: 20px; }
.veh-btn {
    background: #ffffff;
    border: 1.5px solid #b8c4cc;
    border-radius: 6px;
    padding: 14px 8px;
    text-align: center;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    color: #8EA1AE;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    transition: all 0.15s;
}
.veh-btn:hover { border-color: #27363F; color: #27363F; }
.veh-btn.active {
    background: #27363F;
    border-color: #27363F;
    color: #DCE0E8;
}
.veh-btn.active-red {
    background: #6B212C;
    border-color: #6B212C;
    color: #DCE0E8;
}

.conditions-box {
    background: #ffffff;
    border: 0.5px solid #b8c4cc;
    border-left: 3px solid #6B212C;
    border-radius: 6px;
    padding: 14px 16px;
    margin-top: 6px;
    margin-bottom: 4px;
}
.cond-title { font-size: 10px; color: #8EA1AE; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px; font-weight: 600; }
.pills      { display: flex; gap: 6px; flex-wrap: wrap; }
.pill       { font-size: 10px; padding: 4px 12px; border-radius: 3px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
.pill-gray   { background: #DCE0E8; color: #685652; }
.pill-teal   { background: #e0e8ec; color: #27363F; }
.pill-orange { background: #f0e6d8; color: #685652; }
.pill-red    { background: #6B212C; color: #DCE0E8; }
.pill-green  { background: #27363F; color: #DCE0E8; }

.fare-display {
    background: #27363F;
    border-radius: 8px;
    padding: 32px 28px 24px;
    text-align: center;
    margin-top: 8px;
    position: relative;
    overflow: hidden;
}
.fare-display::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: #6B212C;
}
.fare-label-sm {
    font-size: 10px;
    color: #8EA1AE;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 12px;
    font-weight: 600;
}
.fare-amount { font-size: 56px; font-weight: 700; color: #DCE0E8; line-height: 1; letter-spacing: -1px; }
.fare-sub    { font-size: 12px; color: #8EA1AE; margin-top: 10px; }

.breakdown-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 6px;
    margin-top: 20px;
}
.metric-card {
    background: #1e2d36;
    border-radius: 6px;
    padding: 12px;
    text-align: left;
}
.metric-label { font-size: 9px; color: #8EA1AE; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.metric-val   { font-size: 18px; font-weight: 700; color: #DCE0E8; }

.tags { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 18px; justify-content: center; }
.tag  {
    font-size: 10px;
    padding: 4px 12px;
    background: #1e2d36;
    border-radius: 3px;
    color: #8EA1AE;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}

/* estimate button */
div[data-testid="stButton"] > button {
    background: #6B212C !important;
    color: #DCE0E8 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 16px 0 !important;
    width: 60% !important;
    margin: 12px auto !important;
    display: block !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
div[data-testid="stButton"] > button:hover {
    background: #27363F !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button:active {
    background: #1a2428 !important;
    transform: translateY(1px) !important;
}

.footer {
    border-top: 2px solid #27363F;
    padding-top: 20px;
    margin-top: 28px;
    padding-bottom: 20px;
}
.footer-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 12px;
    gap: 16px;
}
.footer-left { flex: 1; }
.footer-section-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #27363F;
    margin-bottom: 8px;
}
.footer-desc { font-size: 12px; color: #685652; line-height: 1.7; }
.footer-author { font-size: 11px; color: #8EA1AE; white-space: nowrap; padding-top: 2px; }
.footer-author span { color: #27363F; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin-top: 14px;
}
.footer-metric {
    background: #27363F;
    border-radius: 6px;
    padding: 14px 16px;
}
.footer-metric-label { font-size: 9px; color: #8EA1AE; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
.footer-metric-val   { font-size: 20px; font-weight: 700; color: #DCE0E8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        with open("ride_fare_model.pkl",    "rb") as f: model    = pickle.load(f)
        with open("ride_fare_features.pkl", "rb") as f: features = pickle.load(f)
        with open("ohe_encoder.pkl",        "rb") as f: ohe      = pickle.load(f)
        with open("city_rates.pkl",         "rb") as f: rates    = pickle.load(f)
        return model, features, ohe, rates, True
    except FileNotFoundError:
        return None, None, None, None, False

model, feature_cols, ohe, city_rates, loaded = load_artifacts()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_traffic(hour, is_weekend):
    if not is_weekend:
        if hour in range(7, 11):  return 4, "Very high traffic", "red"
        if hour in range(17, 21): return 4, "Very high traffic", "red"
        if hour in range(11, 17): return 2, "Medium traffic",    "orange"
        if hour in range(21, 24): return 2, "Medium traffic",    "orange"
        return 1, "Low traffic", "green"
    return (2, "Medium traffic", "orange") if 10 <= hour <= 20 else (1, "Low traffic", "green")

def get_rush(hour, is_weekend):
    if not is_weekend:
        if hour in range(7, 11):  return "Morning rush", "orange"
        if hour in range(17, 21): return "Evening rush", "red"
    return None, None

def get_surge(hour, traffic, is_weekend):
    if traffic == 4:                  return 1.08
    elif traffic == 3:                return 1.05
    elif traffic == 2 and is_weekend: return 1.03
    elif hour <= 4:                   return 1.05
    else:                             return 1.0

def hour_to_period(hour):
    if hour <= 5:    return 0
    elif hour <= 11: return 1
    elif hour <= 17: return 2
    else:            return 3

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <div class="app-title">Fare<span>Sense</span> India</div>
    <div class="app-sub">Fare estimates across 10 cities &nbsp;·&nbsp; Ola &nbsp;·&nbsp; Uber &nbsp;·&nbsp; Rapido</div>
</div>
""", unsafe_allow_html=True)

if not loaded:
    st.error("Model files not found. Please run ride_fare_model.py first.")
    st.stop()

# ─────────────────────────────────────────────
# TWO COLUMN LAYOUT
# ─────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Ride details</div>', unsafe_allow_html=True)

    cities   = list(city_rates.keys())
    vehicles = ["Auto", "Bike", "Sedan", "SUV"]

    city     = st.selectbox("City", options=cities)
    distance = st.slider("Distance (km)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)

    st.markdown('<div class="section-label" style="margin-top:12px">Vehicle type</div>', unsafe_allow_html=True)

    if "vehicle" not in st.session_state:
        st.session_state.vehicle = "Auto"

    v1, v2, v3, v4 = st.columns(4)
    for col, vname in zip([v1, v2, v3, v4], vehicles):
        if col.button(vname, key=f"vbtn_{vname.replace(' ','_')}", use_container_width=True):
            st.session_state.vehicle = vname
            st.rerun()
    vehicle = st.session_state.vehicle

    now        = datetime.now()
    is_weekend = now.weekday() >= 5
    hour       = st.slider("Time of day", min_value=0, max_value=23,
                           value=now.hour, format="%d:00")

    traffic_level, traffic_label, traffic_color = get_traffic(hour, is_weekend)
    rush_label, rush_color                       = get_rush(hour, is_weekend)
    surge                                        = get_surge(hour, traffic_level, is_weekend)
    time_period                                  = hour_to_period(hour)
    day_label                                    = "Weekend" if is_weekend else "Weekday"
    surge_label                                  = f"{surge}x surge" if surge > 1.0 else "No surge"
    surge_color                                  = "red" if surge > 1.0 else "green"

    pills_html  = f'<span class="pill pill-gray">{day_label}</span>'
    pills_html += f'<span class="pill pill-{traffic_color}">{traffic_label}</span>'
    if rush_label:
        pills_html += f'<span class="pill pill-{rush_color}">{rush_label}</span>'
    pills_html += f'<span class="pill pill-{surge_color}">{surge_label}</span>'

    st.markdown(f"""
    <div class="conditions-box">
        <div class="cond-title">Auto-detected conditions</div>
        <div class="pills">{pills_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    if "estimated" not in st.session_state:
        st.session_state.estimated = False

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("Estimate Fare", use_container_width=True):
            st.session_state.estimated = True

with right:
    st.markdown("<div style='padding-top:40px'>", unsafe_allow_html=True)
    if st.session_state.estimated:
        def make_prediction(surge_val):
            num = pd.DataFrame([{
                "Ride_Distance"   : distance,
                "Time_Period"     : time_period,
                "Traffic_Level"   : traffic_level,
                "Surge_Multiplier": surge_val,
                "Is_Weekend"      : int(is_weekend),
            }])
            vehicle_model = "Bike Taxi" if vehicle == "Bike" else vehicle
            cat      = ohe.transform(pd.DataFrame([{"City": city, "Vehicle_Type": vehicle_model}]))
            cat_cols = ohe.get_feature_names_out(["City", "Vehicle_Type"]).tolist()
            cat_df   = pd.DataFrame(cat, columns=cat_cols)
            sample   = pd.concat([num, cat_df], axis=1)[feature_cols]
            return max(model.predict(sample)[0], 20.0)

        fare_base  = make_prediction(1.0)
        fare_surge = make_prediction(surge)

        ampm         = "AM" if hour < 12 else "PM"
        hour_display = f"{hour % 12 or 12}:00 {ampm}"
        surge_note   = f"includes {surge}x surge" if surge > 1.0 else "no surge active"

        vehicle_key     = "Bike Taxi" if vehicle == "Bike" else vehicle
        vrates          = city_rates[city][vehicle_key]
        base_fare       = vrates["base"] + distance * vrates["per_km"]
        traffic_add     = base_fare * 0.03 * (traffic_level - 1)
        surge_add       = base_fare * (surge - 1.0)
        night_add       = base_fare * 0.25 if hour <= 5 or hour >= 23 else 0
        breakdown_total = base_fare + traffic_add + surge_add + night_add
        scale           = fare_surge / breakdown_total if breakdown_total > 0 else 1.0

        st.markdown(f"""
        <div class="fare-display">
            <div class="fare-label-sm">Estimated fare &nbsp;·&nbsp; {surge_note}</div>
            <div class="fare-amount">&#8377;{fare_surge:,.0f}</div>
            <div class="fare-sub">Base fare without surge: &#8377;{fare_base:,.0f} &nbsp;·&nbsp; &#177; a few rupees depending on route</div>
            <div class="breakdown-grid">
                <div class="metric-card">
                    <div class="metric-label">Base fare</div>
                    <div class="metric-val">&#8377;{base_fare * scale:.0f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Traffic</div>
                    <div class="metric-val">&#8377;{traffic_add * scale:.0f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Surge</div>
                    <div class="metric-val">&#8377;{surge_add * scale:.0f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Night</div>
                    <div class="metric-val">&#8377;{night_add * scale:.0f}</div>
                </div>
            </div>
            <div class="tags">
                <span class="tag">{city}</span>
                <span class="tag">{distance} km</span>
                <span class="tag">{vehicle}</span>
                <span class="tag">{hour_display}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="height:60vh;display:flex;align-items:center;justify-content:center;text-align:center;">
            <div>
                <div style="font-size:13px;color:#8EA1AE;text-transform:uppercase;letter-spacing:1.5px;">Your fare estimate will appear here</div>
                <div style="font-size:11px;color:#b8c4cc;margin-top:8px;">Fill in your ride details and click Estimate Fare</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-top">
        <div class="footer-left">
            <div class="footer-section-label">Model performance</div>
            <div class="footer-desc">
                Fare rates were collected from Ola, Uber and Rapido across 10 Indian cities.
                A machine learning model was trained on synthetic rides generated from
                real-world rates, learning how distance, time, traffic, surge, and vehicle
                type affect pricing.
            </div>
        </div>
        <div class="footer-author">by <span>Divya Jain</span></div>
    </div>
    <div class="metrics-row">
        <div class="footer-metric">
            <div class="footer-metric-label">R&#178; Score</div>
            <div class="footer-metric-val">97.87%</div>
        </div>
        <div class="footer-metric">
            <div class="footer-metric-label">MAE</div>
            <div class="footer-metric-val">&#8377;45.50</div>
        </div>
        <div class="footer-metric">
            <div class="footer-metric-label">RMSE</div>
            <div class="footer-metric-val">&#8377;67.62</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
