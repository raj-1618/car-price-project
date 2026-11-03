import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model_engine import BRAND_CATALOG, load_cars24_dataset, train_cars24_models

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoTorque AI | Cars24 Mechanical Price Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# GLASSMORPHISM & MECHANICAL ANIMATIONS CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

.stApp {
    background-color: #080b10;
    background-image: 
        radial-gradient(circle at 12% 18%, rgba(255, 85, 0, 0.11) 0%, transparent 35%),
        radial-gradient(circle at 88% 82%, rgba(0, 242, 254, 0.09) 0%, transparent 40%),
        linear-gradient(135deg, #07090e 0%, #0e131d 50%, #090c12 100%);
    font-family: 'Inter', sans-serif;
}

@keyframes spinGear {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@keyframes pulseBorder {
    0% { box-shadow: 0 0 12px rgba(255, 85, 0, 0.25), inset 0 0 8px rgba(255, 85, 0, 0.1); }
    50% { box-shadow: 0 0 24px rgba(0, 242, 254, 0.45), inset 0 0 14px rgba(0, 242, 254, 0.2); }
    100% { box-shadow: 0 0 12px rgba(255, 85, 0, 0.25), inset 0 0 8px rgba(255, 85, 0, 0.1); }
}

@keyframes floatUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-glass {
    background: rgba(18, 24, 36, 0.68);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-left: 5px solid #FF5500;
    border-radius: 18px;
    padding: 26px 32px;
    margin-bottom: 22px;
    animation: floatUp 0.6s ease-out, pulseBorder 6s infinite ease-in-out;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.15rem;
    font-weight: 900;
    letter-spacing: 1.5px;
    background: linear-gradient(90deg, #FFFFFF 0%, #FF5500 55%, #00F2FE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.hero-sub {
    font-family: 'Rajdhani', sans-serif;
    color: #94A3B8;
    font-size: 1.1rem;
    letter-spacing: 0.8px;
    margin-top: 6px;
}

.gear-badge {
    font-size: 3.1rem;
    display: inline-block;
    animation: spinGear 10s linear infinite;
    filter: drop-shadow(0 0 12px rgba(255, 85, 0, 0.65));
}

.glass-card {
    background: rgba(20, 27, 42, 0.58);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
    transition: all 0.35s ease;
    animation: floatUp 0.5s ease-out;
    margin-bottom: 16px;
}

.glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(255, 85, 0, 0.55);
    box-shadow: 0 12px 32px rgba(255, 85, 0, 0.2);
}

.brand-card {
    background: linear-gradient(145deg, rgba(24, 32, 48, 0.75), rgba(13, 18, 28, 0.88));
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 14px;
    padding: 14px 8px;
    text-align: center;
    transition: all 0.3s ease;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}

.brand-card:hover {
    transform: translateY(-6px) scale(1.03);
    border-color: #00F2FE;
    box-shadow: 0 10px 24px rgba(0, 242, 254, 0.22);
}

.brand-logo {
    width: 42px;
    height: 42px;
    object-fit: contain;
    margin-bottom: 6px;
    transition: transform 0.3s ease;
}

.brand-card:hover .brand-logo {
    transform: scale(1.16);
}

.brand-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.86rem;
    font-weight: 700;
    color: #F8FAFC;
}

.brand-count {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.78rem;
    color: #FF5500;
    font-weight: 700;
}

.brand-tier {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.72rem;
    color: #38BDF8;
    text-transform: uppercase;
}

.metric-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.92rem;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #94A3B8;
}

.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    color: #F8FAFC;
}

.metric-badge {
    display: inline-block;
    margin-top: 6px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
    background: rgba(0, 242, 254, 0.14);
    color: #00F2FE;
    border: 1px solid rgba(0, 242, 254, 0.3);
}

.section-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.28rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: 1px;
    border-bottom: 2px solid rgba(255, 85, 0, 0.4);
    padding-bottom: 8px;
    margin: 20px 0 14px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDEBAR: ENGINE MODE & KAGGLE FILE OVERRIDE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 8px 0;">
            <div class="gear-badge">⚙️</div>
            <h2 style="font-family:'Orbitron',sans-serif; color:#FF5500; margin:6px 0 0 0; font-size:1.25rem;">AUTOTORQUE ECU</h2>
            <p style="font-family:'Rajdhani',sans-serif; color:#94A3B8; font-size:0.9rem;">Cars24 Kaggle Telemetry Engine</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    feature_mode_label = st.radio(
        "⚙️ Feature Telemetry Mode",
        options=[
            "Full Cars24 Spec (R² = 0.78)",
            "Strict 4-Feature Syllabus Mode (R² = 0.47)",
        ],
        index=0,
        help="Switch between full Cars24 columns (Brand, Car Model, Mileage, Fuel, Age, Drive, Body Type, Owner) and the 4 core features (Brand, Mileage, Fuel, Age).",
    )
    active_mode = "full" if "Full" in feature_mode_label else "core"

    uploaded_csv = st.file_uploader(
        "Override `cars_24_combined.csv` (Optional)",
        type=["csv"],
    )

    st.divider()
    st.markdown(
        """
        <div class="glass-card" style="padding:14px;">
            <div class="metric-title">KAGGLE DATASET ACTIVE</div>
            <p style="font-size:0.82rem; color:#CBD5E1; margin:6px 0;">
                <b>• File:</b> cars_24_combined.csv<br>
                <b>• Records:</b> 8,014 Verified Cars<br>
                <b>• Brands:</b> 16 Manufacturers<br>
                <b>• Models:</b> Linear Reg & Random Forest<br>
                <b>• Metrics:</b> MAE, RMSE, R²
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# LOAD DATA & TRAIN MODELS
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Calibrating Cars24 Telemetry & Training Regression Engines...")
def init_cars24_engine(file_buf, mode_key):
    df_clean = load_cars24_dataset(file_buf)
    bundle = train_cars24_models(df_clean, mode=mode_key)
    return df_clean, bundle

df, ml_bundle = init_cars24_engine(uploaded_csv, active_mode)
lr_metrics = ml_bundle["metrics"]["Linear Regression"]
rf_metrics = ml_bundle["metrics"]["Random Forest Regressor"]

# -----------------------------------------------------------------------------
# HERO BANNER
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero-glass">
    <div>
        <div class="hero-title">AUTOTORQUE // CARS24 AI VALUATION</div>
        <div class="hero-sub">MECHANICAL GLASSMORPHISM REGRESSION ENGINE • 8,014 USED CARS • LINEAR REGRESSION & RANDOM FOREST</div>
    </div>
    <div class="gear-badge">⚙️</div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# NAVIGATION TABS
# -----------------------------------------------------------------------------
tab_home, tab_predict, tab_analytics = st.tabs([
    "🏎️ HOME & 16-BRAND SHOWROOM",
    "🔧 INTERACTIVE ECU PRICE PREDICTOR",
    "📊 MAE / RMSE MODEL TELEMETRY",
])

# =============================================================================
# TAB 1: HOME PAGE WITH 16 CAR BRAND LOGOS & LIVE DYNO INSPECTOR
# =============================================================================
with tab_home:
    st.markdown('<div class="section-header">🏁 CARS24 MANUFACTURER TELEMETRY BAY (16 BRANDS)</div>', unsafe_allow_html=True)

    brand_counts = df["Brand"].value_counts().to_dict()
    brand_items = list(BRAND_CATALOG.items())

    for row_idx in range(0, len(brand_items), 8):
        cols = st.columns(8)
        for col, (b_name, b_info) in zip(cols, brand_items[row_idx : row_idx + 8]):
            count_val = brand_counts.get(b_name, 0)
            with col:
                st.markdown(
                    f"""
<div class="brand-card">
    <img src="{b_info['logo']}" class="brand-logo" alt="{b_name}">
    <div class="brand-name">{b_name}</div>
    <div class="brand-count">{count_val:,} Cars</div>
    <div class="brand-tier">{b_info['tier']}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

    # Top KPI Telemetry Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
<div class="glass-card">
    <div class="metric-title">CARS24 VERIFIED RECORDS</div>
    <div class="metric-value">{len(df):,}</div>
    <div class="metric-badge">16 Brands • {df['Car Name'].nunique()} Models</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
<div class="glass-card">
    <div class="metric-title">RANDOM FOREST R² SCORE</div>
    <div class="metric-value" style="color:#00F2FE;">{rf_metrics['R2']*100:.1f}%</div>
    <div class="metric-badge">Mode: {active_mode.upper()}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
<div class="glass-card">
    <div class="metric-title">RANDOM FOREST MAE</div>
    <div class="metric-value" style="color:#FF5500;">₹{rf_metrics['MAE']:,.0f}</div>
    <div class="metric-badge">Mean Absolute Error</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
<div class="glass-card">
    <div class="metric-title">RANDOM FOREST RMSE</div>
    <div class="metric-value">₹{rf_metrics['RMSE']:,.0f}</div>
    <div class="metric-badge">Root Mean Sq Error</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Interactive Brand Dyno Inspector
    st.markdown('<div class="section-header">🔍 INTERACTIVE MANUFACTURER DEPRECIATION DYNO</div>', unsafe_allow_html=True)
    d_col1, d_col2 = st.columns([1, 2.3])

    with d_col1:
        sel_brand = st.selectbox("Select Brand from Cars24 Dataset", options=list(BRAND_CATALOG.keys()), index=0)
        b_df = df[df["Brand"] == sel_brand]
        b_logo = BRAND_CATALOG[sel_brand]["logo"]
        b_tier = BRAND_CATALOG[sel_brand]["tier"]

        st.markdown(
            f"""
<div class="glass-card" style="text-align:center;">
    <img src="{b_logo}" style="width:68px; height:68px; margin-bottom:8px;">
    <div class="metric-value" style="color:#FF5500;">{sel_brand.upper()}</div>
    <div class="metric-badge">{b_tier}</div>
    <hr style="border-color:rgba(255,255,255,0.1); margin:12px 0;">
    <div class="metric-title">MEDIAN RESALE PRICE</div>
    <div class="metric-value" style="font-size:1.4rem; color:#00F2FE;">₹{b_df['Price'].median():,.0f}</div>
    <div style="font-size:0.82rem; color:#94A3B8; margin-top:6px;">
        Avg Mileage: {b_df['Distance'].mean():,.0f} km • Fleet Count: {len(b_df):,}
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with d_col2:
        age_grp = b_df.groupby("Age_of_Vehicle", as_index=False).agg(
            Mean_Price=("Price", "mean"),
            Median_Price=("Price", "median"),
            Count=("Price", "count"),
        )
        fig_dep = go.Figure()
        fig_dep.add_trace(go.Scatter(
            x=age_grp["Age_of_Vehicle"],
            y=age_grp["Mean_Price"],
            mode="lines+markers",
            name="Mean Resale Price",
            line=dict(color="#00F2FE", width=3),
            marker=dict(size=8),
        ))
        fig_dep.add_trace(go.Scatter(
            x=age_grp["Age_of_Vehicle"],
            y=age_grp["Median_Price"],
            mode="lines+markers",
            name="Median Resale Price",
            line=dict(color="#FF5500", width=2, dash="dash"),
        ))
        fig_dep.update_layout(
            title=f"{sel_brand} Real Cars24 Depreciation Curve (Age of Vehicle vs Resale Price)",
            xaxis_title="Age of Vehicle (Years from 2026)",
            yaxis_title="Resale Price (₹)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_dep, use_container_width=True)

# =============================================================================
# TAB 2: INTERACTIVE MECHANICAL PRICE PREDICTOR
# =============================================================================
with tab_predict:
    st.markdown('<div class="section-header">🔧 LIVE ECU RESALE VALUATION CONSOLE</div>', unsafe_allow_html=True)

    c_left, c_right = st.columns([1.15, 1.45])

    with c_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Vehicle Input Parameters")

        brand_options = sorted(df["Brand"].unique().tolist())
        in_brand = st.selectbox("1️⃣ Brand / Manufacturer", options=brand_options, index=brand_options.index("Maruti") if "Maruti" in brand_options else 0)

        filtered_Brand_df = df[df["Brand"] == in_brand]
        model_options = sorted(filtered_Brand_df["Car Name"].unique().tolist())
        in_model = st.selectbox("2️⃣ Exact Car Model (`Car Name`)", options=model_options, index=0)

        col_a, col_b = st.columns(2)
        with col_a:
            fuel_options = sorted(df["Fuel"].unique().tolist())
            in_fuel = st.selectbox("3️⃣ Engine / Fuel Type", options=fuel_options, index=fuel_options.index("Petrol") if "Petrol" in fuel_options else 0)
        with col_b:
            drive_options = sorted(df["Drive"].unique().tolist())
            in_drive = st.selectbox("4️⃣ Transmission (`Drive`)", options=drive_options, index=0)

        col_c, col_d = st.columns(2)
        with col_c:
            type_options = sorted(df["Type"].unique().tolist())
            default_type = filtered_Brand_df[filtered_Brand_df["Car Name"] == in_model]["Type"].mode().iloc[0]
            in_type = st.selectbox("5️⃣ Body Style (`Type`)", options=type_options, index=type_options.index(default_type) if default_type in type_options else 0)
        with col_d:
            in_owner = st.selectbox("6️⃣ Previous Owners (`Owner`)", options=[1, 2, 3, 4], index=0)

        in_age = st.slider("7️⃣ Age of Vehicle (Years)", min_value=1, max_value=18, value=7, step=1)
        in_distance = st.slider("8️⃣ Odometer Mileage (`Distance` in km)", min_value=1000, max_value=200000, value=45000, step=1000)

        active_algo = st.radio(
            "🧠 Select Valuation Algorithm",
            options=["Random Forest Regressor (Best Accuracy)", "Linear Regression", "Hybrid Ensemble Average"],
            horizontal=False,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Construct query DataFrame matching active_mode
    query_df = pd.DataFrame({
        "Brand": [in_brand],
        "Car Name": [in_model],
        "Distance": [in_distance],
        "Fuel": [in_fuel],
        "Age_of_Vehicle": [in_age],
        "Drive": [in_drive],
        "Type": [in_type],
        "Owner": [in_owner],
    })

    pred_lr = max(75000.0, float(ml_bundle["lr_model"].predict(query_df[ml_bundle["features"]])[0]))
    pred_rf = max(75000.0, float(ml_bundle["rf_model"].predict(query_df[ml_bundle["features"]])[0]))
    pred_ens = (pred_lr + pred_rf) / 2.0

    if "Random Forest" in active_algo:
        final_price = pred_rf
        algo_tag = "Random Forest Regressor"
    elif "Linear Regression" in active_algo:
        final_price = pred_lr
        algo_tag = "Linear Regression"
    else:
        final_price = pred_ens
        algo_tag = "Hybrid Ensemble"

    fair_low = final_price * 0.93
    fair_high = final_price * 1.07
    logo_url = BRAND_CATALOG.get(in_brand, {"logo": "https://cdn.simpleicons.org/suzuki/FF5500"})["logo"]

    with c_right:
        max_axis = max(1800000.0, final_price * 1.4)
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=final_price,
                number={"prefix": "₹", "valueformat": ",.0f", "font": {"family": "Orbitron", "size": 36, "color": "#00F2FE"}},
                delta={"reference": pred_lr, "increasing": {"color": "#00F2FE"}, "decreasing": {"color": "#FF5500"}},
                title={"text": f"<b>{in_model.upper()} ({in_fuel.upper()} • {in_drive.upper()})</b><br><span style='font-size:0.85em;color:#94A3B8'>{algo_tag}</span>", "font": {"family": "Rajdhani", "size": 20, "color": "#F8FAFC"}},
                gauge={
                    "axis": {"range": [0, max_axis], "tickcolor": "#94A3B8"},
                    "bar": {"color": "#FF5500", "thickness": 0.35},
                    "bgcolor": "rgba(15, 23, 42, 0.75)",
                    "borderwidth": 2,
                    "bordercolor": "rgba(255,255,255,0.15)",
                    "steps": [
                        {"range": [0, fair_low], "color": "rgba(56, 189, 248, 0.12)"},
                        {"range": [fair_low, fair_high], "color": "rgba(0, 242, 254, 0.28)"},
                        {"range": [fair_high, max_axis], "color": "rgba(255, 85, 0, 0.15)"},
                    ],
                    "threshold": {"line": {"color": "#00F2FE", "width": 4}, "thickness": 0.82, "value": final_price},
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F8FAFC", "family": "Rajdhani"},
            height=320,
            margin=dict(l=30, r=30, t=65, b=15),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(
                f"""
<div class="glass-card" style="border-left: 4px solid #00F2FE;">
    <div style="display:flex; align-items:center; gap:12px;">
        <img src="{logo_url}" style="width:42px; height:42px;">
        <div>
            <div class="metric-title">RANDOM FOREST VALUATION</div>
            <div class="metric-value" style="font-size:1.45rem; color:#00F2FE;">₹{pred_rf:,.0f}</div>
        </div>
    </div>
    <div class="metric-badge">₹{pred_rf/1e5:.2f} Lakhs • Non-Linear Tree Fit</div>
</div>
""",
                unsafe_allow_html=True,
            )
        with rc2:
            st.markdown(
                f"""
<div class="glass-card" style="border-left: 4px solid #FF5500;">
    <div class="metric-title">LINEAR REGRESSION VALUATION</div>
    <div class="metric-value" style="font-size:1.45rem; color:#FF5500;">₹{pred_lr:,.0f}</div>
    <div class="metric-badge" style="background:rgba(255,85,0,0.15); color:#FF5500; border-color:rgba(255,85,0,0.35);">
        Fair Deal Band: ₹{fair_low/1e5:.2f}L – ₹{fair_high/1e5:.2f}L
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        # Show Real Comparable Cars from cars_24_combined.csv
        st.markdown('<div class="metric-title" style="margin-top:8px;">📋 CLOSEST REAL LISTINGS IN CARS_24_COMBINED.CSV</div>', unsafe_allow_html=True)
        comps = df[df["Car Name"] == in_model].copy()
        if comps.empty:
            comps = df[df["Brand"] == in_brand].copy()
        comps["Age_Diff"] = (comps["Age_of_Vehicle"] - in_age).abs()
        comps["Km_Diff"] = (comps["Distance"] - in_distance).abs()
        comps = comps.sort_values(["Age_Diff", "Km_Diff"]).head(5)[
            ["Car Name", "Year", "Age_of_Vehicle", "Distance", "Fuel", "Drive", "Owner", "Location", "Price"]
        ]
        st.dataframe(comps, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3: MODEL EVALUATION TELEMETRY (MAE & RMSE)
# =============================================================================
with tab_analytics:
    st.markdown('<div class="section-header">📊 REGRESSION METRICS: LINEAR REGRESSION VS RANDOM FOREST</div>', unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(
            f"""
<div class="glass-card">
    <h3 style="font-family:'Orbitron',sans-serif; color:#FF5500; margin-top:0;">📐 Linear Regression</h3>
    <div style="display:flex; justify-content:space-between; margin-top:12px;">
        <div>
            <div class="metric-title">MAE</div>
            <div class="metric-value" style="font-size:1.35rem;">₹{lr_metrics['MAE']:,.0f}</div>
        </div>
        <div>
            <div class="metric-title">RMSE</div>
            <div class="metric-value" style="font-size:1.35rem;">₹{lr_metrics['RMSE']:,.0f}</div>
        </div>
        <div>
            <div class="metric-title">R² SCORE</div>
            <div class="metric-value" style="font-size:1.35rem; color:#FF5500;">{lr_metrics['R2']:.4f}</div>
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with mc2:
        st.markdown(
            f"""
<div class="glass-card">
    <h3 style="font-family:'Orbitron',sans-serif; color:#00F2FE; margin-top:0;">🌲 Random Forest Regressor</h3>
    <div style="display:flex; justify-content:space-between; margin-top:12px;">
        <div>
            <div class="metric-title">MAE</div>
            <div class="metric-value" style="font-size:1.35rem;">₹{rf_metrics['MAE']:,.0f}</div>
        </div>
        <div>
            <div class="metric-title">RMSE</div>
            <div class="metric-value" style="font-size:1.35rem;">₹{rf_metrics['RMSE']:,.0f}</div>
        </div>
        <div>
            <div class="metric-title">R² SCORE</div>
            <div class="metric-value" style="font-size:1.35rem; color:#00F2FE;">{rf_metrics['R2']:.4f}</div>
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    ch1, ch2 = st.columns(2)
    with ch1:
        err_df = pd.DataFrame({
            "Metric": ["MAE (Mean Absolute Error)", "RMSE (Root Mean Squared Error)"] * 2,
            "Algorithm": ["Linear Regression", "Linear Regression", "Random Forest Regressor", "Random Forest Regressor"],
            "Error (₹)": [lr_metrics["MAE"], lr_metrics["RMSE"], rf_metrics["MAE"], rf_metrics["RMSE"]],
        })
        fig_bar = px.bar(
            err_df,
            x="Metric",
            y="Error (₹)",
            color="Algorithm",
            barmode="group",
            color_discrete_map={"Linear Regression": "#FF5500", "Random Forest Regressor": "#00F2FE"},
            title="MAE & RMSE Error Telemetry (Lower is Better)",
            template="plotly_dark",
        )
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=360)
        st.plotly_chart(fig_bar, use_container_width=True)

    with ch2:
        y_true = ml_bundle["y_test"][:300]
        rf_p = rf_metrics["preds"][:300]
        lr_p = lr_metrics["preds"][:300]

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=y_true, y=rf_p, mode="markers", name="Random Forest", marker=dict(color="#00F2FE", size=7, opacity=0.75)))
        fig_sc.add_trace(go.Scatter(x=y_true, y=lr_p, mode="markers", name="Linear Regression", marker=dict(color="#FF5500", size=6, opacity=0.55, symbol="diamond")))
        max_v = float(max(y_true.max(), rf_p.max()))
        fig_sc.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines", name="Ideal 1:1 Fit", line=dict(color="#F8FAFC", dash="dot", width=2)))
        fig_sc.update_layout(
            title="Actual vs. Predicted Resale Prices (Cars24 Test Split)",
            xaxis_title="Actual Cars24 Price (₹)",
            yaxis_title="Predicted Price (₹)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
        )
        st.plotly_chart(fig_sc, use_container_width=True)