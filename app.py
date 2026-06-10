import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARDA Crop Yield Forecasting System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  –  ARDA green palette, glassmorphism cards, dashboard look
# ─────────────────────────────────────────────────────────────────────────────
ARDA_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root tokens ── */
:root {
    --arda-green:       #1a6b2e;
    --arda-mid:         #2d8a47;
    --arda-light:       #4caf6e;
    --arda-accent:      #8bc34a;
    --arda-dark:        #0d3d1a;
    --glass-bg:         rgba(255,255,255,0.10);
    --glass-border:     rgba(255,255,255,0.22);
    --card-shadow:      0 8px 32px rgba(0,0,0,0.28);
    --text-primary:     #ffffff;
    --text-secondary:   #d4edda;
    --text-muted:       #a8d5b5;
}

/* ── Full-page background  ── */
.stApp {
    background:
        linear-gradient(160deg, rgba(13,61,26,0.92) 0%, rgba(26,107,46,0.85) 50%, rgba(45,138,71,0.80) 100%),
        url('https://images.unsplash.com/photo-1625246333195-78d9c38ad449?w=1800&q=80')
        center/cover no-repeat fixed;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

/* ── Remove default white main-block padding strip ── */
[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--arda-dark) 0%, var(--arda-green) 100%);
    border-right: 1px solid var(--glass-border);
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Glassmorphism card helper ── */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: var(--card-shadow);
    margin-bottom: 20px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.38);
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(13,61,26,0.75) 0%, rgba(76,175,110,0.45) 100%);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 42px 48px;
    margin-bottom: 32px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: var(--card-shadow);
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
    margin: 0 0 10px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin: 0 0 6px 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(139,195,74,0.25);
    border: 1px solid var(--arda-accent);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: var(--arda-accent);
    font-weight: 600;
    margin-top: 12px;
    letter-spacing: 0.5px;
}

/* ── Section heading ── */
.section-heading {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--arda-accent);
    border-left: 4px solid var(--arda-accent);
    padding-left: 12px;
    margin: 28px 0 18px 0;
    letter-spacing: 0.2px;
}

/* ── Metric row ── */
.metric-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: rgba(26,107,46,0.35);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(139,195,74,0.35);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    transition: all 0.25s ease;
}
.metric-card:hover {
    background: rgba(26,107,46,0.55);
    border-color: var(--arda-accent);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
    font-weight: 600;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
}
.metric-unit {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ── Yield result card ── */
.yield-result-card {
    background: linear-gradient(135deg, rgba(26,107,46,0.70) 0%, rgba(45,138,71,0.55) 100%);
    border: 2px solid var(--arda-accent);
    border-radius: 20px;
    padding: 32px 36px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 0 32px rgba(139,195,74,0.25);
}
.yield-label {
    font-size: 0.85rem;
    color: var(--arda-accent);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
    margin-bottom: 10px;
}
.yield-value {
    font-size: 3.2rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 6px;
}
.yield-unit {
    font-size: 1rem;
    color: var(--text-secondary);
    font-weight: 400;
}

/* ── Info / warning boxes ── */
.info-box {
    background: rgba(139,195,74,0.12);
    border: 1px solid rgba(139,195,74,0.40);
    border-left: 4px solid var(--arda-accent);
    border-radius: 10px;
    padding: 14px 18px;
    color: var(--text-secondary);
    font-size: 0.88rem;
    line-height: 1.6;
    margin-bottom: 12px;
}
.warning-box {
    background: rgba(255,152,0,0.10);
    border: 1px solid rgba(255,152,0,0.35);
    border-left: 4px solid #ff9800;
    border-radius: 10px;
    padding: 14px 18px;
    color: #ffe0b2;
    font-size: 0.88rem;
    line-height: 1.6;
    margin-bottom: 12px;
}
.note-box {
    background: rgba(100,181,246,0.10);
    border: 1px solid rgba(100,181,246,0.30);
    border-left: 4px solid #64b5f6;
    border-radius: 10px;
    padding: 14px 18px;
    color: #e3f2fd;
    font-size: 0.88rem;
    line-height: 1.6;
    margin-bottom: 12px;
}

/* ── Limitation list ── */
.limit-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    font-size: 0.87rem;
    color: var(--text-secondary);
    line-height: 1.5;
}
.limit-icon {
    color: #ff9800;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
}

/* ── Future improvements list ── */
.improve-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 7px 0;
    font-size: 0.87rem;
    color: var(--text-secondary);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    line-height: 1.5;
}
.improve-icon {
    color: var(--arda-accent);
    font-weight: 700;
    flex-shrink: 0;
}

/* ── Sidebar branding ── */
.sidebar-brand {
    padding: 18px 16px 10px;
    border-bottom: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 16px;
}
.sidebar-org {
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.5px;
}
.sidebar-tagline {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 3px;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.15);
    margin: 12px 0;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div > div,
.stTextInput > div > div {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
/* Predict button */
div[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, var(--arda-mid), var(--arda-accent)) !important;
    color: #0d3d1a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.5px;
}
div[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(139,195,74,0.45) !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px) !important;
}
div[data-testid="stExpander"] summary {
    color: var(--arda-accent) !important;
    font-weight: 600 !important;
}

/* ── Matplotlib / pyplot transparent ── */
.stPlotlyChart, .stPyplot { background: transparent !important; }
figure, .stPyplot > div { background: transparent !important; }

/* ── Tab style ── */
button[data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--arda-accent) !important;
    border-bottom-color: var(--arda-accent) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); }
::-webkit-scrollbar-thumb { background: var(--arda-mid); border-radius: 3px; }
</style>
"""
st.markdown(ARDA_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/trained_models/best_crop_yield_model.pkl")
    selected_features = joblib.load("data/featured/selected_features.pkl")
    return model, selected_features


@st.cache_data
def load_performance_metrics():
    X_test = pd.read_csv("data/featured/X_test_featured.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    model, selected_features = load_model()

    X_test_encoded = pd.get_dummies(X_test, columns=["Area", "Item"], drop_first=True)
    X_test_encoded = X_test_encoded.reindex(columns=selected_features, fill_value=0)
    predictions = model.predict(X_test_encoded)

    return {
        "model_name": type(model).__name__,
        "r2": r2_score(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "mae": mean_absolute_error(y_test, predictions),
        "mape": mean_absolute_percentage_error(y_test, predictions) * 100,
    }


@st.cache_data
def load_data():
    return pd.read_csv("data/featured/featured_crop_data.csv")


model, selected_features = load_model()
metrics = load_performance_metrics()
df = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(input_data: pd.DataFrame) -> pd.DataFrame:
    engineered = input_data.copy()
    engineered["rain_temp_interaction"] = (
        engineered["average_rain_fall_mm_per_year"] * engineered["avg_temp"]
    )
    engineered["pesticides_tonnes_log"] = np.log1p(engineered["pesticides_tonnes"])
    return engineered


def prepare_features(input_data: pd.DataFrame, selected_features: list) -> pd.DataFrame:
    encoded = pd.get_dummies(input_data, columns=["Area", "Item"], drop_first=True)
    for col in selected_features:
        if col not in encoded.columns:
            encoded[col] = 0
    return encoded[selected_features]


def styled_chart_theme(ax, fig, title=""):
    """Apply ARDA dark-transparent theme to matplotlib axes."""
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.tick_params(colors="#a8d5b5", labelsize=9)
    ax.xaxis.label.set_color("#a8d5b5")
    ax.yaxis.label.set_color("#a8d5b5")
    ax.title.set_color("#8bc34a")
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_edgecolor((1.0, 1.0, 1.0, 0.15))
    if title:
        ax.set_title(title)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Branding block
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-org">ARDA Zimbabwe</div>
        <div class="sidebar-tagline">Agricultural & Rural Development Authority</div>
        <div class="sidebar-tagline" style="margin-top:6px;color:#8bc34a;font-weight:600;">
            Crop Yield Forecasting System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown("**Forecast Inputs**", unsafe_allow_html=False)
    st.markdown("<p style='font-size:0.78rem;color:#a8d5b5;margin-bottom:10px;'>Configure the conditions below, then click Predict Yield.</p>", unsafe_allow_html=True)

    area = st.selectbox("Country / Region", sorted(df["Area"].unique()))
    crop = st.selectbox("Crop Type", sorted(df["Item"].unique()))
    year = st.number_input("Year", min_value=1990, max_value=2035, value=2025)
    rainfall = st.number_input("Rainfall (mm/year)", min_value=200.0,  value=1200.0)
    pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, value=5000.0)
    temperature = st.number_input("Average Temperature (°C)", value=25.0)

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict Yield", use_container_width=True)

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:0.72rem;color:#a8d5b5;line-height:1.6;'>
    Model: Bagging Regressor<br>
    Validation MAPE: 9.46%<br>
    Developed for ARDA Zimbabwe
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">ARDA Crop Yield Forecasting System</div>
    <div class="hero-subtitle">
        AI-powered predictive analytics for agricultural planning and production forecasting
    </div>
    <div class="hero-subtitle" style="margin-top:4px;font-size:0.88rem;color:#a8d5b5;">
        Agricultural &amp; Rural Development Authority &nbsp;|&nbsp; Zimbabwe
    </div>
    <span class="hero-badge">Machine Learning Prototype &nbsp;–&nbsp; Bagging Regressor Ensemble</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST RESULT  (shown when user clicks Predict)
# ─────────────────────────────────────────────────────────────────────────────
if predict_clicked:
    input_data = pd.DataFrame({
        "Area": [area],
        "Item": [crop],
        "Year": [year],
        "average_rain_fall_mm_per_year": [rainfall],
        "pesticides_tonnes": [pesticides],
        "avg_temp": [temperature],
    })
    engineered_input = engineer_features(input_data)
    input_encoded = prepare_features(engineered_input, selected_features)
    prediction = model.predict(input_encoded)[0]

    lower = prediction * (1 - 0.0946)
    upper = prediction * (1 + 0.0946)

    # ── Yield result panel ─────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Yield Forecast Result</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="yield-result-card">
        <div class="yield-label">Predicted Crop Yield</div>
        <div class="yield-value">{prediction:,.0f}</div>
        <div class="yield-unit">hg / ha &nbsp; (hectograms per hectare)</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence band
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Lower Bound (−9.46%)</div>
            <div class="metric-value">{lower:,.0f}</div>
            <div class="metric-unit">hg/ha</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-card" style="border-color:var(--arda-accent);background:rgba(139,195,74,0.18);">
            <div class="metric-label">Predicted Yield</div>
            <div class="metric-value">{prediction:,.0f}</div>
            <div class="metric-unit">hg/ha</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Upper Bound (+9.46%)</div>
            <div class="metric-value">{upper:,.0f}</div>
            <div class="metric-unit">hg/ha</div>
        </div>""", unsafe_allow_html=True)

    # ── Model information ──────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Model Information</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <p style="margin:0 0 8px 0;"><strong style="color:#8bc34a;">Prediction Model:</strong>&nbsp; Saved Bagging Regressor Model</p>
        <p style="margin:0;font-size:0.87rem;color:#d4edda;line-height:1.6;">
            This forecast is generated using the trained and saved Bagging Regressor ensemble model developed
            from historical ARDA agricultural production data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Reliability ────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Reliability / Accuracy</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Expected Validation Error Band: ±9.46% MAPE</strong><br>
        This means the actual observed yield may vary by approximately 9.46% above or below the predicted value
        under conditions similar to the training data.
    </div>
    <div class="note-box">
        <strong>Confidence Interpretation:</strong> The band shown above ({lower:,.0f} – {upper:,.0f} hg/ha)
        reflects the model's empirical accuracy on the held-out test set. Predictions within these bounds are
        consistent with the model's historical performance.
    </div>
    """.format(lower=lower, upper=upper), unsafe_allow_html=True)

    # ── Limitations ────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Important Forecast Limitations</div>', unsafe_allow_html=True)

    limitations = [
        "Predictions are limited to patterns learned from the training dataset.",
        "The model may not generalise well to highly unusual environmental or operational conditions.",
        "Forecast quality depends heavily on input data quality.",
        "Forecast accuracy assumes feature consistency with training data.",
        "Unexpected climate events may reduce prediction reliability.",
    ]
    items_html = "".join(
        f'<div class="limit-item"><span class="limit-icon">!</span><span>{lim}</span></div>'
        for lim in limitations
    )
    st.markdown(f'<div class="glass-card">{items_html}</div>', unsafe_allow_html=True)

    # ── Interpretation note ────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Forecast Interpretation Note</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="note-box">
        This forecast should support planning and decision-making, but should be interpreted alongside
        agronomic expertise, seasonal field observations, and current environmental conditions.
        It is not a substitute for on-ground agronomic assessment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.12);margin:28px 0;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERFORMANCE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Model Performance Dashboard</div>', unsafe_allow_html=True)

m_r2   = f"{metrics['r2']:.4f}"
m_rmse = f"{metrics['rmse']:,.0f}"
m_mae  = f"{metrics['mae']:,.0f}"
m_mape = f"{metrics['mape']:.2f}%"
m_name = metrics['model_name']

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-label">Model</div>
        <div class="metric-value" style="font-size:0.95rem;">{m_name}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">R² Score</div>
        <div class="metric-value">{m_r2}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">RMSE</div>
        <div class="metric-value">{m_rmse}</div>
        <div class="metric-unit">hg/ha</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">MAE</div>
        <div class="metric-value">{m_mae}</div>
        <div class="metric-unit">hg/ha</div>
    </div>
    <div class="metric-card" style="border-color:rgba(139,195,74,0.55);">
        <div class="metric-label">MAPE</div>
        <div class="metric-value" style="color:#8bc34a;">{m_mape}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# AGRICULTURAL INSIGHTS  (tabbed)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Agricultural Insights</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Yield Distribution", "Yearly Trends", "Crop Comparison"])

PLOT_COLORS = {
    "hist_face": "#4caf6e",
    "hist_edge": "#1a6b2e",
    "line":      "#8bc34a",
    "bar_cmap":  "YlGn",
    "kde":       "#a8d5b5",
}

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            df["hg/ha_yield"], kde=False, ax=ax,
            color=PLOT_COLORS["hist_face"],
            edgecolor=PLOT_COLORS["hist_edge"],
            alpha=0.75,
        )
        # Draw KDE separately for full version compatibility
        sns.kdeplot(
            df["hg/ha_yield"], ax=ax,
            color=PLOT_COLORS["kde"], lw=2,
        )
        # Scale KDE line to match histogram counts
        ax.lines[0].set_ydata(
            ax.lines[0].get_ydata() * len(df["hg/ha_yield"])
            * (df["hg/ha_yield"].max() - df["hg/ha_yield"].min()) / 30
        )
        styled_chart_theme(ax, fig, "Yield Distribution (hg/ha)")
        ax.set_xlabel("hg/ha")
        ax.set_ylabel("Count")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        fig.tight_layout(pad=1.0)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        top_crops = df.groupby("Item")["hg/ha_yield"].median().nlargest(8)
        top_crops.plot(kind="barh", ax=ax, color=PLOT_COLORS["hist_face"], edgecolor=PLOT_COLORS["hist_edge"])
        styled_chart_theme(ax, fig, "Top 8 Crops by Median Yield")
        ax.set_xlabel("Median hg/ha")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 4))
    yearly = df.groupby("Year")["hg/ha_yield"].mean()
    ax.plot(yearly.index, yearly.values, color=PLOT_COLORS["line"], lw=2.5, marker="o", markersize=4)
    ax.fill_between(yearly.index, yearly.values, alpha=0.18, color=PLOT_COLORS["hist_face"])
    styled_chart_theme(ax, fig, "Average Crop Yield Trend (All Crops)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Yield (hg/ha)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(10, 5))
    crop_means = (
        df.groupby("Item")["hg/ha_yield"].mean()
        .sort_values(ascending=False)
        .head(15)
    )
    colors = plt.cm.YlGn(np.linspace(0.35, 0.85, len(crop_means)))
    ax.bar(crop_means.index, crop_means.values, color=colors, edgecolor=PLOT_COLORS["hist_edge"])
    styled_chart_theme(ax, fig, "Mean Yield by Crop Type (Top 15)")
    ax.set_ylabel("Mean Yield (hg/ha)")
    plt.xticks(rotation=40, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FUTURE IMPROVEMENTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Future Improvements</div>', unsafe_allow_html=True)

with st.expander("View Planned Enhancements", expanded=False):
    improvements = [
        "Collect ARDA farm-level data.",
        "Add irrigation availability as a predictive feature.",
        "Incorporate soil quality data.",
        "Include planting and harvest dates.",
        "Use seasonal rainfall instead of annual rainfall averages.",
        "Add fertilizer type and seed variety information.",
        "Replace Country variable with Zimbabwe-specific location variables.",
        "Add crop-specific rainfall thresholds and early-warning indicators.",
        "Validate using Zimbabwe-only and future-year test sets.",
        "Deploy the model as a Streamlit decision-support dashboard for ARDA officers.",
    ]
    items_html = "".join(
        f'<div class="improve-item"><span class="improve-icon">+</span><span>{item}</span></div>'
        for item in improvements
    )
    st.markdown(f'<div style="padding:8px 4px;">{items_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:rgba(255,255,255,0.10);margin:36px 0 20px 0;'>
<div style='text-align:center;'>
    <p style='font-size:0.80rem;color:#a8d5b5;margin:0;line-height:1.8;'>
        ARDA Crop Yield Forecasting System &nbsp;|&nbsp; Machine Learning Prototype<br>
        Developed for the Agricultural &amp; Rural Development Authority &nbsp;|&nbsp; Zimbabwe<br>
        <span style='color:rgba(255,255,255,0.35);font-size:0.72rem;'>
            Predictions are for planning support only and do not replace agronomic field assessment.
        </span>
    </p>
</div>
""", unsafe_allow_html=True)
