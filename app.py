# ============================================================
# SALES FORECASTING STREAMLIT APP
# Product-Based + Store-Based Forecast
# Uses: xgboost_sales_model_artifacts.pkl
# ============================================================

import calendar
from datetime import date

import holidays
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODERN UI DESIGN
# ============================================================


st.markdown(
    """
    <style>
    /* ---------- GLOBAL ---------- */
    :root {
        --navy: #123b75;
        --blue: #2563eb;
        --sky: #0ea5e9;
        --purple: #8b5cf6;
        --ink: #172033;
        --muted: #64748b;
        --line: #dce7f3;
        --page: #f6f9fd;
        --white: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", Arial, sans-serif;
    }

    .stApp {
        background: #f6f9fd;
        color: #172033;
    }

    .block-container {
        max-width: 1420px;
        padding-top: 1.1rem;
        padding-bottom: 3rem;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 10% 90%, rgba(120,75,220,.28), transparent 30%),
            linear-gradient(180deg, #0d4b8f 0%, #143c7a 55%, #273c8d 100%);
        border-right: 0;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.3rem;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    .brand {
        padding: 10px 4px 24px 4px;
    }

    .brand-title {
        font-size: 1.55rem;
        font-weight: 900;
        line-height: 1.05;
        margin-left: 4px;
    }

    .brand-sub {
        color: rgba(255,255,255,.72);
        font-size: .82rem;
        margin-top: 7px;
        margin-left: 4px;
    }

    .side-pill {
        background: rgba(255,255,255,.14);
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 14px;
        padding: 12px 14px;
        margin: 6px 0;
        font-weight: 750;
    }

    .side-item {
        padding: 11px 14px;
        margin: 5px 0;
        color: rgba(255,255,255,.88);
        font-weight: 650;
    }

    .side-bottom {
        margin-top: 80px;
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        background: rgba(5,35,92,.35);
        border: 1px solid rgba(255,255,255,.12);
        line-height: 1.4;
    }

    /* ---------- TOP HERO ---------- */
    .hero {
        background:
            linear-gradient(105deg, #2e72f4 0%, #4f66ee 50%, #9d4ff5 100%);
        color: white;
        padding: 27px 30px;
        border-radius: 0 0 22px 22px;
        margin: -18px -10px 22px -10px;
        box-shadow: 0 14px 35px rgba(72, 88, 220, .18);
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "↗";
        position: absolute;
        right: 60px;
        top: -28px;
        font-size: 9rem;
        font-weight: 900;
        color: rgba(255,255,255,.10);
        transform: rotate(-8deg);
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: -.03em;
    }

    .hero-subtitle {
        margin-top: 5px;
        font-size: 1rem;
        color: rgba(255,255,255,.90);
    }

    /* ---------- STEP / SELECTION CARDS ---------- */
    .step-card {
        border-radius: 16px;
        padding: 15px 17px 10px 17px;
        border: 1px solid #d7e5f4;
        min-height: 104px;
        margin-bottom: 5px;
    }

    .step-blue {
        background: linear-gradient(135deg, #edf6ff, #f5f9ff);
    }

    .step-purple {
        background: linear-gradient(135deg, #f7f1ff, #fbf8ff);
        border-color: #e6d9fb;
    }

    .step-green {
        background: linear-gradient(135deg, #ecfbf6, #f4fffb);
        border-color: #ccefe2;
    }

    .step-orange {
        background: linear-gradient(135deg, #fff5ed, #fffaf6);
        border-color: #f5decc;
    }

    .step-title {
        color: #172033;
        font-size: 1.05rem;
        font-weight: 850;
        margin-bottom: 3px;
    }

    .step-help {
        color: #64748b;
        font-size: .80rem;
    }

    /* ---------- BUSINESS PANEL ---------- */
    .business-head {
        margin-top: 14px;
        padding: 17px 20px;
        border: 1px solid #dce7f3;
        border-bottom: 0;
        border-radius: 17px 17px 0 0;
        background: linear-gradient(135deg, #eef6ff, #f7faff);
    }

    .business-title {
        color: #17315d;
        font-size: 1.1rem;
        font-weight: 850;
    }

    .business-sub {
        color: #64748b;
        font-size: .82rem;
        margin-top: 2px;
    }

    /* ---------- STREAMLIT WIDGETS ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: white;
        border: 1px solid #dfe9f4 !important;
        border-radius: 16px !important;
        box-shadow: 0 6px 22px rgba(31, 64, 105, .05);
    }

    label[data-testid="stWidgetLabel"] p,
    .stSelectbox label p,
    .stNumberInput label p,
    .stDateInput label p,
    .stRadio label p {
        color: #21314b !important;
        font-weight: 750 !important;
        font-size: .88rem !important;
    }

    div[data-baseweb="select"] > div {
        background: white !important;
        border: 1px solid #b8c9dc !important;
        color: #172033 !important;
        border-radius: 10px !important;
        min-height: 45px;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #172033 !important;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stDateInput"] input,
    .stTextInput input {
        background: white !important;
        color: #172033 !important;
        border: 1px solid #b8c9dc !important;
        border-radius: 10px !important;
        min-height: 45px;
    }

    div[data-testid="stNumberInput"] button {
        background: #f3f7fb !important;
        color: #172033 !important;
        border-color: #b8c9dc !important;
    }

    div[data-baseweb="popover"] {
        color: #172033 !important;
    }

    .stRadio > div {
        gap: 12px;
    }

    /* ---------- BUTTON ---------- */
    .stButton > button {
        width: 100%;
        min-height: 3.35rem;
        border: 0 !important;
        border-radius: 11px;
        background: linear-gradient(90deg, #1d75e8 0%, #3168ef 55%, #9a36ef 100%);
        color: white !important;
        font-weight: 900;
        font-size: 1.05rem;
        box-shadow: 0 10px 22px rgba(76, 72, 220, .22);
    }

    .stButton > button:hover {
        filter: brightness(1.04);
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 800;
        color: #2563eb !important;
        background: white !important;
        border: 1px solid #aac4e4 !important;
    }

    /* ---------- METRICS / RESULTS ---------- */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dfe9f4;
        padding: 14px 15px;
        border-radius: 14px;
        box-shadow: 0 5px 16px rgba(30, 60, 100, .05);
    }

    .result-card {
        background: white;
        border: 1px solid #dfe9f4;
        border-radius: 15px;
        padding: 17px;
        min-height: 108px;
        box-shadow: 0 6px 18px rgba(30, 60, 100, .06);
    }

    .result-label {
        color: #64748b;
        font-size: .76rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .06em;
    }

    .result-value {
        color: #17315d;
        font-size: 1.55rem;
        font-weight: 900;
        margin-top: 5px;
    }

    .result-help {
        color: #64748b;
        font-size: .80rem;
        margin-top: 3px;
    }

    h1, h2, h3, h4 {
        color: #172033;
    }

    hr {
        border-color: #dfe8f3;
    }

    /* ---------- Hide Streamlit chrome ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_model_artifacts():
    return joblib.load("xgboost_sales_model_artifacts.pkl")


try:
    artifacts = load_model_artifacts()
except FileNotFoundError:
    st.error(
        "❌ `xgboost_sales_model_artifacts.pkl` file not found. "
        "Keep the PKL file in the same folder as `app.py`."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    st.stop()


model = artifacts["xgb_model"]
label_encoders = artifacts.get("label_encoders", {})
model_features = artifacts["model_features"]
history = artifacts["historical_data"].copy()

history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
history = history.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


# ============================================================
# BASIC COLUMN HELPERS
# ============================================================

def first_existing(columns):
    for col in columns:
        if col in history.columns:
            return col
    return None


product_col = first_existing(["Product_Name", "Product_ID"])
store_col = first_existing(["Store_Location", "Store_ID"])

if product_col is None:
    st.error("❌ Product column not found in saved historical data.")
    st.stop()

if store_col is None:
    st.error("❌ Store column not found in saved historical data.")
    st.stop()


products = sorted(history[product_col].dropna().astype(str).unique().tolist())
stores = sorted(history[store_col].dropna().astype(str).unique().tolist())


# ============================================================
# FEATURE PREPARATION
# ============================================================

def apply_calendar_features(frame, forecast_date, holiday_override=None):
    d = pd.Timestamp(forecast_date)

    frame["Date"] = d
    frame["Year"] = d.year
    frame["Month"] = d.month_name()
    frame["Day"] = d.day
    frame["Day_of_Week"] = d.day_name()
    frame["DayOfYear"] = d.dayofyear
    frame["WeekOfYear"] = int(d.isocalendar().week)
    frame["Quarter"] = f"Q{d.quarter}"
    frame["Is_Weekend"] = int(d.dayofweek >= 5)
    frame["Month_Number"] = d.month
    frame["DayOfWeek_Number"] = d.dayofweek

    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    }
    frame["Season"] = season_map[d.month]

    india_holidays = holidays.country_holidays("IN", years=[d.year])
    india_holidays[pd.Timestamp(d.year, 1, 1).date()] = "New Year"
    detected_holiday = india_holidays.get(d.date())

    if holiday_override is None:
        is_holiday = bool(detected_holiday)
    else:
        is_holiday = bool(holiday_override)

    frame["Holiday_Flag"] = int(is_holiday)

    if is_holiday:
        frame["Holiday_Name"] = str(detected_holiday) if detected_holiday else "Holiday"
    else:
        frame["Holiday_Name"] = "No Holiday"

    return frame


def apply_interaction_features(frame):
    if "Price" in frame.columns and "Discount_Percentage" in frame.columns:
        price = pd.to_numeric(frame["Price"], errors="coerce").fillna(0)
        discount = pd.to_numeric(frame["Discount_Percentage"], errors="coerce").fillna(0)
        frame["Pricing_Effect"] = price * discount
        frame["Discounted_Price"] = price * (1 - discount / 100)

    if "Marketing_Spend" in frame.columns and "Promotion_Flag" in frame.columns:
        frame["Promotion_Effect"] = (
            pd.to_numeric(frame["Marketing_Spend"], errors="coerce").fillna(0)
            * pd.to_numeric(frame["Promotion_Flag"], errors="coerce").fillna(0)
        )

    if "Stock_Availability" in frame.columns and "Price" in frame.columns:
        frame["Inventory_Value"] = (
            pd.to_numeric(frame["Stock_Availability"], errors="coerce").fillna(0)
            * pd.to_numeric(frame["Price"], errors="coerce").fillna(0)
        )

    if "Promotion_Flag" in frame.columns and "Holiday_Flag" in frame.columns:
        frame["Seasonal_Effect"] = (
            pd.to_numeric(frame["Promotion_Flag"], errors="coerce").fillna(0)
            * pd.to_numeric(frame["Holiday_Flag"], errors="coerce").fillna(0)
        )

    if "Price" in frame.columns and "Competitor_Price" in frame.columns:
        frame["Price_Difference"] = (
            pd.to_numeric(frame["Price"], errors="coerce").fillna(0)
            - pd.to_numeric(frame["Competitor_Price"], errors="coerce").fillna(0)
        )

    return frame


def safe_encode(frame):
    encoded = frame.copy()

    for col, encoder in label_encoders.items():
        if col not in encoded.columns:
            continue

        values = encoded[col].astype(str)
        known = set(encoder.classes_)
        fallback = str(encoder.classes_[0])
        values = values.where(values.isin(known), fallback)
        encoded[col] = encoder.transform(values)

    return encoded


def prepare_input(frame):
    x = frame.copy()
    x = x.drop(
        columns=["Units_Sold", "Revenue", "Row_ID", "Date"],
        errors="ignore"
    )

    x = safe_encode(x)

    hist_defaults = history.copy()
    hist_defaults = hist_defaults.drop(
        columns=["Units_Sold", "Revenue", "Row_ID", "Date"],
        errors="ignore"
    )
    hist_defaults = safe_encode(hist_defaults)

    for col in model_features:
        if col not in x.columns:
            if col in hist_defaults.columns:
                default_value = pd.to_numeric(
                    hist_defaults[col], errors="coerce"
                ).median()

                if pd.isna(default_value):
                    mode = hist_defaults[col].mode()
                    default_value = mode.iloc[0] if not mode.empty else 0

                x[col] = default_value
            else:
                x[col] = 0

    x = x[model_features]
    x = x.replace([np.inf, -np.inf], np.nan)

    for col in x.columns:
        if x[col].isna().any():
            numeric = pd.to_numeric(x[col], errors="coerce")
            median = numeric.median()
            x[col] = numeric.fillna(0 if pd.isna(median) else median)

    return x


def create_forecast_dates(start_date, horizon):
    start = pd.Timestamp(start_date)

    if horizon == "Particular Day":
        return pd.DatetimeIndex([start])

    if horizon == "1 Week":
        return pd.date_range(start, periods=7, freq="D")

    if horizon == "1 Month":
        end = start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        return pd.date_range(start, end, freq="D")

    if horizon == "1 Year":
        end = start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        return pd.date_range(start, end, freq="D")

    return pd.DatetimeIndex([start])


def latest_row_for_product_store(product, store):
    temp = history[
        (history[product_col].astype(str) == str(product))
        & (history[store_col].astype(str) == str(store))
    ].copy()

    if temp.empty:
        temp = history[
            history[product_col].astype(str) == str(product)
        ].copy()

    if temp.empty:
        temp = history.tail(1).copy()

    return temp.sort_values("Date").tail(1).copy()


def latest_rows_for_store(store):
    temp = history[
        history[store_col].astype(str) == str(store)
    ].copy()

    if temp.empty:
        return history.tail(1).copy()

    entity_cols = [c for c in ["Product_ID", "Product_Name"] if c in temp.columns]

    if entity_cols:
        temp = (
            temp.sort_values("Date")
            .groupby(entity_cols, dropna=False, as_index=False)
            .tail(1)
        )
    else:
        temp = temp.sort_values("Date").tail(1)

    return temp.copy()


def forecast_rows(base_rows, dates, manual_values=None):
    output = []

    for d in dates:
        future = base_rows.copy()

        holiday_override = None
        if manual_values is not None:
            holiday_override = manual_values.get("Holiday_Flag")

        future = apply_calendar_features(
            future,
            d,
            holiday_override=holiday_override
        )

        if manual_values:
            for col, value in manual_values.items():
                if col == "Holiday_Flag":
                    continue
                if col in future.columns:
                    future[col] = value

        future = apply_interaction_features(future)
        x_future = prepare_input(future)

        pred = model.predict(x_future)
        pred = np.maximum(pred, 0)

        output.append(
            {
                "Date": pd.Timestamp(d),
                "Day": pd.Timestamp(d).day_name(),
                "Week": ((pd.Timestamp(d).dayofyear - 1) // 7) + 1,
                "Month": pd.Timestamp(d).month_name(),
                "Year": pd.Timestamp(d).year,
                "Forecast_Units": round(float(np.sum(pred)), 2)
            }
        )

    return pd.DataFrame(output)


def numeric_default(row, column, fallback=0.0):
    if column not in row.columns:
        return float(fallback)

    val = pd.to_numeric(row[column], errors="coerce").iloc[0]
    if pd.isna(val):
        return float(fallback)

    return float(val)


def show_results(result_df, title):
    total = result_df["Forecast_Units"].sum()
    avg = result_df["Forecast_Units"].mean()
    max_row = result_df.loc[result_df["Forecast_Units"].idxmax()]
    min_row = result_df.loc[result_df["Forecast_Units"].idxmin()]

    st.markdown(f"### 📊 {title}")

    a, b, c, d = st.columns(4)

    with a:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Total Forecast</div>
                <div class="result-value">{total:,.0f}</div>
                <div class="result-help">units</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Daily Average</div>
                <div class="result-value">{avg:,.1f}</div>
                <div class="result-help">units per day</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Highest Day</div>
                <div class="result-value">{max_row['Forecast_Units']:,.0f}</div>
                <div class="result-help">{max_row['Date'].date()}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with d:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Lowest Day</div>
                <div class="result-value">{min_row['Forecast_Units']:,.0f}</div>
                <div class="result-help">{min_row['Date'].date()}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### 📈 Forecast Trend")
    st.line_chart(
        result_df.set_index("Date")[["Forecast_Units"]],
        height=360
    )

    if len(result_df) > 7:
        monthly = (
            result_df.assign(
                Period=result_df["Date"].dt.strftime("%b %Y")
            )
            .groupby("Period", sort=False, as_index=False)["Forecast_Units"]
            .sum()
        )

        st.markdown("#### 🗓️ Monthly Summary")
        st.bar_chart(
            monthly.set_index("Period")[["Forecast_Units"]],
            height=300
        )

    display = result_df.copy()
    display["Date"] = display["Date"].dt.date
    display = display.rename(
        columns={"Forecast_Units": "Forecast Units"}
    )

    st.markdown("#### 📋 Detailed Forecast")
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Forecast CSV",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="sales_forecast.csv",
        mime="text/csv"
    )


# ============================================================
# DASHBOARD SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-title">📈 Sales<br>Forecasting</div>
            <div class="brand-sub">AI-powered future sales planning</div>
        </div>

        <div class="side-pill">🏠 &nbsp; Forecast</div>
        <div class="side-item">📊 &nbsp; Results</div>
        <div class="side-item">🗄️ &nbsp; Data Insights</div>
        <div class="side-item">ℹ️ &nbsp; About</div>

        <div class="side-bottom">
            📊<br>
            <b>Data Driven</b><br>
            Better Decisions
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Sales Forecasting</div>
        <div class="hero-subtitle">Predict Future Sales with AI</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FORECAST MODE
# ============================================================

forecast_type = st.radio(
    "Forecast Mode",
    ["📦 Product Forecast", "🏪 Store Forecast"],
    horizontal=True,
    key="forecast_mode"
)


# ============================================================
# TOP SELECTION CARDS
# ============================================================

top1, top2 = st.columns(2, gap="medium")

with top1:
    st.markdown(
        """
        <div class="step-card step-blue">
            <div class="step-title">📅 &nbsp; 1. Select Date</div>
            <div class="step-help">Choose the date to start forecasting</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    selected_date = st.date_input(
        "Forecast Date",
        value=date.today(),
        key="dashboard_date",
        label_visibility="collapsed"
    )

with top2:
    st.markdown(
        """
        <div class="step-card step-purple">
            <div class="step-title">🎯 &nbsp; 2. Forecast Horizon</div>
            <div class="step-help">Select forecasting period</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    selected_horizon = st.selectbox(
        "Forecast Horizon",
        ["Particular Day", "1 Week", "1 Month", "1 Year"],
        key="dashboard_horizon",
        label_visibility="collapsed"
    )


# ============================================================
# PRODUCT FORECAST
# ============================================================

if forecast_type == "📦 Product Forecast":

    second1, second2 = st.columns(2, gap="medium")

    with second1:
        st.markdown(
            """
            <div class="step-card step-green">
                <div class="step-title">📦 &nbsp; 3. Product</div>
                <div class="step-help">Select product to forecast</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        selected_product = st.selectbox(
            "Product",
            products,
            key="dashboard_product",
            label_visibility="collapsed"
        )

    available_stores = sorted(
        history[
            history[product_col].astype(str) == str(selected_product)
        ][store_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not available_stores:
        available_stores = stores

    with second2:
        st.markdown(
            """
            <div class="step-card step-orange">
                <div class="step-title">🏪 &nbsp; 4. Store</div>
                <div class="step-help">Select store location</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        selected_store = st.selectbox(
            "Store",
            available_stores,
            key="dashboard_store",
            label_visibility="collapsed"
        )

    base_row = latest_row_for_product_store(
        selected_product,
        selected_store
    )

    st.markdown(
        """
        <div class="business-head">
            <div class="business-title">⚙️ &nbsp; 5. Business Inputs</div>
            <div class="business-sub">Enter expected future business conditions</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True):
        r1 = st.columns(3, gap="medium")

        with r1[0]:
            price = st.number_input(
                "💰 Price (₹)",
                min_value=0.0,
                value=numeric_default(base_row, "Price", 0.0),
                step=1.0,
                key="dashboard_price"
            )

        with r1[1]:
            discount = st.number_input(
                "🏷️ Discount %",
                min_value=0.0,
                max_value=100.0,
                value=numeric_default(base_row, "Discount_Percentage", 0.0),
                step=1.0,
                key="dashboard_discount"
            )

        with r1[2]:
            promotion = st.selectbox(
                "📣 Promotion",
                ["No", "Yes"],
                index=1 if numeric_default(base_row, "Promotion_Flag", 0) >= 1 else 0,
                key="dashboard_promotion"
            )

        r2 = st.columns(3, gap="medium")

        with r2[0]:
            stock = st.number_input(
                "📦 Stock Availability",
                min_value=0.0,
                value=numeric_default(base_row, "Stock_Availability", 0.0),
                step=1.0,
                key="dashboard_stock"
            )

        with r2[1]:
            holiday_option = st.selectbox(
                "📅 Holiday",
                ["Auto Detect", "No", "Yes"],
                key="dashboard_holiday"
            )

        with r2[2]:
            local_event = st.selectbox(
                "📍 Local Event",
                ["No", "Yes"],
                index=1 if numeric_default(base_row, "Local_Event_Flag", 0) >= 1 else 0,
                key="dashboard_local_event"
            )

        r3 = st.columns(3, gap="medium")

        with r3[0]:
            competitor_price = st.number_input(
                "🪙 Competitor Price (₹)",
                min_value=0.0,
                value=numeric_default(base_row, "Competitor_Price", 0.0),
                step=1.0,
                key="dashboard_competitor"
            )

        with r3[1]:
            economic_indicator = st.number_input(
                "📊 Economic Indicator",
                value=numeric_default(base_row, "Economic_Indicator", 0.0),
                step=0.1,
                key="dashboard_economic"
            )

        with r3[2]:
            marketing_spend = st.number_input(
                "📈 Marketing Spend (₹)",
                min_value=0.0,
                value=numeric_default(base_row, "Marketing_Spend", 0.0),
                step=100.0,
                key="dashboard_marketing"
            )

        holiday_override = None
        if holiday_option == "Yes":
            holiday_override = 1
        elif holiday_option == "No":
            holiday_override = 0

        manual_values = {
            "Price": price,
            "Discount_Percentage": discount,
            "Promotion_Flag": 1 if promotion == "Yes" else 0,
            "Stock_Availability": stock,
            "Local_Event_Flag": 1 if local_event == "Yes" else 0,
            "Competitor_Price": competitor_price,
            "Economic_Indicator": economic_indicator,
            "Marketing_Spend": marketing_spend,
            "Holiday_Flag": holiday_override
        }

        if st.button(
            "🚀  Generate Forecast",
            key="dashboard_product_button",
            use_container_width=True
        ):
            forecast_date_range = create_forecast_dates(
                selected_date,
                selected_horizon
            )

            with st.spinner("Generating product forecast..."):
                result = forecast_rows(
                    base_row,
                    forecast_date_range,
                    manual_values
                )

            st.session_state["dashboard_result"] = result
            st.session_state["dashboard_title"] = (
                f"{selected_product} · {selected_store} Forecast"
            )


# ============================================================
# STORE FORECAST
# ============================================================

else:

    st.markdown(
        """
        <div class="step-card step-orange">
            <div class="step-title">🏪 &nbsp; 3. Select Store</div>
            <div class="step-help">Forecast total future sales for all products in a store</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_store = st.selectbox(
        "Store",
        stores,
        key="dashboard_store_only",
        label_visibility="collapsed"
    )

    store_rows = latest_rows_for_store(selected_store)

    st.markdown(
        """
        <div class="business-head">
            <div class="business-title">📊 &nbsp; Store Overview</div>
            <div class="business-sub">The model uses the latest saved profile of each product in this store</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True):
        s1, s2, s3 = st.columns(3)

        with s1:
            st.metric("Products / Profiles", f"{len(store_rows):,}")

        with s2:
            latest_sales = 0
            if "Units_Sold" in store_rows.columns:
                latest_sales = pd.to_numeric(
                    store_rows["Units_Sold"],
                    errors="coerce"
                ).fillna(0).sum()
            st.metric("Latest Sales", f"{latest_sales:,.0f}")

        with s3:
            latest_store_date = store_rows["Date"].max()
            st.metric("Latest Store Data", str(latest_store_date.date()))

        if st.button(
            "🚀  Generate Store Forecast",
            key="dashboard_store_button",
            use_container_width=True
        ):
            forecast_date_range = create_forecast_dates(
                selected_date,
                selected_horizon
            )

            with st.spinner("Generating store forecast..."):
                result = forecast_rows(
                    store_rows,
                    forecast_date_range,
                    manual_values=None
                )

            st.session_state["dashboard_result"] = result
            st.session_state["dashboard_title"] = (
                f"{selected_store} Store Forecast"
            )


# ============================================================
# RESULTS
# ============================================================

if "dashboard_result" in st.session_state:
    st.markdown("---")
    show_results(
        st.session_state["dashboard_result"],
        st.session_state.get(
            "dashboard_title",
            "Future Sales Forecast"
        )
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Product forecast uses your selected future Price, Discount, Promotion, Stock, "
    "Holiday, Local Event, Competitor Price, Economic Indicator and Marketing Spend. "
    "Store forecast uses the latest saved profile for each product in that store."
)
