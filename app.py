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
    initial_sidebar_state="collapsed",
)


# ============================================================
# MODERN UI DESIGN
# ============================================================


st.markdown(
    """
    <style>
    .stApp {
        background: #f5f8fc;
        color: #1f2937;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f4c81, #1f6fb2);
        color: white;
        padding: 28px 32px;
        border-radius: 20px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(15, 76, 129, 0.18);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .hero p {
        margin: 8px 0 0 0;
        color: rgba(255,255,255,0.92);
        font-size: 0.98rem;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px;
        min-height: 105px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    }

    .info-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .info-value {
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 6px;
    }

    .info-help {
        color: #64748b;
        font-size: 0.82rem;
        margin-top: 4px;
    }

    .section {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px 20px;
        margin-top: 14px;
        margin-bottom: 16px;
        box-shadow: 0 5px 16px rgba(15, 23, 42, 0.05);
    }

    .section-title {
        color: #0f172a;
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 3px;
    }

    .section-sub {
        color: #64748b;
        font-size: 0.88rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 14px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(15,23,42,0.04);
    }

    .stButton > button {
        width: 100%;
        min-height: 3.05rem;
        border: 0;
        border-radius: 10px;
        background: #1769aa;
        color: white;
        font-weight: 800;
        font-size: 0.98rem;
        box-shadow: 0 5px 14px rgba(23, 105, 170, 0.18);
    }

    .stButton > button:hover {
        background: #0f5a96;
        color: white;
        border: 0;
    }

    .stDownloadButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid #1769aa;
        color: #1769aa;
        background: white;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stDateInput"] input {
        background: #ffffff !important;
        border-radius: 10px !important;
    }

    div[data-testid="stTabs"] button {
        font-weight: 800;
        font-size: 1rem;
        color: #334155;
    }

    h1, h2, h3, h4 {
        color: #0f172a;
    }

    .stCaption, small {
        color: #64748b !important;
    }
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
                "Week": ((pd.Timestamp(d).dayofyear - 1) // 7) + 1,,
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
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <div class="main-title">🔮 Sales Forecasting</div>
        <div class="main-subtitle">
            Product-level and store-level forecasting using your trained XGBoost model.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# COMBINED FORECAST DASHBOARD
# ============================================================

st.markdown(
    """
    <div class="section">
        <div class="section-title">🔮 Combined Future Sales Forecast</div>
        <div class="section-sub">
            Product forecast and store forecast are available in one single form.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

forecast_type = st.radio(
    "🎯 Forecast Type",
    ["📦 Product Based", "🏪 Store Based"],
    horizontal=True,
    key="forecast_type"
)

common1, common2 = st.columns(2)

with common1:
    selected_date = st.date_input(
        "📅 Forecast Date",
        value=date.today(),
        key="combined_date"
    )

with common2:
    selected_horizon = st.selectbox(
        "🔮 Forecast Horizon",
        ["Particular Day", "1 Week", "1 Month", "1 Year"],
        key="combined_horizon"
    )


# ============================================================
# PRODUCT BASED
# ============================================================

if forecast_type == "📦 Product Based":

    st.markdown(
        """
        <div class="section">
            <div class="section-title">📦 Product + Store Inputs</div>
            <div class="section-sub">
                Select one product and one store, then enter expected future business conditions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        selected_product = st.selectbox(
            "📦 Product",
            products,
            key="combined_product"
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

    with c2:
        selected_store = st.selectbox(
            "🏪 Store",
            available_stores,
            key="combined_store_for_product"
        )

    base_row = latest_row_for_product_store(
        selected_product,
        selected_store
    )

    r1 = st.columns(3)

    with r1[0]:
        price = st.number_input(
            "💰 Price",
            min_value=0.0,
            value=numeric_default(base_row, "Price", 0.0),
            step=1.0
        )

    with r1[1]:
        discount = st.number_input(
            "🏷️ Discount %",
            min_value=0.0,
            max_value=100.0,
            value=numeric_default(base_row, "Discount_Percentage", 0.0),
            step=1.0
        )

    with r1[2]:
        promotion = st.selectbox(
            "📢 Promotion",
            ["No", "Yes"],
            index=1 if numeric_default(base_row, "Promotion_Flag", 0) >= 1 else 0
        )

    r2 = st.columns(3)

    with r2[0]:
        stock = st.number_input(
            "📦 Stock Availability",
            min_value=0.0,
            value=numeric_default(base_row, "Stock_Availability", 0.0),
            step=1.0
        )

    with r2[1]:
        holiday_option = st.selectbox(
            "🎉 Holiday",
            ["Auto Detect", "No", "Yes"]
        )

    with r2[2]:
        local_event = st.selectbox(
            "📍 Local Event",
            ["No", "Yes"],
            index=1 if numeric_default(base_row, "Local_Event_Flag", 0) >= 1 else 0
        )

    r3 = st.columns(3)

    with r3[0]:
        competitor_price = st.number_input(
            "💰 Competitor Price",
            min_value=0.0,
            value=numeric_default(base_row, "Competitor_Price", 0.0),
            step=1.0
        )

    with r3[1]:
        economic_indicator = st.number_input(
            "📊 Economic Indicator",
            value=numeric_default(base_row, "Economic_Indicator", 0.0),
            step=0.1
        )

    with r3[2]:
        marketing_spend = st.number_input(
            "📣 Marketing Spend",
            min_value=0.0,
            value=numeric_default(base_row, "Marketing_Spend", 0.0),
            step=100.0
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

    if st.button("🚀 Generate Forecast", key="combined_product_button"):

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

        st.session_state["combined_result"] = result
        st.session_state["combined_result_title"] = (
            f"{selected_product} - {selected_store} Forecast"
        )


# ============================================================
# STORE BASED
# ============================================================

else:

    st.markdown(
        """
        <div class="section">
            <div class="section-title">🏪 Store Forecast</div>
            <div class="section-sub">
                Forecast total sales for all products available in the selected store.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_store = st.selectbox(
        "🏪 Store",
        stores,
        key="combined_store"
    )

    store_rows = latest_rows_for_store(selected_store)

    s1, s2, s3 = st.columns(3)

    with s1:
        st.metric(
            "Products / Profiles",
            f"{len(store_rows):,}"
        )

    with s2:
        if "Units_Sold" in store_rows.columns:
            st.metric(
                "Latest Sales",
                f"{pd.to_numeric(store_rows['Units_Sold'], errors='coerce').fillna(0).sum():,.0f}"
            )

    with s3:
        if "Date" in store_rows.columns:
            st.metric(
                "Latest Store Data",
                str(store_rows["Date"].max().date())
            )

    if st.button("🚀 Generate Forecast", key="combined_store_button"):

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

        st.session_state["combined_result"] = result
        st.session_state["combined_result_title"] = (
            f"{selected_store} Store Forecast"
        )


# ============================================================
# ONE COMMON OUTPUT
# ============================================================

if "combined_result" in st.session_state:

    st.markdown("---")

    show_results(
        st.session_state["combined_result"],
        st.session_state.get(
            "combined_result_title",
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
