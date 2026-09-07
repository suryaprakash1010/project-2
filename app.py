import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import calendar


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI Sales Forecasting",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# MULTI COLOUR DESIGN
# ============================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background:
        radial-gradient(
            circle at 10% 20%,
            rgba(186,104,200,0.35),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(66,165,245,0.35),
            transparent 30%
        ),
        radial-gradient(
            circle at 20% 85%,
            rgba(38,198,218,0.30),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 85%,
            rgba(255,183,77,0.30),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #f3e5f5,
            #e3f2fd,
            #e0f7fa,
            #fff8e1
        );

    background-attachment: fixed;
}


/* MAIN CONTAINER */

.block-container {

    background: rgba(255,255,255,0.55);

    backdrop-filter: blur(12px);

    border-radius: 25px;

    padding: 2rem;

    margin-top: 25px;

    margin-bottom: 25px;

    box-shadow:
        0px 10px 35px
        rgba(31,38,135,0.15);
}


/* SIDEBAR */

[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #512da8,
            #1976d2,
            #00897b
        );
}


[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {

    color: white !important;
}


/* MAIN TITLE */

.main-title {

    background:
        linear-gradient(
            90deg,
            #7b1fa2,
            #3949ab,
            #039be5,
            #00897b
        );

    color: white;

    padding: 25px;

    border-radius: 22px;

    text-align: center;

    font-size: 40px;

    font-weight: 800;

    box-shadow:
        0px 8px 25px
        rgba(63,81,181,0.30);
}


/* SUB TITLE */

.sub-title {

    text-align: center;

    color: #3949ab;

    font-size: 18px;

    font-weight: 600;

    margin-top: 12px;

    margin-bottom: 30px;
}


/* BUTTON */

div.stButton > button {

    width: 100%;

    background:
        linear-gradient(
            90deg,
            #7b1fa2,
            #1976d2,
            #00a896,
            #ff9800
        );

    color: white;

    border: none;

    border-radius: 15px;

    padding: 14px;

    font-size: 20px;

    font-weight: bold;

    box-shadow:
        0px 7px 18px
        rgba(0,0,0,0.18);
}


div.stButton > button:hover {

    transform: translateY(-2px);

    color: white;

    border: none;
}


/* METRIC CARDS */

[data-testid="stMetric"] {

    background: rgba(255,255,255,0.85);

    padding: 20px;

    border-radius: 18px;

    border-left: 6px solid #7b1fa2;

    box-shadow:
        0px 6px 20px
        rgba(0,0,0,0.12);
}


[data-testid="stMetricLabel"] {

    font-weight: bold;

    color: #3949ab;
}


[data-testid="stMetricValue"] {

    color: #00897b;
}


/* RESULT CARD */

.result-card {

    background:
        linear-gradient(
            135deg,
            #7b1fa2,
            #1976d2,
            #009688
        );

    color: white;

    padding: 30px;

    border-radius: 22px;

    text-align: center;

    margin-top: 20px;

    margin-bottom: 20px;

    box-shadow:
        0px 8px 25px
        rgba(0,0,0,0.20);
}


.result-number {

    color: #ffeb3b;

    font-size: 45px;

    font-weight: bold;
}


/* TABLE */

[data-testid="stDataFrame"] {

    background: white;

    border-radius: 18px;

    padding: 10px;

    box-shadow:
        0px 6px 20px
        rgba(0,0,0,0.10);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div class="main-title">
        📈 AI SALES FORECASTING SYSTEM
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
        Smart Future Sales Prediction Dashboard
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    artifacts = joblib.load(
        "xgboost_sales_model_artifacts.pkl"
    )

    model = artifacts["xgb_model"]

    label_encoders = artifacts["label_encoders"]

    model_features = artifacts["model_features"]

    historical_data = artifacts["historical_data"].copy()

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )

    st.stop()


# ============================================================
# DATE CONVERSION
# ============================================================

historical_data["Date"] = pd.to_datetime(
    historical_data["Date"]
)

historical_data = (
    historical_data
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Forecast Settings"
)


# ============================================================
# PRODUCT NAME
# ============================================================

if "Product_Name" in historical_data.columns:

    product_names = sorted(
        historical_data[
            "Product_Name"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_product_name = (
        st.sidebar.selectbox(
            "🛍️ Product Name",
            product_names
        )
    )

    product_data = historical_data[
        historical_data[
            "Product_Name"
        ].astype(str)
        ==
        str(selected_product_name)
    ].copy()

else:

    selected_product_name = None

    product_data = historical_data.copy()


# ============================================================
# PRODUCT ID
# ============================================================

if "Product_ID" in product_data.columns:

    product_ids = sorted(
        product_data[
            "Product_ID"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_product = (
        st.sidebar.selectbox(
            "🏷️ Product ID",
            product_ids
        )
    )

else:

    selected_product = None


# ============================================================
# STORE ID
# ============================================================

store_data = product_data.copy()


if selected_product is not None:

    store_data = store_data[
        store_data[
            "Product_ID"
        ].astype(str)
        ==
        str(selected_product)
    ]


if "Store_ID" in store_data.columns:

    stores = sorted(
        store_data[
            "Store_ID"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_store = (
        st.sidebar.selectbox(
            "🏪 Store",
            stores
        )
    )

else:

    selected_store = None


# ============================================================
# IMPORTANT FEATURES
# ============================================================

st.sidebar.subheader(
    "📊 Important Features"
)


price = st.sidebar.number_input(
    "💰 Price",
    min_value=0.0,
    value=100.0,
    step=10.0
)


discount = st.sidebar.number_input(
    "🏷️ Discount %",
    min_value=0.0,
    max_value=100.0,
    value=0.0
)


marketing_spend = (
    st.sidebar.number_input(
        "📢 Marketing Spend",
        min_value=0.0,
        value=0.0
    )
)


promotion_flag = (
    st.sidebar.selectbox(
        "🎁 Promotion",
        ["No", "Yes"]
    )
)


promotion_flag = (
    1
    if promotion_flag == "Yes"
    else 0
)


stock_availability = (
    st.sidebar.number_input(
        "📦 Stock Availability",
        min_value=0,
        value=100
    )
)


competitor_price = (
    st.sidebar.number_input(
        "💵 Competitor Price",
        min_value=0.0,
        value=100.0
    )
)


holiday_flag = (
    st.sidebar.selectbox(
        "🎉 Holiday",
        ["No", "Yes"]
    )
)


holiday_flag = (
    1
    if holiday_flag == "Yes"
    else 0
)


# ============================================================
# FORECAST TYPE
# ============================================================

st.subheader(
    "🔮 Select Forecast Period"
)


forecast_type = st.selectbox(

    "What do you want to forecast?",

    [
        "Next Week",
        "Next Month",
        "Particular Date",
        "Next Year"
    ]
)


# ============================================================
# LAST DATASET DATE
# ============================================================

last_date = (
    historical_data["Date"].max()
)


st.info(
    "📅 Last available sales date: "
    +
    last_date.strftime(
        "%d-%m-%Y"
    )
)


# ============================================================
# FORECAST DATES
# ============================================================

forecast_dates = None

selected_date = None


# NEXT WEEK

if forecast_type == "Next Week":

    forecast_dates = pd.date_range(

        start=(
            last_date
            +
            timedelta(days=1)
        ),

        periods=7,

        freq="D"
    )


# NEXT MONTH

elif forecast_type == "Next Month":

    next_month_date = (
        last_date
        +
        pd.DateOffset(months=1)
    )

    year = next_month_date.year

    month = next_month_date.month

    total_days = (
        calendar.monthrange(
            year,
            month
        )[1]
    )

    start_date = pd.Timestamp(
        year=year,
        month=month,
        day=1
    )

    forecast_dates = pd.date_range(
        start=start_date,
        periods=total_days,
        freq="D"
    )


# PARTICULAR DATE

elif forecast_type == "Particular Date":

    selected_date = st.date_input(

        "📅 Choose Future Date",

        min_value=(
            last_date
            +
            timedelta(days=1)
        ).date()
    )

    forecast_dates = (
        pd.DatetimeIndex(
            [
                pd.Timestamp(
                    selected_date
                )
            ]
        )
    )


# NEXT YEAR

elif forecast_type == "Next Year":

    next_year = (
        last_date.year + 1
    )

    start_date = pd.Timestamp(
        year=next_year,
        month=1,
        day=1
    )

    end_date = pd.Timestamp(
        year=next_year,
        month=12,
        day=31
    )

    forecast_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )


# ============================================================
# FORECAST BUTTON
# ============================================================

if st.button(
    "🚀 FORECAST SALES",
    use_container_width=True
):


    # ========================================================
    # FILTER HISTORY
    # ========================================================

    history = historical_data.copy()


    # Product Name

    if selected_product_name is not None:

        history = history[
            history[
                "Product_Name"
            ].astype(str)
            ==
            str(selected_product_name)
        ]


    # Product ID

    if selected_product is not None:

        history = history[
            history[
                "Product_ID"
            ].astype(str)
            ==
            str(selected_product)
        ]


    # Store

    if selected_store is not None:

        history = history[
            history[
                "Store_ID"
            ].astype(str)
            ==
            str(selected_store)
        ]


    history = (
        history
        .sort_values("Date")
        .reset_index(drop=True)
    )


    # ========================================================
    # CHECK HISTORY
    # ========================================================

    if len(history) < 30:

        st.error(
            "Not enough previous sales history "
            "for selected Product + Store."
        )

        st.stop()


    # ========================================================
    # SALES HISTORY
    # ========================================================

    sales_history = (
        history[
            "Units_Sold"
        ]
        .astype(float)
        .tolist()
    )


    predictions = []


    last_row = history.iloc[-1]


    # ========================================================
    # FUTURE PREDICTION LOOP
    # ========================================================

    for future_date in forecast_dates:

        row = {}


        # ====================================================
        # PRODUCT INFORMATION
        # ====================================================

        if selected_product_name is not None:

            row[
                "Product_Name"
            ] = selected_product_name


        if selected_product is not None:

            row[
                "Product_ID"
            ] = selected_product


        if selected_store is not None:

            row[
                "Store_ID"
            ] = selected_store


        # ====================================================
        # USER INPUT FEATURES
        # ====================================================

        row["Price"] = price

        row[
            "Discount_Percentage"
        ] = discount

        row[
            "Marketing_Spend"
        ] = marketing_spend

        row[
            "Promotion_Flag"
        ] = promotion_flag

        row[
            "Stock_Availability"
        ] = stock_availability

        row[
            "Competitor_Price"
        ] = competitor_price

        row[
            "Holiday_Flag"
        ] = holiday_flag


        # ====================================================
        # DATE FEATURES
        # ====================================================

        row["Year"] = (
            future_date.year
        )

        row["Month_Number"] = (
            future_date.month
        )

        row["Day"] = (
            future_date.day
        )

        row["DayOfYear"] = (
            future_date.dayofyear
        )

        row["WeekOfYear"] = int(
            future_date
            .isocalendar()
            .week
        )

        row[
            "DayOfWeek_Number"
        ] = future_date.dayofweek


        # ====================================================
        # COPY OTHER FEATURES FROM LAST RECORD
        # ====================================================

        for column in history.columns:

            if (
                column not in row
                and
                column not in [
                    "Date",
                    "Units_Sold",
                    "Revenue",
                    "Row_ID"
                ]
            ):

                row[column] = (
                    last_row[column]
                )


        # ====================================================
        # UPDATE DATE CATEGORICAL FEATURES
        # ====================================================

        if "Day_of_Week" in row:

            row[
                "Day_of_Week"
            ] = future_date.day_name()


        if "Month" in row:

            row[
                "Month"
            ] = future_date.month_name()


        if "Quarter" in row:

            row[
                "Quarter"
            ] = (
                "Q"
                +
                str(
                    future_date.quarter
                )
            )


        if "Is_Weekend" in row:

            row[
                "Is_Weekend"
            ] = (
                1
                if future_date.dayofweek >= 5
                else 0
            )


        # ====================================================
        # INTERACTION FEATURES
        # ====================================================

        row[
            "Pricing_Effect"
        ] = (
            price
            *
            discount
        )


        row[
            "Promotion_Effect"
        ] = (
            marketing_spend
            *
            promotion_flag
        )


        row[
            "Inventory_Value"
        ] = (
            stock_availability
            *
            price
        )


        row[
            "Seasonal_Effect"
        ] = (
            promotion_flag
            *
            holiday_flag
        )


        row[
            "Price_Difference"
        ] = (
            price
            -
            competitor_price
        )


        row[
            "Discounted_Price"
        ] = (
            price
            *
            (
                1
                -
                discount / 100
            )
        )


        # ====================================================
        # SALES LAG FEATURES
        # ====================================================

        row[
            "Sales_Lag_1"
        ] = sales_history[-1]


        row[
            "Sales_Lag_2"
        ] = sales_history[-2]


        row[
            "Sales_Lag_3"
        ] = sales_history[-3]


        row[
            "Sales_Lag_7"
        ] = sales_history[-7]


        row[
            "Sales_Lag_14"
        ] = sales_history[-14]


        row[
            "Sales_Lag_30"
        ] = sales_history[-30]


        # ====================================================
        # ROLLING FEATURES
        # ====================================================

        row[
            "Sales_Rolling_Mean_3"
        ] = np.mean(
            sales_history[-3:]
        )


        row[
            "Sales_Rolling_Mean_7"
        ] = np.mean(
            sales_history[-7:]
        )


        row[
            "Sales_Rolling_Mean_14"
        ] = np.mean(
            sales_history[-14:]
        )


        row[
            "Sales_Rolling_Mean_30"
        ] = np.mean(
            sales_history[-30:]
        )


        row[
            "Sales_Rolling_Std_7"
        ] = np.std(
            sales_history[-7:]
        )


        # ====================================================
        # SALES CHANGE FEATURES
        # ====================================================

        row[
            "Sales_Change_1"
        ] = (
            sales_history[-1]
            -
            sales_history[-2]
        )


        row[
            "Sales_Change_7"
        ] = (
            sales_history[-1]
            -
            sales_history[-7]
        )


        # ====================================================
        # CREATE DATAFRAME
        # ====================================================

        future_df = pd.DataFrame(
            [row]
        )


        # ====================================================
        # ENCODING
        # ====================================================

        for (
            column,
            encoder
        ) in label_encoders.items():

            if column in future_df.columns:

                value = str(
                    future_df[
                        column
                    ].iloc[0]
                )

                if value in encoder.classes_:

                    future_df[
                        column
                    ] = encoder.transform(
                        [value]
                    )

                else:

                    future_df[
                        column
                    ] = 0


        # ====================================================
        # ADD MISSING MODEL FEATURES
        # ====================================================

        for feature in model_features:

            if feature not in future_df.columns:

                future_df[
                    feature
                ] = 0


        # ====================================================
        # SAME FEATURE ORDER AS TRAINING
        # ====================================================

        future_df = future_df[
            model_features
        ]


        # ====================================================
        # CLEAN DATA
        # ====================================================

        future_df = (
            future_df
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )


        # ====================================================
        # PREDICT
        # ====================================================

        prediction = (
            model.predict(
                future_df
            )[0]
        )


        prediction = max(
            0,
            float(prediction)
        )


        predictions.append(
            prediction
        )


        # Add prediction for next day's lag

        sales_history.append(
            prediction
        )


    # ========================================================
    # RESULT
    # ========================================================

    result = pd.DataFrame({

        "Date":
            forecast_dates,

        "Predicted_Sales":
            predictions

    })


    result[
        "Predicted_Sales"
    ] = (
        result[
            "Predicted_Sales"
        ]
        .round()
        .astype(int)
    )


    # ========================================================
    # SUCCESS
    # ========================================================

    st.success(
        "✅ Sales Forecast Completed!"
    )


    # ========================================================
    # PARTICULAR DATE RESULT
    # ========================================================

    if forecast_type == "Particular Date":

        predicted_sales = (
            result[
                "Predicted_Sales"
            ].iloc[0]
        )


        st.markdown(
            f"""
            <div class="result-card">

                <h2 style="color:white;">
                    🎯 Predicted Sales
                </h2>

                <div class="result-number">
                    {predicted_sales} Units
                </div>

                <br>

                🛍️ Product:
                {selected_product_name}

                <br>

                🏷️ Product ID:
                {selected_product}

                <br>

                🏪 Store:
                {selected_store}

                <br>

                📅 Date:
                {pd.Timestamp(selected_date).strftime("%d-%m-%Y")}

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # WEEK / MONTH / YEAR RESULT
    # ========================================================

    else:

        total_sales = (
            result[
                "Predicted_Sales"
            ].sum()
        )


        average_sales = (
            result[
                "Predicted_Sales"
            ].mean()
        )


        highest_sales = (
            result[
                "Predicted_Sales"
            ].max()
        )


        lowest_sales = (
            result[
                "Predicted_Sales"
            ].min()
        )


        col1, col2 = st.columns(2)

        col3, col4 = st.columns(2)


        col1.metric(
            "📦 Total Sales",
            f"{total_sales:,} Units"
        )


        col2.metric(
            "📊 Average Daily Sales",
            f"{average_sales:.0f} Units"
        )


        col3.metric(
            "🔥 Highest Daily Sales",
            f"{highest_sales:,} Units"
        )


        col4.metric(
            "📉 Lowest Daily Sales",
            f"{lowest_sales:,} Units"
        )


        # ====================================================
        # GRAPH
        # ====================================================

        st.subheader(
            "📈 Future Sales Forecast"
        )


        chart_data = (
            result
            .set_index("Date")[
                "Predicted_Sales"
            ]
        )


        st.line_chart(
            chart_data
        )


        # ====================================================
        # TABLE
        # ====================================================

        st.subheader(
            "📋 Forecast Details"
        )


        st.dataframe(
            result,
            use_container_width=True
        )
