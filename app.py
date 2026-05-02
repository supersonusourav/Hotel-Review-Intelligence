import streamlit as st
import pandas as pd
import numpy as np
import os


st.set_page_config(
    page_title="Hotelier Intelligence Dashboard",
    page_icon="🏨",
    layout="wide"
)


# ============================================================
# Config
# ============================================================
DATA_CANDIDATES = [
    "df_exploded.csv"
]


ASPECT_ORDER = [
    "Staff",
    "Cleanliness",
    "Comfort",
    "Food_Beverage",
    "Infrastructure",
    "Value_Pricing",
    "Location_Transit",
    "General",
]


CITY_MAP = {
    "amsterdam": "Amsterdam",
    "barcelona": "Barcelona",
    "london": "London",
    "milan": "Milan",
    "paris": "Paris",
    "vienna": "Vienna",
}


TECH_STACK = {
    "Problem framing": "Quarterly hotel sentiment intelligence using review text and aspect tracking for guest discovery and hotel operations.",
    "Dataset shaping": "Columns used include Hotel_Name, Hotel_Address, Review_Date, Reviewer_Score, Negative_Review, Positive_Review, and Tags; city is derived from hotel address.",
    "Temporal design": "Review_Date is converted to datetime and binned into Quarter for quarterly trend analysis.",
    "Text representation": "Unified review text is formed by combining Negative_Review, Positive_Review, and Tags with separators.",
    "Aspect extraction": "Rule-based multi-label aspect tagging across Staff, Cleanliness, Comfort, Food_Beverage, Infrastructure, Value_Pricing, Location_Transit, and General.",
    "Model family": "DistilBERT used for 5-class sentiment classification.",
    "Training style": "Parameter-efficient fine-tuning using LoRA.",
    "LoRA setup": "LoRA rank 8 on DistilBERT backbone.",
    "Split strategy": "90/10 train-validation split with label stratification.",
    "Labeling": "Reviewer_Score mapped into 5 sentiment classes.",
    "Optimization": "AdamW optimizer with exponential decay learning rate schedule, batch size 64, max token length 128, and mixed precision.",
    "Backend / system": "Keras with torch backend and RTX 4050-oriented development setup.",
    "Tracked metrics": "Validation accuracy, validation loss, and trainable vs frozen parameter counts.",
    "Deployment scope": "This UI is analytics-only and does not load sarcasm detection at runtime.",
    "Analytics layer": "Quarterly aspect sentiment averages and velocity are computed from Predicted_Sentiment over Quarter.",
}


TRAINING_SUMMARY = {
    "Base model": "distilbert-base-uncased",
    "Classes": 5,
    "Validation split": "10%",
    "Epochs in notebook run": 1,
    "Batch size": 64,
    "Tokenizer max length": 128,
    "Learning rate": "Initial 3e-5 with exponential decay",
    "Loss": "SparseCategoricalCrossentropy(from_logits=True)",
    "Metric": "Accuracy",
    "Precision policy": "mixed_float16",
    "LoRA rank": 8,
}


# ============================================================
# Helpers
# ============================================================
def safe_read_csv(path_candidates):
    for path in path_candidates:
        if os.path.exists(path):
            return pd.read_csv(path), path
    return None, None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Hotel_Name": "HotelName",
        "Reviewer_Score": "ReviewerScore",
        "Review_Date": "ReviewDate",
        "Predicted_Sentiment": "PredictedSentiment",
        "Aspect": "Aspects",
        "Hotel_Address": "HotelAddress",
    }
    return df.rename(columns={c: rename_map.get(c, c) for c in df.columns})


def normalize_city(city_value: str) -> str:
    if pd.isna(city_value):
        return city_value
    city_text = str(city_value).strip()
    city_lower = city_text.lower()
    for key, value in CITY_MAP.items():
        if key in city_lower:
            return value
    return city_text


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df_raw, used_path = safe_read_csv(DATA_CANDIDATES)
    if df_raw is None:
        return pd.DataFrame()
    return normalize_columns(df_raw)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    required = ["City", "HotelName", "ReviewDate", "ReviewerScore", "Aspects"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    work = df.copy()

    work["City"] = work["City"].astype(str).str.strip().apply(normalize_city)
    work["HotelName"] = work["HotelName"].astype(str).str.strip()
    work["Aspects"] = work["Aspects"].astype(str).str.strip()

    work["ReviewDate"] = pd.to_datetime(work["ReviewDate"], errors="coerce")
    work = work.dropna(subset=["ReviewDate", "City", "HotelName", "Aspects"])

    if "Quarter" not in work.columns:
        work["Quarter"] = work["ReviewDate"].dt.to_period("Q").astype(str)
    else:
        work["Quarter"] = work["Quarter"].astype(str)

    if "PredictedSentiment" not in work.columns:
        work["PredictedSentiment"] = np.ceil(
            pd.to_numeric(work["ReviewerScore"], errors="coerce") / 2.0
        )

    work["PredictedSentiment"] = pd.to_numeric(work["PredictedSentiment"], errors="coerce")
    work["ReviewerScore"] = pd.to_numeric(work["ReviewerScore"], errors="coerce")
    work = work.dropna(subset=["PredictedSentiment", "ReviewerScore"])

    valid_aspects = [a for a in ASPECT_ORDER if a in work["Aspects"].unique().tolist()]
    if valid_aspects:
        work["Aspects"] = pd.Categorical(work["Aspects"], categories=valid_aspects, ordered=True)

    return work


def aspect_quarterly(df: pd.DataFrame, city: str = None, hotel: str = None) -> pd.DataFrame:
    scope = df.copy()

    if city:
        scope = scope[scope["City"].str.lower() == city.lower()]
    if hotel:
        scope = scope[scope["HotelName"] == hotel]

    grp = (
        scope.groupby(["Quarter", "Aspects"], observed=False)
        .agg(
            AvgSentiment=("PredictedSentiment", "mean"),
            AvgRating=("ReviewerScore", "mean"),
            ReviewCount=("PredictedSentiment", "size"),
        )
        .reset_index()
        .sort_values(["Quarter", "Aspects"])
    )
    return grp


def hotel_summary(df: pd.DataFrame, city: str = None) -> pd.DataFrame:
    scope = df.copy()

    if city:
        scope = scope[scope["City"].str.lower() == city.lower()]

    out = (
        scope.groupby(["City", "HotelName"], observed=False)
        .agg(
            Reviews=("PredictedSentiment", "size"),
            AvgSentiment=("PredictedSentiment", "mean"),
            AvgRating=("ReviewerScore", "mean"),
        )
        .reset_index()
        .sort_values(["AvgSentiment", "AvgRating", "Reviews"], ascending=[False, False, False])
    )
    return out


def velocity_table(grp: pd.DataFrame) -> pd.DataFrame:
    if grp.empty:
        return grp

    piv = grp.pivot(index="Aspects", columns="Quarter", values="AvgSentiment").sort_index(axis=1)
    piv = piv.ffill(axis=1).bfill(axis=1)

    quarter_cols = list(piv.columns)

    if len(quarter_cols) >= 2:
        piv["Velocity"] = piv[quarter_cols[-1]] - piv[quarter_cols[-2]]
    else:
        piv["Velocity"] = 0.0

    latest = quarter_cols[-1] if quarter_cols else None
    previous = quarter_cols[-2] if len(quarter_cols) >= 2 else None

    result = piv.reset_index()

    if latest:
        result = result.rename(columns={latest: "LatestQuarterScore"})
    if previous:
        result = result.rename(columns={previous: "PreviousQuarterScore"})

    return result.sort_values("Velocity")


def render_sidebar() -> str:
    st.sidebar.title("Navigation")
    role = st.sidebar.radio("User role", ["Guest", "Hotelier"], index=0)

    with st.sidebar.expander("Project tech stack", expanded=False):
        for k, v in TECH_STACK.items():
            st.markdown(f"**{k}:** {v}")

    with st.sidebar.expander("Training reference", expanded=False):
        for k, v in TRAINING_SUMMARY.items():
            st.markdown(f"**{k}:** {v}")

    return role


# ============================================================
# App
# ============================================================
raw = load_data()
role = render_sidebar()

if raw.empty:
    st.error("Data file not found. Place df_exploded.csv beside app.py.")
    st.stop()

try:
    data = prepare_data(raw)
except Exception as e:
    st.error(f"Unable to prepare dataset: {e}")
    st.stop()

if data.empty:
    st.error("Dataset loaded but no usable rows were found after cleaning.")
    st.stop()

all_cities = sorted(data["City"].dropna().astype(str).unique().tolist())
all_hotels = sorted(data["HotelName"].dropna().astype(str).unique().tolist())
quarters = sorted(data["Quarter"].dropna().astype(str).unique().tolist())

st.title("Hotelier Intelligence Dashboard")
st.caption("Quarterly hotel sentiment trends based on the full dataset and aspect-wise analysis.")

c1, c2, c3 = st.columns(3)
c1.metric("Cities", len(all_cities))
c2.metric("Hotels", len(all_hotels))
c3.metric("Quarter bins", len(quarters))


# ============================================================
# Guest View
# ============================================================
if role == "Guest":
    st.subheader("Guest search")

    # City + hotel selection (mirrors hotelier flow, but from a guest perspective)
    left, right = st.columns([2, 1])
    with left:
        selected_city = st.selectbox("Select a city", all_cities)
    with right:
        top_n = st.slider("Top hotels to show", min_value=3, max_value=20, value=5)

    st.success(f"Showing hotels for: {selected_city}")

    # City-level table for discovery
    city_hotels = hotel_summary(data, selected_city)

    st.markdown("### Hotels in this city")
    st.dataframe(
        city_hotels[["HotelName", "AvgSentiment", "AvgRating", "Reviews"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(f"### Top {top_n} hotels")
    st.dataframe(
        city_hotels.head(top_n),
        use_container_width=True,
        hide_index=True,
    )

    # NEW: hotel selectbox for guests (to mirror hotelier layout)
    hotel_options_guest = city_hotels["HotelName"].tolist()
    if hotel_options_guest:
        selected_hotel_guest = st.selectbox(
            "Select a hotel to inspect quarterly trends",
            hotel_options_guest,
        )

        # Filter dataset for that hotel (same as hotelier path)
        selected_hotel_df_guest = data[
            (data["City"] == selected_city)
            & (data["HotelName"] == selected_hotel_guest)
        ]

        hotel_q_guest = aspect_quarterly(selected_hotel_df_guest)
        hotel_stats_guest = hotel_summary(selected_hotel_df_guest)

        # Same style of metrics as Hotelier page
        m1, m2, m3 = st.columns(3)
        if not hotel_stats_guest.empty:
            row_g = hotel_stats_guest.iloc[0]
            m1.metric("Average sentiment", f"{row_g['AvgSentiment']:.2f}")
            m2.metric("Average reviewer score", f"{row_g['AvgRating']:.2f}")
            m3.metric("Aspect-tagged review rows", int(row_g["Reviews"]))

        # Hotel-level quarterly chart (not city-level)
        st.markdown("### Hotel-level quarterly merged line graph")
        st.line_chart(
            hotel_q_guest,
            x="Quarter",
            y="AvgSentiment",
            color="Aspects",
            use_container_width=True,
        )

        # Hotel-level velocity table for this guest-selected hotel
        latest_hotel_velocity_guest = velocity_table(hotel_q_guest)
        if not latest_hotel_velocity_guest.empty:
            st.markdown("### Current aspect movement for selected hotel")
            cols_guest = [
                c
                for c in [
                    "Aspects",
                    "LatestQuarterScore",
                    "PreviousQuarterScore",
                    "Velocity",
                ]
                if c in latest_hotel_velocity_guest.columns
            ]
            st.dataframe(
                latest_hotel_velocity_guest[cols_guest],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No hotels available for this city in the dataset.")


# ============================================================
# Hotelier View
# ============================================================
else:
    st.subheader("Hotelier dashboard")

    city_choice = st.selectbox("Select city", all_cities)
    hotel_options = sorted(
        data.loc[data["City"] == city_choice, "HotelName"].unique().tolist()
    )
    hotel_choice = st.selectbox("Select hotel", hotel_options)

    selected_hotel_df = data[
        (data["City"] == city_choice)
        & (data["HotelName"] == hotel_choice)
    ]

    hotel_q = aspect_quarterly(selected_hotel_df)
    hotel_stats = hotel_summary(selected_hotel_df)

    m1, m2, m3 = st.columns(3)
    if not hotel_stats.empty:
        row = hotel_stats.iloc[0]
        m1.metric("Average sentiment", f"{row['AvgSentiment']:.2f}")
        m2.metric("Average reviewer score", f"{row['AvgRating']:.2f}")
        m3.metric("Aspect-tagged review rows", int(row["Reviews"]))

    st.markdown("### Hotel quarterly merged line graph")
    st.line_chart(
        hotel_q,
        x="Quarter",
        y="AvgSentiment",
        color="Aspects",
        use_container_width=True,
    )

    improve = velocity_table(hotel_q)
    if not improve.empty:
        st.markdown("### Areas needing improvement")
        needs_work = improve[improve["Velocity"] <= 0].copy()

        if needs_work.empty:
            st.success("No declining aspect was detected in the latest quarter comparison.")
        else:
            cols = [
                c
                for c in [
                    "Aspects",
                    "LatestQuarterScore",
                    "PreviousQuarterScore",
                    "Velocity",
                ]
                if c in needs_work.columns
            ]
            st.dataframe(
                needs_work[cols],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### All aspect velocity values")
        cols = [
            c
            for c in [
                "Aspects",
                "LatestQuarterScore",
                "PreviousQuarterScore",
                "Velocity",
            ]
            if c in improve.columns
        ]
        st.dataframe(
            improve[cols],
            use_container_width=True,
            hide_index=True,
        )

st.markdown("---")
st.caption(
    "UI simplified as requested: no sarcasm detector, quarterly trends from the full dataset, simple guest search, and hotelier improvement tracking."
)