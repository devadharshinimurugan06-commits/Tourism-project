import os
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "clf": "best_classification_model.pkl",
    "reg": "best_regression_model.pkl",
    "label_encoder": "classification_label_encoder.pkl",
    "classes": "classification_classes.pkl",
    "tfidf": "tfidf_vectorizer.pkl",
    "cosine": "cosine_similarity.pkl",
    "rec_data": "attraction_recommendation_data.pkl",
    "prep_objects": "preprocessing_objects.pkl",
}

def find_file(filename):
    """Look for filename directly next to app.py first; if not found there,
    search one level of subdirectories too (handles repos that keep the
    .pkl artifacts inside a data/ or similar subfolder)."""
    direct = os.path.join(BASE_DIR, filename)
    if os.path.exists(direct):
        return direct
    for root, _dirs, files in os.walk(BASE_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None

def p(name):
    return find_file(FILES[name])

@st.cache_resource
def load_artifacts():
    out, errors = {}, []
    for key, fn in FILES.items():
        fp = p(key)
        if fp is None:
            errors.append(f"{fn}: not found anywhere under {BASE_DIR}")
            continue
        try:
            out[key] = joblib.load(fp)
        except Exception as e1:
            try:
                with open(fp, "rb") as f:
                    out[key] = pickle.load(f)
            except Exception as e2:
                errors.append(f"{fn}: joblib error: {e1!r} | pickle error: {e2!r}")
    return out, errors

artifacts, load_errors = load_artifacts()

@st.cache_data
def load_raw():
    names = [
        "Transaction.xlsx", "User.xlsx", "City.xlsx", "Country.xlsx",
        "Region.xlsx", "Continent.xlsx", "Type.xlsx", "Mode.xlsx",
        "Updated_Item.xlsx"
    ]
    data = {}
    for fn in names:
        fp = os.path.join(BASE_DIR, fn)
        if os.path.exists(fp):
            data[fn] = pd.read_excel(fp)
        else:
            data[fn] = None

    # Fallback: allow the attraction file to be inside the project subfolder.
    if data["Updated_Item.xlsx"] is None:
        fp = os.path.join(BASE_DIR, "Additional_Data_for_Attraction_Sites", "Updated_Item.xlsx")
        if os.path.exists(fp):
            data["Updated_Item.xlsx"] = pd.read_excel(fp)
    return data

raw = load_raw()

@st.cache_data
def build_attraction_location(raw):
    """Correct attraction-level location, built fresh from the raw item/city/
    country tables. Fixes the bug where CityName/Country showed the VISITING
    USER's home city instead of the attraction's own location."""
    item = raw["Updated_Item.xlsx"].copy()
    city = raw["City.xlsx"].copy()
    country = raw["Country.xlsx"].copy()

    loc = item[["AttractionId", "Attraction", "AttractionCityId"]].merge(
        city, left_on="AttractionCityId", right_on="CityId", how="left"
    )
    loc = loc.merge(country, on="CountryId", how="left")
    loc = loc[["AttractionId", "Attraction", "CityName", "Country"]].drop_duplicates("Attraction")
    return loc

attraction_location = build_attraction_location(raw)

# ------------------------------------------------------------
# EXACT TRAINING-TIME FEATURE ENGINEERING REPRODUCED FOR APP
# Model inspection confirmed:
#   Regression = 208 features
#   Classification = same 208 + 5 User_Mode_Prob features = 213
# The model feature names are used as the final column contract.
# ------------------------------------------------------------

MODE_LABELS = ["Business", "Couples", "Family", "Friends", "Solo"]

@st.cache_data
def build_display_data(raw):
    """Merge the raw lookup tables purely so the UI can list users/attractions
    and read a user's Country/Region/Continent or an attraction's AttractionType.
    This performs no statistics/encoder fitting, so the result is a plain,
    picklable DataFrame that st.cache_data can serialize without issue."""
    tx = raw["Transaction.xlsx"].copy()
    user = raw["User.xlsx"].copy()
    city = raw["City.xlsx"].copy()
    country = raw["Country.xlsx"].copy()
    region = raw["Region.xlsx"].copy()
    continent = raw["Continent.xlsx"].copy()
    typ = raw["Type.xlsx"].copy()
    item = raw["Updated_Item.xlsx"].copy()

    df = tx.merge(user, on="UserId", how="left")
    df = df.merge(city, on="CityId", how="left", suffixes=("", "_city"))
    df = df.merge(country, on="CountryId", how="left", suffixes=("", "_country"))
    df = df.merge(region, on="RegionId", how="left", suffixes=("", "_region"))
    df = df.merge(continent, on="ContinentId", how="left", suffixes=("", "_cont"))
    df = df.merge(item, on="AttractionId", how="left")
    df = df.merge(typ, on="AttractionTypeId", how="left")
    return df

@st.cache_data
def compute_dataset_insights(raw):
    """Purely descriptive analytics computed from the FULL raw dataset (every
    transaction, not just the training split used by the models). This is
    used ONLY to show extra context/charts to the app user — it is never
    used as model input and does not touch build_display_data, prep, or
    make_prediction_row in any way."""
    tx = raw["Transaction.xlsx"].copy()
    user = raw["User.xlsx"].copy()
    city = raw["City.xlsx"].copy()
    country = raw["Country.xlsx"].copy()
    region = raw["Region.xlsx"].copy()
    continent = raw["Continent.xlsx"].copy()
    typ = raw["Type.xlsx"].copy()
    item = raw["Updated_Item.xlsx"].copy()
    mode = raw.get("Mode.xlsx")

    df = tx.merge(user, on="UserId", how="left")
    df = df.merge(city, on="CityId", how="left", suffixes=("", "_city"))
    df = df.merge(country, on="CountryId", how="left", suffixes=("", "_country"))
    df = df.merge(region, on="RegionId", how="left", suffixes=("", "_region"))
    df = df.merge(continent, on="ContinentId", how="left", suffixes=("", "_cont"))
    df = df.merge(item, on="AttractionId", how="left")
    df = df.merge(typ, on="AttractionTypeId", how="left")

    if mode is not None:
        mode_map = mode.rename(columns={"VisitMode": "VisitModeName"})
        df = df.merge(
            mode_map[["VisitModeId", "VisitModeName"]],
            left_on="VisitMode", right_on="VisitModeId", how="left"
        )
    else:
        fallback = {1: "Business", 2: "Couples", 3: "Family", 4: "Friends", 5: "Solo"}
        df["VisitModeName"] = df["VisitMode"].map(fallback)

    visit_mode_counts = (
        df["VisitModeName"].value_counts()
        .rename_axis("VisitMode").reset_index(name="Count")
    )

    rating_counts = (
        tx["Rating"].value_counts().sort_index()
        .rename_axis("Rating").reset_index(name="Count")
    )

    top_attractions = (
        df.groupby(["Attraction", "AttractionType"])
        .agg(VisitCount=("TransactionId", "count"), AvgRating=("Rating", "mean"))
        .reset_index()
        .sort_values("VisitCount", ascending=False)
        .head(10)
    )

    attraction_type_counts = (
        df.groupby("AttractionType")
        .agg(VisitCount=("TransactionId", "count"))
        .reset_index()
        .sort_values("VisitCount", ascending=False)
    )

    continent_counts = (
        df.drop_duplicates("UserId")["Continent"]
        .value_counts().rename_axis("Continent").reset_index(name="Users")
    )

    monthly_counts = (
        df.groupby("VisitMonth").size()
        .reindex(range(1, 13), fill_value=0)
        .rename_axis("Month").reset_index(name="Visits")
    )

    return {
        "df": df,
        "total_transactions": int(len(tx)),
        "total_users": int(df["UserId"].nunique()),
        "total_attractions_used": int(df["AttractionId"].nunique()),
        "total_attractions_catalog": int(item["AttractionId"].nunique()),
        "avg_rating": float(tx["Rating"].mean()),
        "visit_mode_counts": visit_mode_counts,
        "rating_counts": rating_counts,
        "top_attractions": top_attractions,
        "attraction_type_counts": attraction_type_counts,
        "continent_counts": continent_counts,
        "monthly_counts": monthly_counts,
    }

prep = None
try:
    pobj = artifacts.get("prep_objects")
    if pobj is None:
        detail = next(
            (e for e in load_errors if "preprocessing_objects.pkl" in e),
            "no specific error was captured — check the 'Artifact details' "
            "expander in the sidebar for the full list of load errors."
        )
        raise RuntimeError(f"preprocessing_objects.pkl could not be loaded: {detail}")

    prep = {
        "df": build_display_data(raw),
        "ohe": pobj["ohe"],
        "scaler": pobj["scaler"],
        "numeric_features": pobj["numeric_features"],
        "categorical_features": pobj["categorical_features"],
        "user_avg_lookup": pobj["user_avg_lookup"],
        "attr_avg_lookup": pobj["attr_avg_lookup"],
        "user_visit_count": pobj["user_visit_count"],
        "attraction_popularity": pobj["attraction_popularity"],
        "user_mode_probs": pobj["user_mode_probs"],
        "global_probs": pobj["global_probs"],
        # Weighted fallbacks for users/attractions with no training history.
        "global_user_avg": float(
            np.average(pobj["user_avg_lookup"], weights=pobj["user_visit_count"])
        ),
        "global_attr_avg": float(pobj["attr_avg_lookup"].mean()),
    }
except Exception as e:
    prep_error = str(e)

# Separate, non-critical load: descriptive dataset insights for extra context
# shown to the user. Wrapped independently so a failure here can NEVER break
# the prediction pages above.
insights = None
try:
    insights = compute_dataset_insights(raw)
except Exception as e:
    insights_error = str(e)

def make_prediction_row(user_id, attraction_id, year, month):
    df = prep["df"]
    NUMERIC = prep["numeric_features"]
    CATEGORICAL = prep["categorical_features"]

    user_rows = df[df["UserId"] == user_id]
    attr_rows = df[df["AttractionId"] == attraction_id]

    if user_rows.empty:
        raise ValueError("Selected User ID is not available in the training data.")
    if attr_rows.empty:
        raise ValueError("Selected attraction is not available in the training data.")

    # User profile comes from the User merge; attraction details from Item merge.
    u = user_rows.iloc[0]
    a = attr_rows.iloc[0]

    row = {
        "UserId": user_id,
        "VisitYear": int(year),
        "VisitMonth": int(month),
        "AttractionId": attraction_id,
        "Country": u.get("Country", "Unknown"),
        "Region": u.get("Region", "Unknown"),
        "Continent": u.get("Continent", "Unknown"),
        "AttractionType": a.get("AttractionType", "Unknown")
    }

    x = pd.DataFrame([row])

    # Exact training-time lookups (loaded from preprocessing_objects.pkl),
    # not re-derived statistics — this keeps app-time features on the same
    # scale the scaler/model were actually fit on.
    x["User_Avg_Rating"] = x["UserId"].map(prep["user_avg_lookup"]).fillna(prep["global_user_avg"])
    x["Attraction_Avg_Rating"] = x["AttractionId"].map(prep["attr_avg_lookup"]).fillna(prep["global_attr_avg"])
    x["User_Visit_Count"] = x["UserId"].map(prep["user_visit_count"]).fillna(0.0)
    x["Attraction_Popularity"] = x["AttractionId"].map(prep["attraction_popularity"]).fillna(0.0)
    x["Log_User_Visits"] = np.log1p(x["User_Visit_Count"])
    x["Log_Attraction_Popularity"] = np.log1p(x["Attraction_Popularity"])
    x["VisitQuarter"] = ((x["VisitMonth"] - 1) // 3 + 1).astype(int)

    season_map = {
        1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
        12: "Winter"
    }
    x["Season"] = x["VisitMonth"].map(season_map)
    for c in CATEGORICAL:
        x[c] = x[c].fillna("Unknown").astype(str)

    scaled = prep["scaler"].transform(x[NUMERIC])
    encoded = prep["ohe"].transform(x[CATEGORICAL])
    base = pd.DataFrame(
        np.hstack([scaled, encoded]),
        columns=NUMERIC + list(prep["ohe"].get_feature_names_out(CATEGORICAL))
    )

    # Reindex to the exact model contract.
    reg_model = artifacts["reg"]
    base = base.reindex(columns=list(reg_model.feature_names_in_), fill_value=0.0)

    clf_model = artifacts["clf"]
    clf_base = base.copy()

    if user_id in prep["user_mode_probs"].index:
        probs = prep["user_mode_probs"].loc[user_id].reindex(MODE_LABELS, fill_value=0.0).to_numpy(dtype=float)
    else:
        probs = prep["global_probs"].reindex(MODE_LABELS, fill_value=0.0).to_numpy(dtype=float)

    for label, prob in zip(MODE_LABELS, probs):
        clf_base["User_Mode_Prob_" + label] = prob

    clf_base = clf_base.reindex(
        columns=list(clf_model.feature_names_in_), fill_value=0.0
    )
    return base, clf_base

# ------------------------------------------------------------
# Recommendation helpers
# ------------------------------------------------------------
def get_recommendations(attraction_name, top_n):
    rec = artifacts.get("rec_data")
    sim = artifacts.get("cosine")

    if rec is None or sim is None:
        raise ValueError("Recommendation artifacts are missing.")

    rec = rec.copy().reset_index(drop=True)
    names = rec["Attraction"].astype(str).tolist()

    if attraction_name not in names:
        raise ValueError("Selected attraction is not available in recommendation data.")

    i = names.index(attraction_name)
    scores = np.asarray(sim[i]).ravel()

    out = rec.copy()
    out["SimilarityScore"] = scores

    # Exclude the selected attraction itself.
    out = out[out["Attraction"].astype(str) != attraction_name].copy()

    # Existing project artifact stores PopularityScore.
    if "PopularityScore" not in out.columns:
        out["PopularityScore"] = 0.0

    out["HybridScore"] = (
        0.80 * out["SimilarityScore"] +
        0.20 * out["PopularityScore"]
    )
    out = out.sort_values("HybridScore", ascending=False).head(top_n)

    # The saved recommendation metadata contains attraction-city mappings
    # that can be inconsistent with the source Item addresses. Therefore
    # only show metadata that is safe in the saved artifact.
    cols = [c for c in [
        "Attraction", "AttractionType", "VisitCount",
        "SimilarityScore", "PopularityScore", "HybridScore"
    ] if c in out.columns]
    return out[cols]

# ------------------------------------------------------------
# Descriptive context helpers (for extra info shown to the user only —
# these are never used as model input and never touch make_prediction_row
# or get_recommendations above).
# ------------------------------------------------------------
def user_history_snapshot(insights, user_id):
    if insights is None:
        return None
    d = insights["df"]
    rows = d[d["UserId"] == user_id]
    if rows.empty:
        return None
    u0 = rows.iloc[0]
    return {
        "country": u0.get("Country", "Unknown"),
        "region": u0.get("Region", "Unknown"),
        "continent": u0.get("Continent", "Unknown"),
        "visit_count": int(len(rows)),
        "avg_rating": float(rows["Rating"].mean()),
        "mode_counts": rows["VisitModeName"].value_counts(),
    }

def attraction_snapshot(insights, attraction_id):
    if insights is None:
        return None
    d = insights["df"]
    rows = d[d["AttractionId"] == attraction_id]
    if rows.empty:
        return None
    a0 = rows.iloc[0]
    return {
        "type": a0.get("AttractionType", "Unknown"),
        "address": a0.get("AttractionAddress", "Unknown"),
        "visit_count": int(len(rows)),
        "avg_rating": float(rows["Rating"].mean()),
        "mode_counts": rows["VisitModeName"].value_counts(),
    }

def show_also_like(attraction_name, n=3):
    """Cross-feature add-on for the Classification/Regression pages: reuses
    the existing, unmodified get_recommendations() to surface similar
    attractions right after a prediction. Purely additive display —
    does not change get_recommendations() or any prediction logic."""
    rec = artifacts.get("rec_data")
    if rec is None or attraction_name not in rec["Attraction"].astype(str).tolist():
        return
    try:
        similar = get_recommendations(attraction_name, n)
    except Exception:
        return
    if similar.empty:
        return
    st.markdown("#### 🔗 You might also like")
    cols = st.columns(len(similar))
    for col, (_, row) in zip(cols, similar.iterrows()):
        with col:
            st.markdown(
                f'<div class="also-like-card">'
                f'<b>{row["Attraction"]}</b><br>'
                f'🏷️ {row.get("AttractionType","")}<br>'
                f'🔥 match {row["HybridScore"]:.2f}'
                f'</div>', unsafe_allow_html=True
            )

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown("""
<style>
.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #38bdf8, #0284c7);
    color: white;
    border: 3px solid #f5b301;
    margin-bottom: 1.5rem;
}
.card {
    padding: 1rem;
    border-radius: 15px;
    border: 1px solid rgba(2,132,199,.3);
    background: rgba(56,189,248,.10);
}
.snapshot-card {
    padding: 0.9rem 1.1rem;
    border-radius: 14px;
    border: 1px solid rgba(2,132,199,.3);
    background: rgba(191,232,255,.55);
    color: #0b3556;
    margin-bottom: 0.6rem;
}
.also-like-card {
    padding: 0.8rem 1rem;
    border-radius: 14px;
    border: 1px solid rgba(245,179,1,.5);
    background: rgba(245,179,1,.12);
    color: #0b3556;
    margin-bottom: 0.5rem;
}
/* ---- Sidebar navigation styling ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #bfe8ff 0%, #8fd3ff 100%);
}
section[data-testid="stSidebar"] .stRadio > div {
    gap: 0.45rem;
}
section[data-testid="stSidebar"] .stRadio > div > label {
    padding: 0.65rem 1rem;
    border-radius: 12px;
    background: rgba(255,255,255,0.5);
    border: 1px solid rgba(2,132,199,0.25);
    transition: all .15s ease-in-out;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(245,179,1,0.35);
    border-color: rgba(245,179,1,0.7);
}
section[data-testid="stSidebar"] .stRadio > div > label > div:first-child {
    display: none;
}
.sidebar-stat {
    font-size: 0.82rem;
    opacity: 0.9;
    margin: 0.15rem 0;
    color: #0b3556;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🌍 Tourism Analytics")
st.sidebar.caption("Explore, predict & get recommendations")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🎯 Visitor Mode Prediction", "⭐ Attraction Rating Prediction",
     "🧭 Recommendation", "🥥 Explore Insights"],
    label_visibility="collapsed"
)

if insights is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📌 Quick stats**")
    st.sidebar.markdown(f'<div class="sidebar-stat">📝 {insights["total_transactions"]:,} transactions</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="sidebar-stat">👥 {insights["total_users"]:,} users</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="sidebar-stat">📍 {insights["total_attractions_used"]} attractions with visit data</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="sidebar-stat">⭐ {insights["avg_rating"]:.2f} average rating</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Built by Devadharshini")

if load_errors:
    with st.sidebar.expander("Artifact details", expanded=True):
        for e in load_errors:
            st.write("•", e)

if "prep_error" in globals():
    st.error("Could not build the exact training preprocessing: " + prep_error)
    st.stop()

# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>🌍 Tourism Experience Analytics</h1>
        <p>Visitor Mode Prediction • Attraction Rating Prediction • Recommendation • Explore Insights</p>
        <p><b>Built by Devadharshini</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Overview")
    st.write(
        "An end-to-end tourism analytics application combining feature "
        "engineering, machine learning and attraction recommendation."
    )

    if insights is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", f'{insights["total_transactions"]:,}')
        c2.metric("Users", f'{insights["total_users"]:,}')
        c3.metric(
            "Attractions",
            f'{insights["total_attractions_used"]}',
            help=f'{insights["total_attractions_catalog"]:,} total in the attraction catalog; '
                 f'{insights["total_attractions_used"]} have recorded visits.'
        )
        c4.metric("Avg. Rating", f'{insights["avg_rating"]:.2f} / 5')
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", "52,930")
        c2.metric("Users", "33,530")
        c3.metric("Attractions", "1,698")
        c4.metric("Visit Modes", "5")

    st.markdown("### 🔬 Application Modules")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 🎯 Visitor Mode Prediction")
        st.write("Predicts Business, Couples, Family, Friends or Solo visit mode.")
    with b:
        st.markdown("#### ⭐ Attraction Rating Prediction")
        st.write("Predicts an expected attraction rating on a 1–5 scale.")
    with c:
        st.markdown("#### 🧭 Recommendation")
        st.write("Ranks similar attractions using content similarity and popularity.")

    with st.expander("ℹ️ What does each visit mode mean?"):
        st.markdown(
            "- **Business** — work-related travel\n"
            "- **Couples** — traveling as a pair/romantic trip\n"
            "- **Family** — traveling with family members, often including children\n"
            "- **Friends** — traveling in a group of friends\n"
            "- **Solo** — traveling alone"
        )

    if insights is not None:
        st.markdown("---")
        st.info("📊 Want the full charts — top attractions, rating distribution, "
                "seasonality, and an attraction-by-attraction explorer? Head to "
                "**🥥 Explore Insights** in the sidebar.")
    else:
        st.markdown("---")
        st.info(
            "Dataset insights are unavailable right now"
            + (f": {insights_error}" if "insights_error" in globals() else ".")
        )

    st.markdown("---")
    st.info(
        "The deployed app reproduces the saved model's training feature schema "
        "before calling prediction; it does not pass the four UI inputs directly to the model."
    )

# ============================================================
# CLASSIFICATION
# ============================================================
elif page == "🎯 Visitor Mode Prediction":
    st.title("🎯 Visitor Mode Prediction")
    st.write("Enter a visitor and trip context to predict the likely visit mode.")

    model = artifacts.get("clf")
    if model is None:
        st.error("Classification model is missing.")
        st.stop()

    df = prep["df"]
    users = sorted(df["UserId"].dropna().unique().astype(int).tolist())
    attrs = (
        df[["AttractionId", "Attraction"]]
        .dropna()
        .drop_duplicates("AttractionId")
        .sort_values("Attraction")
    )

    l, r = st.columns(2)
    with l:
        user_id = st.selectbox("User ID", users, index=users.index(14) if 14 in users else 0)
        year = st.number_input("Visit Year", 2013, 2022, 2018, 1)
    with r:
        attr_names = attrs["Attraction"].astype(str).tolist()
        default_attr = "Sacred Monkey Forest Sanctuary"
        attr_index = attr_names.index(default_attr) if default_attr in attr_names else 0
        selected_attr = st.selectbox("Attraction", attr_names, index=attr_index)
        month = st.slider("Visit Month", 1, 12, 12)

    with st.expander("📋 User & attraction snapshot", expanded=False):
        s1, s2 = st.columns(2)
        u_snap = user_history_snapshot(insights, user_id)
        with s1:
            st.markdown("**User**")
            if u_snap:
                st.markdown(
                    f'<div class="snapshot-card">'
                    f'📍 {u_snap["country"]}, {u_snap["region"]}, {u_snap["continent"]}<br>'
                    f'🧳 {u_snap["visit_count"]} past visit(s) in the dataset<br>'
                    f'⭐ {u_snap["avg_rating"]:.2f} average rating given'
                    f'</div>', unsafe_allow_html=True
                )
                if len(u_snap["mode_counts"]) > 0:
                    st.caption("Past visit modes:")
                    st.bar_chart(u_snap["mode_counts"])
            else:
                st.caption("No history available for this user.")
        a_snap = attraction_snapshot(insights, int(
            attrs.loc[attrs["Attraction"].astype(str) == selected_attr, "AttractionId"].iloc[0]
        ))
        with s2:
            st.markdown("**Attraction**")
            if a_snap:
                st.markdown(
                    f'<div class="snapshot-card">'
                    f'🏷️ {a_snap["type"]}<br>'
                    f'📌 {a_snap["address"]}<br>'
                    f'🧳 {a_snap["visit_count"]} recorded visits<br>'
                    f'⭐ {a_snap["avg_rating"]:.2f} average rating'
                    f'</div>', unsafe_allow_html=True
                )
                if len(a_snap["mode_counts"]) > 0:
                    st.caption("Who typically visits:")
                    st.bar_chart(a_snap["mode_counts"])
            else:
                st.caption("No history available for this attraction.")

    if st.button("Predict Visit Mode", type="primary", use_container_width=True):
        try:
            attraction_id = int(attrs.loc[attrs["Attraction"].astype(str) == selected_attr, "AttractionId"].iloc[0])
            _, Xclf = make_prediction_row(user_id, attraction_id, year, month)

            pred = int(model.predict(Xclf)[0])
            proba = model.predict_proba(Xclf)[0] if hasattr(model, "predict_proba") else None
            label = MODE_LABELS[pred]

            st.success(f"### Predicted Visit Mode: {label}")

            if proba is not None:
                prob_df = pd.DataFrame({
                    "Visit Mode": MODE_LABELS,
                    "Probability": proba
                }).sort_values("Probability", ascending=False)
                fig = px.bar(prob_df, x="Visit Mode", y="Probability",
                             title="Prediction Probability")
                fig.update_yaxes(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            a_snap = attraction_snapshot(insights, attraction_id)
            if a_snap and len(a_snap["mode_counts"]) > 0:
                top_hist_mode = a_snap["mode_counts"].idxmax()
                if top_hist_mode == label:
                    st.caption(
                        f"✅ This matches the most common historical visit mode for "
                        f"**{selected_attr}** ({top_hist_mode})."
                    )
                else:
                    st.caption(
                        f"ℹ️ Note: the most common historical visit mode for "
                        f"**{selected_attr}** is **{top_hist_mode}**, though the model "
                        f"predicts **{label}** for this specific user & trip context."
                    )

            st.caption(
                f"Model input shape: {Xclf.shape[1]} features — matching the saved XGBoost model."
            )

            show_also_like(selected_attr)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============================================================
# REGRESSION
# ============================================================
elif page == "⭐ Attraction Rating Prediction":
    st.title("⭐ Attraction Rating Prediction")
    st.write("Estimate the expected attraction rating on a 1–5 scale.")

    model = artifacts.get("reg")
    if model is None:
        st.error("Regression model is missing.")
        st.stop()

    df = prep["df"]
    users = sorted(df["UserId"].dropna().unique().astype(int).tolist())
    attrs = (
        df[["AttractionId", "Attraction"]]
        .dropna()
        .drop_duplicates("AttractionId")
        .sort_values("Attraction")
    )

    l, r = st.columns(2)
    with l:
        user_id = st.selectbox("User ID", users, index=users.index(14) if 14 in users else 0, key="reg_user")
        year = st.number_input("Visit Year", 2013, 2022, 2018, 1, key="reg_year")
    with r:
        attr_names = attrs["Attraction"].astype(str).tolist()
        default_attr = "Sacred Monkey Forest Sanctuary"
        attr_index = attr_names.index(default_attr) if default_attr in attr_names else 0
        selected_attr = st.selectbox("Attraction", attr_names, index=attr_index, key="reg_attr")
        month = st.slider("Visit Month", 1, 12, 12, key="reg_month")

    with st.expander("📋 User & attraction snapshot", expanded=False):
        s1, s2 = st.columns(2)
        u_snap = user_history_snapshot(insights, user_id)
        with s1:
            st.markdown("**User**")
            if u_snap:
                st.markdown(
                    f'<div class="snapshot-card">'
                    f'📍 {u_snap["country"]}, {u_snap["region"]}, {u_snap["continent"]}<br>'
                    f'🧳 {u_snap["visit_count"]} past visit(s) in the dataset<br>'
                    f'⭐ {u_snap["avg_rating"]:.2f} average rating given'
                    f'</div>', unsafe_allow_html=True
                )
            else:
                st.caption("No history available for this user.")
        a_snap = attraction_snapshot(insights, int(
            attrs.loc[attrs["Attraction"].astype(str) == selected_attr, "AttractionId"].iloc[0]
        ))
        with s2:
            st.markdown("**Attraction**")
            if a_snap:
                st.markdown(
                    f'<div class="snapshot-card">'
                    f'🏷️ {a_snap["type"]}<br>'
                    f'📌 {a_snap["address"]}<br>'
                    f'🧳 {a_snap["visit_count"]} recorded visits<br>'
                    f'⭐ {a_snap["avg_rating"]:.2f} average rating'
                    f'</div>', unsafe_allow_html=True
                )
            else:
                st.caption("No history available for this attraction.")

    if st.button("Predict Rating", type="primary", use_container_width=True):
        try:
            attraction_id = int(attrs.loc[attrs["Attraction"].astype(str) == selected_attr, "AttractionId"].iloc[0])
            Xreg, _ = make_prediction_row(user_id, attraction_id, year, month)
            pred = float(model.predict(Xreg)[0])
            pred = float(np.clip(pred, 1, 5))

            st.success(f"### Predicted Rating: {pred:.2f} / 5")
            st.progress(pred / 5)

            a_snap = attraction_snapshot(insights, attraction_id)
            if insights is not None and a_snap:
                compare_df = pd.DataFrame({
                    "Metric": ["Predicted Rating", f"{selected_attr}'s Historical Avg", "Overall Dataset Avg"],
                    "Rating": [pred, a_snap["avg_rating"], insights["avg_rating"]]
                })
                fig = px.bar(
                    compare_df, x="Metric", y="Rating", color="Metric",
                    title="Predicted Rating vs Historical Averages",
                    range_y=[0, 5]
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                attr_ratings = insights["df"].loc[
                    insights["df"]["AttractionId"] == attraction_id, "Rating"
                ]
                if len(attr_ratings) >= 5:
                    fig2 = px.histogram(
                        attr_ratings, nbins=5,
                        title=f"Historical Rating Distribution — {selected_attr}",
                        labels={"value": "Rating"}
                    )
                    fig2.add_vline(x=pred, line_dash="dash", line_color="orange",
                                    annotation_text="Predicted")
                    fig2.update_layout(showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)

            show_also_like(selected_attr)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============================================================
# RECOMMENDATION
# ============================================================
elif page == "🧭 Recommendation":
    st.title("🧭 Attraction Recommendation")
    st.write("Find similar attractions using the saved content-similarity matrix and popularity score.")

    rec = artifacts.get("rec_data")
    if rec is None:
        st.error("attraction_recommendation_data.pkl is missing.")
        st.stop()

    names = rec["Attraction"].astype(str).tolist()
    default_attr = "Sacred Monkey Forest Sanctuary"
    default_idx = names.index(default_attr) if default_attr in names else 0

    selected_attr = st.selectbox("Choose an attraction", names, index=default_idx)
    top_n = st.slider("Number of recommendations", 3, min(10, max(3, len(names)-1)), 7)

    sel_row = rec[rec["Attraction"].astype(str) == selected_attr]
    if not sel_row.empty:
        r0 = sel_row.iloc[0]
        loc_row = attraction_location[attraction_location["Attraction"].astype(str) == selected_attr]
        city_val = loc_row["CityName"].iloc[0] if not loc_row.empty else r0.get("CityName", "Unknown")
        country_val = loc_row["Country"].iloc[0] if not loc_row.empty else r0.get("Country", "Unknown")
        st.markdown(
            f'<div class="snapshot-card">'
            f'🏷️ {r0.get("AttractionType","Unknown")} &nbsp;|&nbsp; '
            f'📍 {city_val}, {country_val} &nbsp;|&nbsp; '
            f'🧳 {int(r0.get("VisitCount",0)):,} visits &nbsp;|&nbsp; '
            f'🔥 popularity score {float(r0.get("PopularityScore",0)):.2f}'
            f'</div>', unsafe_allow_html=True
        )
    if st.button("Recommend Attractions", type="primary", use_container_width=True):
        try:
            result = get_recommendations(selected_attr, top_n)
            st.success(f"### Top {len(result)} recommendations for {selected_attr}")
            st.dataframe(result, use_container_width=True, hide_index=True)

            if "AttractionType" in result.columns and len(result) > 0:
                type_counts = result["AttractionType"].value_counts().reset_index()
                type_counts.columns = ["AttractionType", "Count"]
                fig = px.bar(
                    type_counts, x="AttractionType", y="Count",
                    title="Attraction Types Among These Recommendations"
                )
                fig.update_xaxes(tickangle=-20)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Recommendation failed: {e}")

# ============================================================
# EXPLORE INSIGHTS
# ============================================================
elif page == "🥥 Explore Insights":
    st.title("🥥 Explore Insights")
    st.write(
        "Browse the full dataset before making a prediction — see what's "
        "popular, how ratings and visit modes break down, and dig into any "
        "single attraction's history."
    )

    if insights is None:
        st.info(
            "Dataset insights are unavailable right now"
            + (f": {insights_error}" if "insights_error" in globals() else ".")
        )
        st.stop()

    st.markdown("### 📊 Dataset-wide charts")
    d1, d2 = st.columns(2)
    with d1:
        fig = px.bar(
            insights["top_attractions"].sort_values("VisitCount"),
            x="VisitCount", y="Attraction", orientation="h",
            color="AvgRating", color_continuous_scale="Blues",
            title="Top 10 Most-Visited Attractions",
            labels={"VisitCount": "Visits", "AvgRating": "Avg Rating"}
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        fig = px.pie(
            insights["visit_mode_counts"], names="VisitMode", values="Count",
            title="Visit Mode Share", hole=0.45
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    d3, d4 = st.columns(2)
    with d3:
        fig = px.bar(
            insights["rating_counts"], x="Rating", y="Count",
            title="Rating Distribution (all transactions)",
            text="Count"
        )
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)
    with d4:
        fig = px.bar(
            insights["continent_counts"].sort_values("Users"),
            x="Users", y="Continent", orientation="h",
            title="Users by Continent"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📅 Visits by Month (seasonality)")
    fig = px.line(
        insights["monthly_counts"], x="Month", y="Visits", markers=True,
        title="Total Visits per Month (across all years)"
    )
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🏛️ Popularity by Attraction Type")
    fig = px.bar(
        insights["attraction_type_counts"],
        x="AttractionType", y="VisitCount",
        title="Visit Volume by Attraction Type"
    )
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔎 Attraction Explorer")
    st.write(
        "Pick any attraction to see its full profile — useful for deciding "
        "what to enter on the prediction or recommendation pages."
    )

    attr_options = sorted(insights["df"]["Attraction"].dropna().unique().tolist())
    picked = st.selectbox("Choose an attraction to explore", attr_options)

    picked_id = int(
        insights["df"].loc[insights["df"]["Attraction"] == picked, "AttractionId"].iloc[0]
    )
    snap = attraction_snapshot(insights, picked_id)
    if snap:
        e1, e2 = st.columns([1, 1])
        with e1:
            st.markdown(
                f'<div class="snapshot-card">'
                f'🏷️ {snap["type"]}<br>'
                f'📌 {snap["address"]}<br>'
                f'🧳 {snap["visit_count"]:,} recorded visits<br>'
                f'⭐ {snap["avg_rating"]:.2f} average rating'
                f'</div>', unsafe_allow_html=True
            )
            if len(snap["mode_counts"]) > 0:
                st.caption("Who typically visits:")
                st.bar_chart(snap["mode_counts"])
        with e2:
            st.caption("🔗 Similar attractions (from the Recommendation model):")
            try:
                similar = get_recommendations(picked, 5)
                st.dataframe(
                    similar[["Attraction", "AttractionType", "HybridScore"]],
                    use_container_width=True, hide_index=True
                )
            except Exception:
                st.caption("No recommendation data available for this attraction.")

st.markdown("---")
st.caption("Tourism Experience Analytics • Built by Devadharshini")
