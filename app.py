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

def p(name):
    return os.path.join(BASE_DIR, FILES[name])

@st.cache_resource
def load_artifacts():
    out, errors = {}, []
    for key, fn in FILES.items():
        fp = p(key)
        if not os.path.exists(fp):
            errors.append(f"{fn} not found")
            continue
        try:
            out[key] = joblib.load(fp)
        except Exception:
            try:
                with open(fp, "rb") as f:
                    out[key] = pickle.load(f)
            except Exception as e:
                errors.append(f"{fn}: {e}")
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

prep = None
try:
    pobj = artifacts.get("prep_objects")
    if pobj is None:
        raise RuntimeError(
            "preprocessing_objects.pkl is missing. This file stores the exact "
            "fitted OneHotEncoder/StandardScaler and lookup tables from training "
            "and is required for correct predictions."
        )

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
# Styling
# ------------------------------------------------------------
st.markdown("""
<style>
.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    color: white;
    margin-bottom: 1.5rem;
}
.card {
    padding: 1rem;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,.25);
    background: rgba(128,128,128,.05);
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🌍 Tourism Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🎯 Classification", "⭐ Regression", "🧭 Recommendation"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Built by Devadharshini")

if load_errors:
    with st.sidebar.expander("Artifact details"):
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
        <p>Classification • Rating Prediction • Attraction Recommendation</p>
        <p><b>Built by Devadharshini</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Overview")
    st.write(
        "An end-to-end tourism analytics application combining feature "
        "engineering, machine learning and attraction recommendation."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", "52,930")
    c2.metric("Users", "33,530")
    c3.metric("Attractions", "1,698")
    c4.metric("Visit Modes", "5")

    st.markdown("### 🔬 Application Modules")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 🎯 Classification")
        st.write("Predicts Business, Couples, Family, Friends or Solo visit mode.")
    with b:
        st.markdown("#### ⭐ Regression")
        st.write("Predicts an expected attraction rating on a 1–5 scale.")
    with c:
        st.markdown("#### 🧭 Recommendation")
        st.write("Ranks similar attractions using content similarity and popularity.")

    st.markdown("---")
    st.info(
        "The deployed app reproduces the saved model's training feature schema "
        "before calling prediction; it does not pass the four UI inputs directly to the model."
    )

# ============================================================
# CLASSIFICATION
# ============================================================
elif page == "🎯 Classification":
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

            st.caption(
                f"Model input shape: {Xclf.shape[1]} features — matching the saved XGBoost model."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============================================================
# REGRESSION
# ============================================================
elif page == "⭐ Regression":
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

    if st.button("Predict Rating", type="primary", use_container_width=True):
        try:
            attraction_id = int(attrs.loc[attrs["Attraction"].astype(str) == selected_attr, "AttractionId"].iloc[0])
            Xreg, _ = make_prediction_row(user_id, attraction_id, year, month)
            pred = float(model.predict(Xreg)[0])
            pred = float(np.clip(pred, 1, 5))

            st.success(f"### Predicted Rating: {pred:.2f} / 5")
            st.progress(pred / 5)
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

    if st.button("Recommend Attractions", type="primary", use_container_width=True):
        try:
            result = get_recommendations(selected_attr, top_n)
            st.success(f"### Top {len(result)} recommendations for {selected_attr}")
            st.dataframe(result, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Recommendation failed: {e}")

st.markdown("---")
st.caption("Tourism Experience Analytics • Built by Devadharshini")
