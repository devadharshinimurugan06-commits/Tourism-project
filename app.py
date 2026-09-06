import os
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# TOURISM EXPERIENCE ANALYTICS
# Built by Devadharshini
# ============================================================

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "clf": "best_classification_model.pkl",
    "reg": "best_regression_model.pkl",
    "label_encoder": "classification_label_encoder.pkl",
    "classes": "classification_classes.pkl",
    "preprocess": "preprocessing_objects.pkl",
    "tfidf": "tfidf_vectorizer.pkl",
    "cosine": "cosine_similarity.pkl",
    "rec_data": "attraction_recommendation_data.pkl",
}

def path(name):
    return os.path.join(BASE_DIR, FILES[name])

# -----------------------------
# Load saved artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    errors = []

    for key in ["clf", "reg", "label_encoder", "classes", "preprocess",
                "tfidf", "cosine", "rec_data"]:
        p = path(key)
        if not os.path.exists(p):
            errors.append(f"{FILES[key]} not found")
            continue
        try:
            if key == "rec_data":
                artifacts[key] = pd.read_pickle(p)
            else:
                artifacts[key] = joblib.load(p)
        except Exception:
            try:
                with open(p, "rb") as f:
                    artifacts[key] = pickle.load(f)
            except Exception as e:
                errors.append(f"{FILES[key]} could not be loaded: {e}")

    return artifacts, errors

artifacts, load_errors = load_artifacts()

# -----------------------------
# Load repository datasets
# -----------------------------
@st.cache_data
def load_data():
    names = [
        "Transaction.xlsx", "User.xlsx", "City.xlsx",
        "Country.xlsx", "Region.xlsx", "Continent.xlsx",
        "Type.xlsx", "Mode.xlsx", "Updated_Item.xlsx"
    ]
    data = {}
    for name in names:
        p = os.path.join(BASE_DIR, name)
        if os.path.exists(p):
            try:
                data[name] = pd.read_excel(p)
            except Exception:
                data[name] = None
        else:
            data[name] = None
    return data

raw = load_data()

# -----------------------------
# Build master data for UI
# -----------------------------
@st.cache_data
def build_master(raw):
    transaction = raw.get("Transaction.xlsx")
    user = raw.get("User.xlsx")
    city = raw.get("City.xlsx")
    country = raw.get("Country.xlsx")
    region = raw.get("Region.xlsx")
    continent = raw.get("Continent.xlsx")
    attr_type = raw.get("Type.xlsx")
    mode = raw.get("Mode.xlsx")
    item = raw.get("Updated_Item.xlsx")

    if transaction is None:
        return pd.DataFrame()

    df = transaction.copy()

    # Original project uses VisitMode as an ID in Transaction
    if "VisitMode" in df.columns and "VisitModeId" not in df.columns:
        df = df.rename(columns={"VisitMode": "VisitModeId"})

    if user is not None and "UserId" in user.columns:
        df = df.merge(user, on="UserId", how="left")

    if city is not None and "CityId" in df.columns:
        df = df.merge(city, on="CityId", how="left", suffixes=("", "_city"))

    if country is not None and "CountryId" in df.columns:
        df = df.merge(country, on="CountryId", how="left", suffixes=("", "_country"))

    if region is not None and "RegionId" in df.columns:
        df = df.merge(region, on="RegionId", how="left", suffixes=("", "_region"))

    if continent is not None and "ContinentId" in df.columns:
        df = df.merge(continent, on="ContinentId", how="left", suffixes=("", "_continent"))

    if item is not None and "AttractionId" in df.columns:
        df = df.merge(item, on="AttractionId", how="left", suffixes=("", "_item"))

    if attr_type is not None and "AttractionTypeId" in df.columns:
        left = df["AttractionTypeId"].astype(str)
        at = attr_type.copy()
        at["AttractionTypeId"] = at["AttractionTypeId"].astype(str)
        df["AttractionTypeId"] = left
        df = df.merge(at, on="AttractionTypeId", how="left", suffixes=("", "_type"))

    if mode is not None and "VisitModeId" in df.columns:
        md = mode.copy()
        md["VisitModeId"] = md["VisitModeId"].astype(str)
        df["VisitModeId"] = df["VisitModeId"].astype(str)
        df = df.merge(md[["VisitModeId", "VisitMode"]], on="VisitModeId",
                      how="left", suffixes=("", "_mode"))

    return df

master = build_master(raw)

# -----------------------------
# Helper: model prediction
# -----------------------------
def find_transformer(obj, preferred_words):
    """Find a preprocessing object inside a saved dictionary."""
    if hasattr(obj, "transform"):
        return obj

    if isinstance(obj, dict):
        # Prefer keys that look like the requested transformer.
        for key, value in obj.items():
            key_l = str(key).lower()
            if any(w in key_l for w in preferred_words) and hasattr(value, "transform"):
                return value

        for value in obj.values():
            if hasattr(value, "transform"):
                return value

    return None


def prepare_for_model(model, prep, row, kind):
    """
    Tries the safest saved-artifact routes:
    1) A complete sklearn Pipeline.
    2) A saved preprocessor/transformer.
    3) A feature-name based DataFrame.
    """
    # Complete Pipeline: model handles preprocessing itself.
    if hasattr(model, "named_steps") and hasattr(model, "predict"):
        return row

    # Some saved models expose feature_names_in_.
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        missing = [c for c in cols if c not in row.columns]
        if not missing:
            return row[cols]

    # Saved preprocessing object.
    words = ["classification", "class", "clf"] if kind == "classification" else [
        "regression", "reg", "preprocessor", "transform"
    ]
    transformer = find_transformer(prep, words)

    if transformer is not None:
        # ColumnTransformer/Pipeline can transform the raw DataFrame.
        try:
            return transformer.transform(row)
        except Exception:
            pass

    # Try any transformer in the saved dictionary.
    if isinstance(prep, dict):
        for value in prep.values():
            if hasattr(value, "transform"):
                try:
                    return value.transform(row)
                except Exception:
                    continue

    # Last attempt: raw row.
    return row


def predict_model(model, prep, row, kind):
    X = prepare_for_model(model, prep, row, kind)
    try:
        pred = model.predict(X)
    except Exception as e:
        raise RuntimeError(
            "The saved model and saved preprocessing artifacts do not accept "
            "the current app input directly. Please verify the exact training "
            "preprocessing artifact saved in preprocessing_objects.pkl. "
            f"Original error: {e}"
        )

    probability = None
    if kind == "classification" and hasattr(model, "predict_proba"):
        try:
            probability = model.predict_proba(X)[0]
        except Exception:
            probability = None

    return pred, probability

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.hero {
    padding: 2rem 2rem 1.5rem 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 { margin-bottom: 0.2rem; }
.card {
    padding: 1rem;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,.25);
    background: rgba(128,128,128,.05);
}
.small { opacity: .75; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🌍 Tourism Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🎯 Classification", "⭐ Regression", "🧭 Recommendation"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Built by Devadharshini")

if load_errors:
    st.sidebar.warning("Some saved artifacts could not be loaded.")
    with st.sidebar.expander("Artifact details"):
        for e in load_errors:
            st.write("•", e)

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
        "An end-to-end tourism analytics application that combines data "
        "preprocessing, machine learning and recommendation techniques to "
        "analyze visitor behaviour, estimate attraction ratings and recommend "
        "similar attractions."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Transactions", "52,930")
    c2.metric("Users", "33,530")
    c3.metric("Attractions", "1,698")
    c4.metric("Visit Modes", "5")

    st.markdown("### 🔬 What this application does")

    a, b, c = st.columns(3)

    with a:
        st.markdown("#### 🎯 Classification")
        st.write("Predicts the visitor's mode of travel such as Couples, Family, Friends, Solo or Business.")

    with b:
        st.markdown("#### ⭐ Regression")
        st.write("Predicts the expected attraction rating on a 1–5 scale.")

    with c:
        st.markdown("#### 🧭 Recommendation")
        st.write("Ranks similar attractions using content similarity and popularity.")

    st.markdown("---")
    st.markdown("### 🔄 End-to-End Workflow")
    st.info(
        "Raw Tourism Data → Data Cleaning → Feature Engineering → "
        "Preprocessing → Classification / Regression / Recommendation → Streamlit"
    )

    st.markdown("### 🤖 Saved Models")
    model_cols = st.columns(4)
    model_cols[0].success("✓ Classification model")
    model_cols[1].success("✓ Regression model")
    model_cols[2].success("✓ Recommendation artifacts")
    model_cols[3].success("✓ Preprocessing artifacts")

    st.caption(
        "The application uses the saved artifacts from the project rather than retraining models inside Streamlit."
    )

# ============================================================
# CLASSIFICATION
# ============================================================
elif page == "🎯 Classification":

    st.title("🎯 Visitor Mode Prediction")
    st.write("Predict the likely visit mode from the saved classification model.")

    model = artifacts.get("clf")
    prep = artifacts.get("preprocess")

    if model is None:
        st.error("best_classification_model.pkl is not available.")
        st.stop()

    if master.empty:
        st.error("Repository datasets are required for the current model input workflow.")
        st.stop()

    # Use existing user and attraction information so categorical values
    # remain consistent with the training data.
    users = sorted(master["UserId"].dropna().unique().tolist())
    attractions = master[["AttractionId", "Attraction"]].drop_duplicates("AttractionId")
    attractions = attractions.dropna(subset=["AttractionId"])

    left, right = st.columns(2)

    with left:
        user_id = st.selectbox("User ID", users)
        year = st.number_input("Visit Year", min_value=2013, max_value=2022, value=2022, step=1)

    with right:
        selected_attr = st.selectbox(
            "Attraction",
            attractions["Attraction"].astype(str).tolist()
        )
        month = st.slider("Visit Month", 1, 12, 10)

    if st.button("Predict Visit Mode", type="primary", use_container_width=True):

        # Take the user's known profile and attraction's known profile
        # from the repository data.
        user_rows = master[master["UserId"] == user_id]
        attr_rows = master[master["Attraction"].astype(str) == selected_attr]

        if user_rows.empty or attr_rows.empty:
            st.error("Could not construct the requested input from repository data.")
        else:
            base = attr_rows.iloc[0].copy()

            # Overlay user profile where available.
            for col in [
                "ContinentId", "RegionId", "CountryId", "CityId",
                "CityName", "Country", "Region", "Continent"
            ]:
                if col in user_rows.columns and col in base.index:
                    base[col] = user_rows.iloc[0][col]

            base["UserId"] = user_id
            base["AttractionId"] = base.get("AttractionId", np.nan)
            base["VisitYear"] = year
            base["VisitMonth"] = month

            row = pd.DataFrame([base])

            # Add feature-engineering fields used in the project.
            row["VisitQuarter"] = ((row["VisitMonth"] - 1) // 3) + 1
            season_map = {
                1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
                5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
                9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter"
            }
            row["VisitSeason"] = row["VisitMonth"].map(season_map)
            if "AttractionAddress" in row.columns:
                row["HasAttractionAddress"] = (
                    row["AttractionAddress"].fillna("-").astype(str).ne("-").astype(int)
                )

            try:
                pred, proba = predict_model(model, prep, row, "classification")

                pred_value = pred[0]

                # Decode if the model returns numeric labels.
                encoder = artifacts.get("label_encoder")
                classes = artifacts.get("classes")

                if np.issubdtype(np.asarray(pred_value).dtype, np.number):
                    idx = int(pred_value)
                    if encoder is not None and hasattr(encoder, "inverse_transform"):
                        try:
                            predicted_label = encoder.inverse_transform([idx])[0]
                        except Exception:
                            predicted_label = str(classes[idx]) if classes is not None else str(idx)
                    elif classes is not None and idx < len(classes):
                        predicted_label = classes[idx]
                    else:
                        predicted_label = str(idx)
                else:
                    predicted_label = str(pred_value)

                st.success(f"### Predicted Visit Mode: {predicted_label}")

                if proba is not None:
                    class_names = None
                    if classes is not None:
                        class_names = [str(x) for x in classes]
                    elif encoder is not None and hasattr(encoder, "classes_"):
                        class_names = [str(x) for x in encoder.classes_]

                    if class_names and len(class_names) == len(proba):
                        prob_df = pd.DataFrame({
                            "Visit Mode": class_names,
                            "Probability": proba
                        }).sort_values("Probability", ascending=False)

                        fig = px.bar(
                            prob_df,
                            x="Visit Mode",
                            y="Probability",
                            title="Prediction Probability"
                        )
                        fig.update_yaxes(range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(str(e))
                st.info(
                    "The saved model files are present, but the exact preprocessing "
                    "schema needs to match the training-time artifact."
                )

# ============================================================
# REGRESSION
# ============================================================
elif page == "⭐ Regression":

    st.title("⭐ Attraction Rating Prediction")
    st.write("Estimate the expected attraction rating on a 1–5 scale.")

    model = artifacts.get("reg")
    prep = artifacts.get("preprocess")

    if model is None:
        st.error("best_regression_model.pkl is not available.")
        st.stop()

    if master.empty:
        st.error("Repository datasets are required for the current model input workflow.")
        st.stop()

    users = sorted(master["UserId"].dropna().unique().tolist())
    attractions = master[["AttractionId", "Attraction"]].drop_duplicates("AttractionId")
    attractions = attractions.dropna(subset=["AttractionId"])

    left, right = st.columns(2)

    with left:
        user_id = st.selectbox("User ID", users, key="reg_user")
        year = st.number_input("Visit Year", min_value=2013, max_value=2022, value=2022, step=1, key="reg_year")

    with right:
        selected_attr = st.selectbox(
            "Attraction",
            attractions["Attraction"].astype(str).tolist(),
            key="reg_attr"
        )
        month = st.slider("Visit Month", 1, 12, 10, key="reg_month")

    if st.button("Predict Rating", type="primary", use_container_width=True):

        user_rows = master[master["UserId"] == user_id]
        attr_rows = master[master["Attraction"].astype(str) == selected_attr]

        if user_rows.empty or attr_rows.empty:
            st.error("Could not construct the requested input.")
        else:
            base = attr_rows.iloc[0].copy()

            for col in [
                "ContinentId", "RegionId", "CountryId", "CityId",
                "CityName", "Country", "Region", "Continent"
            ]:
                if col in user_rows.columns and col in base.index:
                    base[col] = user_rows.iloc[0][col]

            base["UserId"] = user_id
            base["VisitYear"] = year
            base["VisitMonth"] = month

            row = pd.DataFrame([base])
            row["VisitQuarter"] = ((row["VisitMonth"] - 1) // 3) + 1

            season_map = {
                1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
                5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
                9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter"
            }
            row["VisitSeason"] = row["VisitMonth"].map(season_map)

            if "AttractionAddress" in row.columns:
                row["HasAttractionAddress"] = (
                    row["AttractionAddress"].fillna("-").astype(str).ne("-").astype(int)
                )

            try:
                pred, _ = predict_model(model, prep, row, "regression")
                rating = float(np.asarray(pred).ravel()[0])
                rating = float(np.clip(rating, 1, 5))

                st.success(f"### Predicted Rating: {rating:.2f} / 5")

                st.progress(rating / 5)

                if rating >= 4:
                    st.info("High predicted satisfaction")
                elif rating >= 3:
                    st.info("Moderate predicted satisfaction")
                else:
                    st.info("Lower predicted satisfaction")

            except Exception as e:
                st.error(str(e))
                st.info(
                    "The saved regression model is present, but the exact "
                    "training-time preprocessing schema must match."
                )

# ============================================================
# RECOMMENDATION
# ============================================================
else:

    st.title("🧭 Attraction Recommendation System")
    st.write(
        "Hybrid recommendation using attraction content similarity and popularity."
    )

    rec = artifacts.get("rec_data")
    cosine = artifacts.get("cosine")

    if rec is None or cosine is None:
        st.error(
            "Recommendation artifacts are missing. "
            "Expected attraction_recommendation_data.pkl and cosine_similarity.pkl."
        )
        st.stop()

    rec = rec.copy()

    if "Attraction" not in rec.columns:
        st.error("Attraction column is missing from recommendation data.")
        st.stop()

    attraction_names = rec["Attraction"].dropna().astype(str).drop_duplicates().tolist()

    selected = st.selectbox(
        "Choose an attraction",
        attraction_names
    )

    top_n = st.slider("Number of recommendations", 5, 10, 10)

    if st.button("Recommend Attractions", type="primary", use_container_width=True):

        matches = rec.index[
            rec["Attraction"].astype(str).str.lower() == selected.lower()
        ].tolist()

        if not matches:
            st.warning("Attraction not found.")
        else:
            idx = matches[0]

            try:
                similarity_scores = np.asarray(cosine)[idx]

                # Same hybrid idea used during recommendation development:
                # 80% content similarity + 20% popularity.
                if "PopularityScore" in rec.columns:
                    popularity = pd.to_numeric(
                        rec["PopularityScore"], errors="coerce"
                    ).fillna(0).to_numpy()
                elif "VisitCount" in rec.columns:
                    visits = pd.to_numeric(
                        rec["VisitCount"], errors="coerce"
                    ).fillna(0).to_numpy()
                    mx = visits.max()
                    popularity = visits / mx if mx > 0 else np.zeros(len(visits))
                else:
                    popularity = np.zeros(len(rec))

                hybrid = 0.80 * similarity_scores + 0.20 * popularity

                order = np.argsort(hybrid)[::-1]

                rows = []
                for i in order:
                    if i == idx:
                        continue

                    row = {
                        "Attraction": rec.iloc[i].get("Attraction", ""),
                        "AttractionType": rec.iloc[i].get("AttractionType", ""),
                        "City": rec.iloc[i].get("CityName", rec.iloc[i].get("City", "")),
                        "Country": rec.iloc[i].get("Country", ""),
                        "VisitCount": rec.iloc[i].get("VisitCount", 0),
                        "SimilarityScore": float(similarity_scores[i]),
                        "PopularityScore": float(popularity[i]),
                        "RecommendationScore": float(hybrid[i]),
                    }
                    rows.append(row)

                    if len(rows) == top_n:
                        break

                result = pd.DataFrame(rows)

                st.success(f"Top {len(result)} recommendations for {selected}")

                st.dataframe(
                    result.style.format({
                        "SimilarityScore": "{:.3f}",
                        "PopularityScore": "{:.3f}",
                        "RecommendationScore": "{:.3f}",
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            except Exception as e:
                st.error(f"Recommendation could not be generated: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Tourism Experience Analytics • Built by Devadharshini")
