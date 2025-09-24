import streamlit as st
import joblib
import pickle
import numpy as np
import os
import pandas as pd

# ====================
# LOAD ARTIFACTS FUNCTION
# ====================
def load_artifacts(sign_type):
    try:
        model_dir = os.path.join("model", sign_type)

        scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))

        with open(os.path.join(model_dir, "feature_names.pkl"), "rb") as f:
            feature_names = pickle.load(f)

        with open(os.path.join(model_dir, "model_results.pkl"), "rb") as f:
            model_results = pickle.load(f)

        return scaler, feature_names, model_results
    except Exception as e:
        st.error(f"❌ Error loading artifacts for {sign_type}: {e}")
        return None, None, None

# ====================
# STREAMLIT UI
# ====================
st.title("📦 Shipping Cost Predictor 🤖")

# --------------------
# SELECT SIGN TYPE
# --------------------

sign_types = {
    "Blade Sign": "blade_sign",
    "Flatcut Letters": "flatcut_letters",
    "Backlit Metal Signs": "halolit_channel_letters"
}
sign_type_label = st.selectbox("Choose Sign Type", list(sign_types.keys()))
sign_type = sign_types[sign_type_label]

scaler, feature_names, model_results = load_artifacts(sign_type)

if scaler and feature_names and model_results:

    # --------------------
    # SELECT MODEL
    # --------------------

    results_df = pd.DataFrame([
        {"Model": name, "MAE": r["MAE"], "RMSE": r["RMSE"], "R²": r["R2"]}
        for name, r in model_results.items()
    ]).sort_values("R²", ascending=False)

    model_name = st.selectbox("Select Model", results_df["Model"].tolist())
    selected_model = model_results[model_name]["model"]
    selected_r2 = model_results[model_name]["R2"]

    st.info(f"✅ You selected **{model_name}** with R² = {selected_r2:.3f}")

    # --------------------
    # USER INPUT FORM
    # --------------------
    st.header("📝 Enter Sign Details")
    col1, col2 = st.columns(2)

    with col1:
        width = st.number_input("Width (in)", min_value=1.0, max_value=500.0, value=30.0, step=1.0)
        height = st.number_input("Height (in)", min_value=1.0, max_value=500.0, value=20.0, step=1.0)

    with col2:
        depth = st.number_input("Depth (in)", min_value=1.0, max_value=5.0, value=1.0, step=1.0)

    # Auto-calc area
    area = width * height
    st.number_input("Sign Area (sq.ft)", value=area, disabled=True)

    # Match training order: ['width', 'height', 'depth', 'Sign Area (sq.ft)']
    inputs = [width, height, depth, area]

    if st.button("🔮 Predict Cost"):
        X_scaled = scaler.transform([inputs])
        prediction = selected_model.predict(X_scaled)[0]
        st.success(f"💰 Estimated Shipping Cost: **${prediction:.2f}**")

    # --------------------
    # MODEL PERFORMANCE SUMMARY
    # --------------------
    st.header("📊 Model Performance Summary")
    st.dataframe(results_df.style.format({
        "MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.3f}"
    }))

else:
    st.warning("⚠️ Please run the training notebook for this sign type to generate models.")



















