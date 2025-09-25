from flask import Flask, render_template, request
import os, joblib, pickle
import pandas as pd

app = Flask(__name__)


# --------------------
# Helper to load artifacts
# --------------------
def load_artifacts(sign_type):
    model_dir = os.path.join("model", sign_type)
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))

    with open(os.path.join(model_dir, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)

    with open(os.path.join(model_dir, "model_results.pkl"), "rb") as f:
        model_results = pickle.load(f)

    return scaler, feature_names, model_results


# --------------------
# Routes
# --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    sign_types = {
        "Blade Sign": "blade_sign",
        "Flatcut Letters": "flatcut_letters",
        "Backlit Metal Signs": "backlit_metal_signs"
    }

    if request.method == "POST":
        # Collect form data
        sign_type = request.form.get("sign_type")
        model_name = request.form.get("model_name")
        width = float(request.form.get("width"))
        height = float(request.form.get("height"))
        depth = float(request.form.get("depth"))
        area = width * height

        # Load artifacts
        scaler, feature_names, model_results = load_artifacts(sign_type)

        # Selected model
        selected_model = model_results[model_name]["model"]
        selected_r2 = model_results[model_name]["R2"]

        # Scale input
        inputs = [width, height, depth, area]
        X_scaled = scaler.transform([inputs])
        prediction = selected_model.predict(X_scaled)[0]

        # Performance summary
        results_df = pd.DataFrame([
            {"Model": n, "MAE": r["MAE"], "RMSE": r["RMSE"], "R²": r["R2"]}
            for n, r in model_results.items()
        ]).sort_values("R²", ascending=False)

        results_df["MAE"] = results_df["MAE"].round(2)
        results_df["RMSE"] = results_df["RMSE"].round(2)
        results_df["R²"] = results_df["R²"].round(2)

        table_html = results_df.to_html(classes="table table-striped", index=False, escape=False)

        return render_template("results.html",
                               prediction=prediction,
                               model_name=model_name,
                               selected_r2=selected_r2,
                               sign_type=sign_type,
                               width=width, height=height, depth=depth, area=area,
                               models=list(model_results.keys()),
                               table_html=table_html)

    # GET request (first load)
    # Default: show Blade Sign models
    default_sign_type = "blade_sign"
    _, _, model_results = load_artifacts(default_sign_type)

    return render_template("index.html",
                           sign_types=sign_types,
                           default_sign_type=default_sign_type,
                           models=list(model_results.keys()))


if __name__ == "__main__":
    app.run(debug=True)
