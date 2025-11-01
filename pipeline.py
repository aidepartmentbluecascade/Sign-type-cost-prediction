import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
import joblib
import sklearn


class SignCostPredictor:
    def __init__(self):
        self.depth_map = {
            'Blade Sign': 2.0,
            'Flatcut Letters': 0.25,
            'Halo lit Chanel letter sign': 1.0,
            'Fabricated Letters- Nonlit': 1.0,
            'UV Acrylic Sign': 0.75,
            'Halo lit Chanel letter sign with backboard': 3.0,
            'Facelit Channel letter': 5.0,
            'Lightbox': 5.0,
            'Push-thru Cabinet Sign': 2.3,
            'Facelit and Halo lit Channel letter': 3.5
        }
        self.target_signs = list(self.depth_map.keys())
        self.model_dir = "models"
        self.shipping_model = None
        self.production_model = None
        self.df = None

    # ---------------------------------------------------------------
    def load_and_prepare_data(self, file_path='top10_signs.xlsx'):
        """Load & clean dataset"""
        df = pd.read_excel(file_path)
        df = df[df['sign_type'].isin(self.target_signs)].copy()

        # Derived features
        df['depth'] = df['sign_type'].map(self.depth_map)
        df['area'] = df['width'] * df['height']

        # Clean invalid rows
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df[(df['width'] > 0) & (df['height'] > 0) &
                (df['shipping_cost'] > 0) & (df['production_cost'] > 0)]

        # Convert numerics to float32 to reduce memory
        df = df.astype({
            'width': 'float32',
            'height': 'float32',
            'depth': 'float32',
            'area': 'float32',
            'shipping_cost': 'float32',
            'production_cost': 'float32'
        })

        print(f"📊 Clean data loaded: {len(df)} rows, memory optimized.")
        self.df = df
        return df

    # ---------------------------------------------------------------
    def _build_pipeline(self):
        """Build lightweight ML pipeline compatible with all sklearn versions"""
        categorical = ['sign_type']
        numeric = ['width', 'height', 'depth', 'area']

        # Detect scikit-learn version for safe OneHotEncoder setup
        skl_version = tuple(map(int, sklearn.__version__.split(".")[:2]))
        print(f"🧩 Detected scikit-learn version: {skl_version}")

        try:
            if skl_version >= (1, 4):
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
                print("✅ Using sparse_output=True (new sklearn syntax).")
            else:
                encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)
                print("✅ Using sparse=True (legacy sklearn syntax).")
        except TypeError:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)
            print("✅ Fallback to sparse=True (TypeError caught).")

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric),
            ('cat', encoder, categorical)
        ])

        # Small, efficient RandomForest
        model = RandomForestRegressor(
            n_estimators=40,
            max_depth=10,
            min_samples_split=6,
            n_jobs=-1,
            random_state=42
        )

        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

    # ---------------------------------------------------------------
    def train_models(self, df):
        """Train compact RandomForest models"""
        X = df[['sign_type', 'width', 'height', 'depth', 'area']]
        y_ship = df['shipping_cost']
        y_prod = df['production_cost']

        X_train, X_test, y_train_s, y_test_s = train_test_split(
            X, y_ship, test_size=0.2, random_state=42
        )
        X_train2, X_test2, y_train_p, y_test_p = train_test_split(
            X, y_prod, test_size=0.2, random_state=42
        )

        print("🧠 Training lightweight shipping model...")
        ship = self._build_pipeline()
        ship.fit(X_train, y_train_s)

        print("🧠 Training lightweight production model...")
        prod = self._build_pipeline()
        prod.fit(X_train2, y_train_p)

        # Evaluate
        rmse_s = np.sqrt(mean_squared_error(y_test_s, ship.predict(X_test)))
        rmse_p = np.sqrt(mean_squared_error(y_test_p, prod.predict(X_test2)))
        print(f"📦 RMSE Shipping: {rmse_s:.2f} | 🏭 Production: {rmse_p:.2f}")

        self.shipping_model, self.production_model = ship, prod
        self._save_models()
        return ship, prod

    # ---------------------------------------------------------------
    def _model_paths(self):
        os.makedirs(self.model_dir, exist_ok=True)
        return (
            os.path.join(self.model_dir, "shipping_model.pkl"),
            os.path.join(self.model_dir, "production_model.pkl")
        )

    def _save_models(self):
        """Save models using ultra compression (XZ)"""
        ship_path, prod_path = self._model_paths()
        joblib.dump(self.shipping_model, ship_path, compress=("xz", 9))
        joblib.dump(self.production_model, prod_path, compress=("xz", 9))
        print("💾 Models saved with XZ compression (GitHub-friendly).")

    def _load_models(self):
        """Load models if already trained"""
        ship_path, prod_path = self._model_paths()
        if os.path.exists(ship_path) and os.path.exists(prod_path):
            self.shipping_model = joblib.load(ship_path)
            self.production_model = joblib.load(prod_path)
            print("✅ Loaded lightweight models from disk.")
            return True
        return False

    # ---------------------------------------------------------------
    def predict_costs(self, sign_type, width, height):
        """Predict shipping and production costs"""
        if not self.shipping_model or not self.production_model:
            raise RuntimeError("Models not trained or loaded.")
        if sign_type not in self.depth_map:
            raise ValueError(f"Unknown sign type: {sign_type}")

        depth = self.depth_map[sign_type]
        area = width * height
        X = pd.DataFrame([[sign_type, width, height, depth, area]],
                         columns=['sign_type', 'width', 'height', 'depth', 'area'])
        s = self.shipping_model.predict(X)[0]
        p = self.production_model.predict(X)[0]
        return {
            'shipping_cost': round(float(s), 2),
            'production_cost': round(float(p), 2),
            'total_cost': round(float(s + p), 2)
        }

    def find_actual(self, sign_type, width, height):
        """Find exact cost match in dataset"""
        if self.df is None or self.df.empty:
            return None
        match = self.df[
            (self.df['sign_type'] == sign_type) &
            (self.df['width'] == width) &
            (self.df['height'] == height)
        ]
        if match.empty:
            return None
        row = match.iloc[0]
        return {
            'shipping_cost': round(float(row['shipping_cost']), 2),
            'production_cost': round(float(row['production_cost']), 2),
            'total_cost': round(float(row['shipping_cost'] + row['production_cost']), 2)
        }


# ---------------------------------------------------------------
def initialize_predictor():
    predictor = SignCostPredictor()
    df = predictor.load_and_prepare_data()
    if predictor._load_models():
        return predictor
    print("🧩 No saved models found — training new lightweight models...")
    predictor.train_models(df)
    return predictor
