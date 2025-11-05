



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import os
import joblib


class SignCostPredictor:
    def __init__(self):
        self.depth_map = {
            'Blade Sign': 2.0,
            'Flatcut Letters': 0.25,
            'Halo lit Channel letter sign': 1.0,
            'Fabricated Letters- Nonlit': 1.0,
            'UV Acrylic Sign': 0.75,
            'Halo lit Channel letter sign with backboard': 3.0,
            'Facelit Channel letter': 5.0,
            'Lightbox': 5.0,
            'Push-thru Cabinet Sign': 2.3,
            'Facelit and Halo lit Channel letter': 3.5,
            'Facelit Channel letter with raceway' : 5.5,
            'Halo lit Channel letter sign with Raceway' : 3.0,
            'Vinyl Banner' : 0.25,
            'Neon Sign' : 0.5,
            'Facelit Channel letter with backboard' : 5.0,
            'Open Face Neon Letters' : 3.0,
            'Panel Sign' : 0.5,
            'Flatcut Letters with backboard' : 2.5,
            'Facelit and Halo lit Channel Letters with Backboard' : 5.5 
        }
        self.target_signs = list(self.depth_map.keys())
        self.shipping_model = None
        self.production_model = None
        self.model_dir = "models"
        self.df = None

    # ---------------------------------------------------------------
    def load_and_prepare_data(self, file_path='top10_signs.xlsx', augment=False):
        """Load and prepare dataset for training"""
        df = pd.read_excel(file_path)

        # Filter dataset for target sign types only
        df = df[df['sign_type'].isin(self.target_signs)].copy()

        # Derived features
        df['depth'] = df['sign_type'].map(self.depth_map)
        df['area'] = df['width'] * df['height']

        # Remove invalid or missing rows
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df[(df['width'] > 0) & (df['height'] > 0) &
                (df['shipping_cost'] > 0) & (df['production_cost'] > 0)]

        if augment:
            print("🚀 Performing traditional data augmentation...")
            df_aug = self._augment_traditional(df)
            print(f"✅ Added {len(df_aug)} synthetic rows.")
            df = pd.concat([df, df_aug], ignore_index=True)

            # ✅ Save augmented dataset to Excel
            output_file = "top10_signs_augmented.xlsx"
            df.to_excel(output_file, index=False)
            print(f"📁 Augmented dataset saved as: {output_file}")

        self.df = df.copy()
        return df

    # ---------------------------------------------------------------
    def _augment_traditional(self, df):
        """Generate synthetic samples using statistical scaling and noise"""
        augmented_rows = []
        np.random.seed(42)

        for _, row in df.iterrows():
            sign_type = row['sign_type']
            width = row['width']
            height = row['height']
            shipping = row['shipping_cost']
            production = row['production_cost']

            # ✅ Skip invalid rows
            if width <= 0 or height <= 0 or shipping <= 0 or production <= 0:
                continue

            for _ in range(5):  # create 5 new samples per record
                width_factor = np.random.uniform(0.9, 1.2)
                height_factor = np.random.uniform(0.9, 1.2)

                new_width = width * width_factor
                new_height = height * height_factor

                old_area = width * height
                new_area = new_width * new_height
                if old_area == 0:
                    continue  # avoid divide-by-zero

                area_factor = new_area / old_area

                # Add small random Gaussian noise (±5%)
                noise_ship = np.random.normal(1, 0.05)
                noise_prod = np.random.normal(1, 0.05)

                new_shipping = shipping * area_factor * noise_ship
                new_production = production * area_factor * noise_prod

                # ✅ Skip invalid or extreme
                if new_shipping <= 0 or new_production <= 0:
                    continue

                augmented_rows.append({
                    'sign_type': sign_type,
                    'width': round(new_width, 2),
                    'height': round(new_height, 2),
                    'shipping_cost': round(new_shipping, 2),
                    'production_cost': round(new_production, 2)
                })

        # Convert to DataFrame and clean
        aug_df = pd.DataFrame(augmented_rows)
        if aug_df.empty:
            print("⚠️ Warning: No augmented data generated.")
            return pd.DataFrame()

        aug_df['depth'] = aug_df['sign_type'].map(self.depth_map)
        aug_df['area'] = aug_df['width'] * aug_df['height']

        # Remove invalid data
        aug_df = aug_df.replace([np.inf, -np.inf], np.nan).dropna()
        aug_df = aug_df[(aug_df['width'] > 0) & (aug_df['height'] > 0) &
                        (aug_df['shipping_cost'] > 0) & (aug_df['production_cost'] > 0)]
        return aug_df

    # ---------------------------------------------------------------
    def train_models(self, df):
        """Train ML models for shipping and production cost"""
        X = df[['sign_type', 'width', 'height', 'area']]
        y = df[['shipping_cost', 'production_cost']]

        # Oversampling for balanced training
        X_aug = resample(X, replace=True, n_samples=len(X) * 2, random_state=42)
        y_aug = y.loc[X_aug.index]

        categorical_features = ['sign_type']
        numerical_features = ['width', 'height', 'area']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ]
        )

        model = RandomForestRegressor(random_state=42)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [10, 20, None],
            'regressor__min_samples_split': [2, 5]
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X_aug, y_aug, test_size=0.2, random_state=42
        )

        # Train for shipping cost
        self.shipping_model = GridSearchCV(
            pipeline, param_grid, cv=3,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        self.shipping_model.fit(X_train, y_train['shipping_cost'])

        # Train for production cost
        self.production_model = GridSearchCV(
            pipeline, param_grid, cv=3,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        self.production_model.fit(X_train, y_train['production_cost'])

        # Evaluate models
        y_pred_shipping = self.shipping_model.predict(X_test)
        y_pred_production = self.production_model.predict(X_test)

        rmse_shipping = np.sqrt(mean_squared_error(y_test['shipping_cost'], y_pred_shipping))
        rmse_production = np.sqrt(mean_squared_error(y_test['production_cost'], y_pred_production))

       

        self._save_models()
        return self.shipping_model, self.production_model

    # ---------------------------------------------------------------
    def _model_paths(self):
        os.makedirs(self.model_dir, exist_ok=True)
        return (
            os.path.join(self.model_dir, "shipping_model.pkl"),
            os.path.join(self.model_dir, "production_model.pkl"),
        )

    def _save_models(self):
        ship_path, prod_path = self._model_paths()
        joblib.dump(self.shipping_model, ship_path)
        joblib.dump(self.production_model, prod_path)

    def _load_models(self):
        ship_path, prod_path = self._model_paths()
        if os.path.exists(ship_path) and os.path.exists(prod_path):
            self.shipping_model = joblib.load(ship_path)
            self.production_model = joblib.load(prod_path)
            return True
        return False

    # ---------------------------------------------------------------
    def find_actual(self, sign_type: str, width: float, height: float):
        """Return actual costs if exact match exists in dataset"""
        if self.df is None or self.df.empty:
            return None
        try:
            mask = (
                (self.df['sign_type'] == sign_type) &
                (self.df['width'] == width) &
                (self.df['height'] == height)
            )
            if not mask.any():
                return None
            row = self.df.loc[mask].iloc[0]
            actual_shipping = float(row['shipping_cost'])
            actual_production = float(row['production_cost'])
            return {
                'shipping_cost': round(actual_shipping, 2),
                'production_cost': round(actual_production, 2),
                'total_cost': round(actual_shipping + actual_production, 2)
            }
        except Exception:
            return None

    def predict_costs(self, sign_type, width, height):
        """Predict shipping and production cost"""
        if not self.shipping_model or not self.production_model:
            raise ValueError("Models not trained yet. Call train_models() first.")

        if sign_type not in self.depth_map:
            raise ValueError(f"Unknown sign type: {sign_type}")

        depth = self.depth_map[sign_type]
        area = width * height
        input_df = pd.DataFrame([[sign_type, width, height, depth, area]],
                                columns=['sign_type', 'width', 'height', 'depth', 'area'])

        shipping_cost = self.shipping_model.predict(input_df)[0]
        production_cost = self.production_model.predict(input_df)[0]
        total_cost = shipping_cost + production_cost

        return {
            'shipping_cost': round(shipping_cost, 2),
            'production_cost': round(production_cost, 2),
            'total_cost': round(total_cost, 2)
        }


# ---------------------------------------------------------------
def initialize_predictor():
    predictor = SignCostPredictor()
    df = predictor.load_and_prepare_data(augment=True)  # ✅ performs augmentation and saves augmented file
    if predictor._load_models():
        print("✅ Loaded models from disk.")
    else:
        print("🧠 Training models from scratch...")
        predictor.train_models(df)
    return predictor

