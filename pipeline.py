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
            'Halo lit Chanel letter sign': 1.0
        }
        self.shipping_model = None
        self.production_model = None
        self.model_dir = "models"
        self.df = None
        
    def load_and_prepare_data(self, file_path='rawData.xlsx'):
        """Load and prepare the dataset for training"""
        # Load dataset
        df = pd.read_excel('rawData.xlsx')
        
        # Filter dataset to keep only target sign types
        target_signs = ['Flatcut Letters', 'Blade Sign', 'Halo lit Chanel letter sign']
        df = df[df['sign_type'].isin(target_signs)].copy()
        
        # Add depth feature
        df['depth'] = df['sign_type'].map(self.depth_map)
        
        # Add area feature
        df['area'] = df['width'] * df['height']
        
        # Drop rows with missing values
        df.dropna(inplace=True)
        
        # Store prepared dataframe for exact match lookups
        self.df = df.copy()
        return df
    
    def train_models(self, df):
        """Train the machine learning models"""
        # Separate features and targets
        X = df[['sign_type', 'width', 'height', 'depth', 'area']]
        y = df[['shipping_cost', 'production_cost']]
        
        # Data augmentation - oversample whole dataset to 2x size by sampling with replacement
        X_aug = resample(X, replace=True, n_samples=len(X)*2, random_state=42)
        y_aug = y.loc[X_aug.index]
        
        # Define preprocessing pipeline
        categorical_features = ['sign_type']
        numerical_features = ['width', 'height', 'depth', 'area']
        
        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), numerical_features),
                         ('cat', OneHotEncoder(drop='first'), categorical_features)]
        )
        
        # Define regression model
        model = RandomForestRegressor(random_state=42)
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', model)])
        
        # Hyperparameter grid for tuning
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [10, 20, None],
            'regressor__min_samples_split': [2, 5]
        }
        
        # Split augmented data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)
        
        # Grid search for shipping cost
        self.shipping_model = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        self.shipping_model.fit(X_train, y_train['shipping_cost'])
        
        # Grid search for production cost
        self.production_model = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        self.production_model.fit(X_train, y_train['production_cost'])
        
        # Evaluate models
        y_pred_shipping = self.shipping_model.predict(X_test)
        y_pred_production = self.production_model.predict(X_test)
        
        rmse_shipping = np.sqrt(mean_squared_error(y_test['shipping_cost'], y_pred_shipping))
        rmse_production = np.sqrt(mean_squared_error(y_test['production_cost'], y_pred_production))
        
        print(f'Shipping Cost RMSE: {rmse_shipping:.2f}')
        print(f'Production Cost RMSE: {rmse_production:.2f}')
        
        # Persist trained models for faster future loads
        self._save_models()
        return self.shipping_model, self.production_model

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

    def _load_models(self) -> bool:
        ship_path, prod_path = self._model_paths()
        if os.path.exists(ship_path) and os.path.exists(prod_path):
            self.shipping_model = joblib.load(ship_path)
            self.production_model = joblib.load(prod_path)
            return True
        return False

    def find_actual(self, sign_type: str, width: float, height: float):
        """Return actual costs from dataset if the combination exactly matches a row."""
        if self.df is None or self.df.empty:
            return None
        try:
            # Exact match on sign_type, width, height
            m = (
                (self.df['sign_type'] == sign_type) &
                (self.df['width'] == width) &
                (self.df['height'] == height)
            )
            if not m.any():
                return None
            row = self.df.loc[m].iloc[0]
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
        """Predict shipping and production costs for given parameters"""
        if not self.shipping_model or not self.production_model:
            raise ValueError("Models not trained yet. Call train_models() first.")
        
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

# Initialize and train the predictor
def initialize_predictor():
    """Initialize and train the sign cost predictor"""
    predictor = SignCostPredictor()
    df = predictor.load_and_prepare_data()
    # Prefer loading persisted models for speed; train if not available
    if predictor._load_models():
        print("Loaded models from disk.")
    else:
        predictor.train_models(df)
    return predictor
