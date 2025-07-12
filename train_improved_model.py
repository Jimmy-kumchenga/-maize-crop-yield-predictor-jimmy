# train_improved_rf_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score
import joblib

# 📂 Load your synthetic Malawi maize dataset
df = pd.read_csv('synthetic_malawi_maize.csv')

# 📌 Define input and target columns
feature_columns = ['Year', 'Maize_Type', 'Region', 'Soil_Quality', 'Fertilizer_Type',
                   'Irrigated', 'Crop_Rotation', 'Farmer_Experience', 'Area_ha',
                   'Rainfall_mm', 'Avg_Temp_C', 'Fertilizer_kg_ha']
target_column = 'Yield_kg_ha'

X = df[feature_columns]
y = df[target_column]

# 🧹 Preprocessing
numeric_features = ['Year', 'Farmer_Experience', 'Area_ha', 'Rainfall_mm', 'Avg_Temp_C', 'Fertilizer_kg_ha']
categorical_features = ['Maize_Type', 'Region', 'Soil_Quality', 'Fertilizer_Type']
binary_features = ['Irrigated', 'Crop_Rotation']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('bin', 'passthrough', binary_features)
    ]
)

# 🌲 Define the model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=42))
])

# 🔍 Hyperparameter tuning (optional)
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 12],
    'rf__min_samples_leaf': [2, 4],
    'rf__max_features': [0.5, 1.0]
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# ✅ Evaluate
y_pred = grid.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"✅ Random Forest R² (hold-out): {r2:.2f}")
print("💾 Saving model as 'improved_rf_yield_model.pkl'...")

# 💾 Save the best model
joblib.dump(grid.best_estimator_, 'improved_rf_yield_model.pkl')
