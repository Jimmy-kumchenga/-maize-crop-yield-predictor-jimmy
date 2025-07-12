import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Malawi Maize Yield Predictor 🌽", layout="centered")
st.title("🌽 Malawi Maize Yield Predictor")
st.markdown("Predict **maize yield (kg/ha)** based on simple farm details.")

# -----------------------------
# Load and filter dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_malawi_maize.csv")
    df = df[(df["Year"] >= 2011) & (df["Year"] <= 2025)]  # Only recent data
    return df

df = load_data()

# -----------------------------
# Optional: Show data sample
# -----------------------------
if st.checkbox("📊 Show sample training data used"):
    st.write(df.head())

# -----------------------------
# Preprocess
# -----------------------------
df = pd.get_dummies(df, columns=["Maize_Type"], drop_first=True)
X = df.drop("Yield_kg_ha", axis=1)
y = df["Yield_kg_ha"]

# -----------------------------
# Train model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Input form
# -----------------------------
st.subheader("📋 Enter Farm Details")

year = st.selectbox("Year", sorted(df["Year"].unique()), index=len(df["Year"].unique()) - 1)
maize_type = st.selectbox("Maize Type", ["Local", "Hybrid"])
area = st.slider("Area Cultivated (ha)", 0.1, 10.0, 1.5, 0.1)
rainfall = st.slider("Estimated Rainfall (mm)", 500, 1500, 1000, 10)
temp = st.slider("Estimated Avg Temperature (°C)", 20.0, 30.0, 25.0, 0.1)
fert = st.slider("Fertilizer Usage (kg/ha)", 0, 200, 80, 5)

# -----------------------------
# Format input for prediction
# -----------------------------
input_data = pd.DataFrame([{
    "Year": year,
    "Area_Cultivated_ha": area,
    "Rainfall_mm": rainfall,
    "Avg_Temp_C": temp,
    "Fertilizer_kg_ha": fert,
    "Maize_Type_Hybrid": 1 if maize_type == "Hybrid" else 0
}])

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Yield"):
    prediction = model.predict(input_data)[0]
    st.success(f"🌾 Estimated Yield: **{prediction:.2f} kg/ha**")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit · Data: Synthetic for demonstration purposes only.")
