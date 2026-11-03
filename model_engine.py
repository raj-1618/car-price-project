import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# All 16 Brands present in cars_24_combined.csv with Vector Logos & Mechanical Tiers
BRAND_CATALOG = {
    "Maruti": {"logo": "https://cdn.simpleicons.org/suzuki/FF5500", "tier": "Market Leader • Economy"},
    "Hyundai": {"logo": "https://cdn.simpleicons.org/hyundai/00F2FE", "tier": "Smart Tech • Hatch/SUV"},
    "Honda": {"logo": "https://cdn.simpleicons.org/honda/EF4444", "tier": "i-VTEC Reliability"},
    "Renault": {"logo": "https://cdn.simpleicons.org/renault/FACC15", "tier": "Compact Utility"},
    "Tata": {"logo": "https://cdn.simpleicons.org/tata/38BDF8", "tier": "5-Star Safety • SUV"},
    "Volkswagen": {"logo": "https://cdn.simpleicons.org/volkswagen/60A5FA", "tier": "German TSI Engineering"},
    "Ford": {"logo": "https://cdn.simpleicons.org/ford/3B82F6", "tier": "EcoSport / Performance"},
    "Toyota": {"logo": "https://cdn.simpleicons.org/toyota/F43F5E", "tier": "Bulletproof Durability"},
    "Mahindra": {"logo": "https://cdn.simpleicons.org/mahindra/FB7185", "tier": "4x4 Off-Road & SUV"},
    "Datsun": {"logo": "https://cdn.simpleicons.org/nissan/A855F7", "tier": "Entry Hatchback"},
    "Skoda": {"logo": "https://cdn.simpleicons.org/skoda/4ADE80", "tier": "European Luxury Sedan"},
    "Kia": {"logo": "https://cdn.simpleicons.org/kia/FB923C", "tier": "Modern Connected SUV"},
    "Nissan": {"logo": "https://cdn.simpleicons.org/nissan/E2E8F0", "tier": "Japanese Crossover"},
    "Jeep": {"logo": "https://cdn.simpleicons.org/jeep/22C55E", "tier": "Trail Rated 4x4"},
    "Mg": {"logo": "https://cdn.simpleicons.org/mg/F87171", "tier": "Connected Premium SUV"},
    "Bmw": {"logo": "https://cdn.simpleicons.org/bmw/38BDF8", "tier": "Ultimate Driving Luxury"},
}


def load_cars24_dataset(uploaded_file=None, current_year: int = 2026) -> pd.DataFrame:
    """Loads and cleans cars_24_combined.csv with exact column standardization."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists("cars_24_combined.csv"):
        df = pd.read_csv("cars_24_combined.csv")
    else:
        raise FileNotFoundError("Please place 'cars_24_combined.csv' in the project root folder or upload via sidebar.")

    # Drop unnecessary index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Drop rows missing core target or input features (1 row in cars_24_combined.csv)
    df = df.dropna(subset=["Car Name", "Year", "Distance", "Fuel", "Price"]).copy()

    # Clean & engineer syllabus features
    df["Car Name"] = df["Car Name"].astype(str).str.strip()
    df["Brand"] = df["Car Name"].str.split().str[0].str.title()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["Age_of_Vehicle"] = (current_year - df["Year"]).clip(lower=1)
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(int)
    df["Fuel"] = df["Fuel"].astype(str).str.strip().str.title()
    df["Drive"] = df["Drive"].fillna("Manual").astype(str).str.strip().str.title()
    df["Type"] = df["Type"].fillna("HatchBack").astype(str).str.strip()
    df["Owner"] = pd.to_numeric(df["Owner"], errors="coerce").fillna(1).astype(int)
    df["Location"] = df["Location"].fillna("Unknown").astype(str)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").astype(int)

    # Filter out extreme data-entry typos (e.g., 971,212 km outlier or 0 km entries)
    df_clean = df[(df["Distance"] > 100) & (df["Distance"] <= 250000)].copy()

    return df_clean


def train_cars24_models(df: pd.DataFrame, mode: str = "full"):
    """
    Trains Linear Regression & Random Forest Regressor on cars_24_combined.csv.
    - mode='core': Uses strictly the 4 syllabus features (Brand, Distance, Fuel, Age_of_Vehicle)
    - mode='full': Uses Brand, Car Name, Distance, Fuel, Age_of_Vehicle, Drive, Type, Owner
    """
    if mode == "core":
        cat_cols = ["Brand", "Fuel"]
        num_cols = ["Distance", "Age_of_Vehicle"]
    else:
        cat_cols = ["Brand", "Car Name", "Fuel", "Drive", "Type"]
        num_cols = ["Distance", "Age_of_Vehicle", "Owner"]

    features = cat_cols + num_cols
    X = df[features]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, max_depth=18, random_state=42, n_jobs=-1)),
    ])

    lr_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    lr_preds = lr_pipeline.predict(X_test)
    rf_preds = rf_pipeline.predict(X_test)

    metrics = {
        "Linear Regression": {
            "MAE": mean_absolute_error(y_test, lr_preds),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, lr_preds))),
            "R2": r2_score(y_test, lr_preds),
            "preds": lr_preds,
        },
        "Random Forest Regressor": {
            "MAE": mean_absolute_error(y_test, rf_preds),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, rf_preds))),
            "R2": r2_score(y_test, rf_preds),
            "preds": rf_preds,
        },
    }

    return {
        "lr_model": lr_pipeline,
        "rf_model": rf_pipeline,
        "metrics": metrics,
        "y_test": y_test.values,
        "X_test": X_test,
        "features": features,
    }