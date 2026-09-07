# ============================================================
# COGNIFYZ TECHNOLOGIES - DATA SCIENCE INTERNSHIP
# LEVEL 1 : Tasks 1, 2 and 3
# Dataset: Zomato restaurant dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional for the map task:
# pip install folium
try:
    import folium
except ImportError:
    folium = None

# -----------------------------
# 0. LOAD DATA
# -----------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "Dataset.csv"

df = pd.read_csv(FILE)


print("=" * 60)
print("DATASET LOADED")
print("=" * 60)
print("Rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# Standardize column names
df.columns = (
    df.columns
      .str.strip()
      .str.replace(" ", "_")
      .str.replace("/", "_")
      .str.replace("(", "", regex=False)
      .str.replace(")", "", regex=False)
)

# -----------------------------
# TASK 1: DATA EXPLORATION
# -----------------------------
print("\n" + "=" * 60)
print("LEVEL 1 - TASK 1: DATA EXPLORATION AND PREPROCESSING")
print("=" * 60)

print("\nShape:", df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False))

# Convert common numeric columns
numeric_candidates = [
    "Aggregate_rating", "Votes", "Price_range",
    "Latitude", "Longitude", "Country_Code"
]

for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove duplicate rows
before = len(df)
df = df.drop_duplicates().copy()
print(f"\nDuplicate rows removed: {before - len(df)}")

# Handle missing values
# Numeric -> median
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

# Categorical -> mode
for col in df.select_dtypes(include="object").columns:
    if df[col].isnull().any():
        mode = df[col].mode()
        if len(mode) > 0:
            df[col] = df[col].fillna(mode.iloc[0])

print("\nMissing values after preprocessing:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

# Target distribution
if "Aggregate_rating" in df.columns:
    print("\nAggregate rating statistics:")
    print(df["Aggregate_rating"].describe())

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Aggregate_rating"], bins=20, kde=True)
    plt.title("Distribution of Aggregate Restaurant Ratings")
    plt.xlabel("Aggregate Rating")
    plt.ylabel("Number of Restaurants")
    plt.tight_layout()
    plt.show()

    # Rating categories to inspect possible imbalance
    bins = [-0.01, 2, 3, 4, 5]
    labels = ["0-2", "2-3", "3-4", "4-5"]
    rating_groups = pd.cut(df["Aggregate_rating"], bins=bins, labels=labels)
    print("\nRating group distribution:")
    print(rating_groups.value_counts().sort_index())
    print("\nRating group percentage:")
    print((rating_groups.value_counts(normalize=True).sort_index() * 100).round(2))

# -----------------------------
# TASK 2: DESCRIPTIVE ANALYSIS
# -----------------------------
print("\n" + "=" * 60)
print("LEVEL 1 - TASK 2: DESCRIPTIVE ANALYSIS")
print("=" * 60)

print("\nNumerical descriptive statistics:")
print(df.describe(include=[np.number]).T)

# Mean, median, standard deviation
num_cols = df.select_dtypes(include=np.number).columns
summary = pd.DataFrame({
    "Mean": df[num_cols].mean(),
    "Median": df[num_cols].median(),
    "Std_Dev": df[num_cols].std()
}).sort_index()

print("\nMean / Median / Standard Deviation:")
print(summary)

# Country
if "Country_Code" in df.columns:
    print("\nTop countries by number of restaurants:")
    print(df["Country_Code"].value_counts().head(10))

# City
if "City" in df.columns:
    print("\nTop 10 cities by number of restaurants:")
    print(df["City"].value_counts().head(10))

    plt.figure(figsize=(10, 6))
    df["City"].value_counts().head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 Cities by Number of Restaurants")
    plt.xlabel("Number of Restaurants")
    plt.ylabel("City")
    plt.tight_layout()
    plt.show()

# Cuisines
if "Cuisines" in df.columns:
    cuisine_counts = (
        df["Cuisines"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
    )

    print("\nTop 15 cuisines:")
    print(cuisine_counts.head(15))

    plt.figure(figsize=(10, 6))
    cuisine_counts.head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 Cuisines")
    plt.xlabel("Restaurant Count")
    plt.ylabel("Cuisine")
    plt.tight_layout()
    plt.show()

# -----------------------------
# TASK 3: GEOSPATIAL ANALYSIS
# -----------------------------
print("\n" + "=" * 60)
print("LEVEL 1 - TASK 3: GEOSPATIAL ANALYSIS")
print("=" * 60)

if {"Latitude", "Longitude"}.issubset(df.columns):
    geo = df[["Latitude", "Longitude"]].copy()

    print("\nLatitude summary:")
    print(geo["Latitude"].describe())

    print("\nLongitude summary:")
    print(geo["Longitude"].describe())

    plt.figure(figsize=(10, 7))
    plt.scatter(
        df["Longitude"],
        df["Latitude"],
        s=8,
        alpha=0.35
    )
    plt.title("Geographical Distribution of Restaurants")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

    # Create an interactive map if folium is installed
    if folium is not None:
        sample = df.dropna(subset=["Latitude", "Longitude"]).copy()
        sample = sample.sample(min(2000, len(sample)), random_state=42)

        center = [
            sample["Latitude"].mean(),
            sample["Longitude"].mean()
        ]

        restaurant_map = folium.Map(location=center, zoom_start=2)

        for _, row in sample.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=2,
                popup=str(row.get("Restaurant_Name", "Restaurant")),
                fill=True
            ).add_to(restaurant_map)

        restaurant_map.save("restaurant_locations.html")
        print("\nInteractive map saved as: restaurant_locations.html")
    else:
        print("\nInstall folium for the interactive map: pip install folium")

# Location vs rating
if {"City", "Aggregate_rating"}.issubset(df.columns):
    city_rating = (
        df.groupby("City")["Aggregate_rating"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
    )

    print("\nAverage rating for cities with at least 50 restaurants:")
    print(city_rating[city_rating["count"] >= 50].sort_values("mean", ascending=False).head(15))

if {"Country_Code", "Aggregate_rating"}.issubset(df.columns):
    country_rating = (
        df.groupby("Country_Code")["Aggregate_rating"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
    )

    print("\nAverage rating by country (minimum 20 restaurants):")
    print(country_rating[country_rating["count"] >= 20].sort_values("mean", ascending=False))

print("\nLEVEL 1 COMPLETE.")
