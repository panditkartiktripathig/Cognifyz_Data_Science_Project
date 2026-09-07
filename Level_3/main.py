# ============================================================
# COGNIFYZ TECHNOLOGIES - DATA SCIENCE INTERNSHIP
# LEVEL 3 : Tasks 1, 2 and 3
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FILE = "Dataset.csv"   # Change if required
df = pd.read_csv(FILE)

# Standardize column names
df.columns = (
    df.columns
      .str.strip()
      .str.replace(" ", "_")
      .str.replace("/", "_")
      .str.replace("(", "", regex=False)
      .str.replace(")", "", regex=False)
)

# Convert numerical columns
for col in ["Aggregate_rating", "Votes", "Price_range",
            "Latitude", "Longitude", "Country_Code"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop_duplicates().copy()

# -----------------------------
# TASK 1: CUSTOMER PREFERENCE ANALYSIS
# -----------------------------
print("=" * 60)
print("LEVEL 3 - TASK 1: CUSTOMER PREFERENCE ANALYSIS")
print("=" * 60)

if {"Cuisines", "Votes"}.issubset(df.columns):

    # Split multiple cuisines
    cuisine_votes = (
        df[["Cuisines", "Votes"]]
        .dropna()
        .assign(Cuisines=lambda x: x["Cuisines"].str.split(","))
        .explode("Cuisines")
    )

    cuisine_votes["Cuisines"] = cuisine_votes["Cuisines"].str.strip()

    # Number of restaurants associated with each cuisine
    cuisine_restaurants = cuisine_votes["Cuisines"].value_counts()

    # Total votes associated with each cuisine
    popular_cuisines = (
        cuisine_votes.groupby("Cuisines")["Votes"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTop 15 cuisines by total votes:")
    print(popular_cuisines.head(15))

    plt.figure(figsize=(10, 6))
    popular_cuisines.head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 Cuisines by Total Customer Votes")
    plt.xlabel("Total Votes")
    plt.ylabel("Cuisine")
    plt.tight_layout()
    plt.show()

if {"Cuisines", "Aggregate_rating"}.issubset(df.columns):

    cuisine_rating = (
        df[["Cuisines", "Aggregate_rating"]]
        .dropna()
        .assign(Cuisines=lambda x: x["Cuisines"].str.split(","))
        .explode("Cuisines")
    )

    cuisine_rating["Cuisines"] = cuisine_rating["Cuisines"].str.strip()

    rating_by_cuisine = (
        cuisine_rating.groupby("Cuisines")["Aggregate_rating"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    # Avoid tiny samples dominating the result
    reliable_rating = rating_by_cuisine[rating_by_cuisine["count"] >= 20]

    print("\nTop cuisines by average rating (minimum 20 records):")
    print(reliable_rating.head(15))

    plt.figure(figsize=(10, 6))
    reliable_rating.head(10)["mean"].sort_values().plot(kind="barh")
    plt.title("Top 10 Cuisines by Average Rating")
    plt.xlabel("Average Aggregate Rating")
    plt.ylabel("Cuisine")
    plt.tight_layout()
    plt.show()

# -----------------------------
# TASK 2: DATA VISUALIZATION
# -----------------------------
print("\n" + "=" * 60)
print("LEVEL 3 - TASK 2: DATA VISUALIZATION")
print("=" * 60)

# Histogram
if "Aggregate_rating" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Aggregate_rating"].dropna(), bins=20, kde=True)
    plt.title("Distribution of Aggregate Ratings")
    plt.xlabel("Aggregate Rating")
    plt.ylabel("Number of Restaurants")
    plt.tight_layout()
    plt.show()

# Rating bar plot
if "Aggregate_rating" in df.columns:
    rating_counts = df["Aggregate_rating"].round(1).value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    rating_counts.plot(kind="bar")
    plt.title("Restaurant Ratings")
    plt.xlabel("Aggregate Rating")
    plt.ylabel("Number of Restaurants")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Average rating by city
if {"City", "Aggregate_rating"}.issubset(df.columns):
    city_stats = (
        df.groupby("City")["Aggregate_rating"]
        .agg(["count", "mean"])
    )
    city_stats = city_stats[city_stats["count"] >= 20]
    city_stats = city_stats.sort_values("mean", ascending=False).head(10)

    print("\nTop cities by average rating (minimum 20 restaurants):")
    print(city_stats)

    plt.figure(figsize=(10, 6))
    city_stats["mean"].sort_values().plot(kind="barh")
    plt.title("Top Cities by Average Restaurant Rating")
    plt.xlabel("Average Rating")
    plt.ylabel("City")
    plt.tight_layout()
    plt.show()

# Votes vs rating
if {"Votes", "Aggregate_rating"}.issubset(df.columns):
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Votes",
        y="Aggregate_rating",
        alpha=0.35
    )
    plt.title("Votes vs Aggregate Rating")
    plt.xlabel("Votes")
    plt.ylabel("Aggregate Rating")
    plt.tight_layout()
    plt.show()

    print("\nCorrelation between Votes and Aggregate Rating:")
    print(df[["Votes", "Aggregate_rating"]].corr())

# Price range vs rating
if {"Price_range", "Aggregate_rating"}.issubset(df.columns):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Price_range", y="Aggregate_rating")
    plt.title("Aggregate Rating by Price Range")
    plt.xlabel("Price Range")
    plt.ylabel("Aggregate Rating")
    plt.tight_layout()
    plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=np.number)

if len(numeric_df.columns) >= 2:
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# -----------------------------
# TASK 3: PREDICTIVE MODELING
# -----------------------------
print("\n" + "=" * 60)
print("LEVEL 3 - TASK 3: PREDICTIVE MODELING")
print("=" * 60)

target = "Aggregate_rating"

if target not in df.columns:
    raise ValueError("Aggregate_rating column was not found.")

# Useful predictors from the available dataset
preferred_features = [
    "Country_Code",
    "City",
    "Latitude",
    "Longitude",
    "Votes",
    "Price_range",
    "Has_Table_booking",
    "Has_Online_delivery",
    "Restaurant_Name_Length",
    "Address_Length",
    "Cuisine_Count"
]

# Feature engineering if needed
if "Restaurant_Name" in df.columns and "Restaurant_Name_Length" not in df.columns:
    df["Restaurant_Name_Length"] = (
        df["Restaurant_Name"].fillna("").astype(str).str.len()
    )

if "Address" in df.columns and "Address_Length" not in df.columns:
    df["Address_Length"] = (
        df["Address"].fillna("").astype(str).str.len()
    )

if "Cuisines" in df.columns and "Cuisine_Count" not in df.columns:
    df["Cuisine_Count"] = (
        df["Cuisines"]
        .fillna("")
        .astype(str)
        .apply(lambda x: len([c for c in x.split(",") if c.strip()]))
    )

features = [c for c in preferred_features if c in df.columns]

if len(features) < 2:
    raise ValueError("Not enough usable predictor columns were found.")

model_df = df[features + [target]].copy()

# Remove rows where target is unavailable
model_df = model_df.dropna(subset=[target])

X = model_df[features]
y = model_df[target]

print("\nFeatures used:")
print(features)
print("\nNumber of observations:", len(model_df))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_features = X.select_dtypes(include=np.number).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=12,
        min_samples_leaf=5
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_depth=20,
        min_samples_leaf=2
    )
}

results = []
predictions = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    predictions[name] = pred

results_df = pd.DataFrame(results).sort_values("RMSE")

print("\nMODEL COMPARISON:")
print(results_df.round(4))

# Best model
best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model based on lowest RMSE: {best_model_name}")

# Actual vs predicted for best model
best_pred = predictions[best_model_name]

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_pred, alpha=0.4)

min_value = min(y_test.min(), best_pred.min())
max_value = max(y_test.max(), best_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value]
)

plt.title(f"Actual vs Predicted Ratings - {best_model_name}")
plt.xlabel("Actual Aggregate Rating")
plt.ylabel("Predicted Aggregate Rating")
plt.tight_layout()
plt.show()

# Residual plot
residuals = y_test - best_pred

plt.figure(figsize=(8, 5))
plt.scatter(best_pred, residuals, alpha=0.4)
plt.axhline(0, linestyle="--")
plt.title(f"Residual Plot - {best_model_name}")
plt.xlabel("Predicted Rating")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()

# Save results
results_df.to_csv("Level_3_Model_Results.csv", index=False)
print("\nSaved: Level_3_Model_Results.csv")

print("\nLEVEL 3 COMPLETE.")
