# ============================================================
# COGNIFYZ TECHNOLOGIES - DATA SCIENCE INTERNSHIP
# LEVEL 2 : Tasks 1, 2 and 3
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILE = "Dataset.csv"   # Change if required
df = pd.read_csv(FILE)

# Standardize names
df.columns = (
    df.columns
      .str.strip()
      .str.replace(" ", "_")
      .str.replace("/", "_")
      .str.replace("(", "", regex=False)
      .str.replace(")", "", regex=False)
)

# Convert relevant columns
for col in ["Aggregate_rating", "Votes", "Price_range"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop_duplicates().copy()

# -----------------------------
# TASK 1: TABLE BOOKING & ONLINE DELIVERY
# -----------------------------
print("=" * 60)
print("LEVEL 2 - TASK 1: TABLE BOOKING AND ONLINE DELIVERY")
print("=" * 60)

if "Has_Table_booking" in df.columns:
    table_booking_pct = (
        df["Has_Table_booking"].value_counts(normalize=True) * 100
    ).round(2)

    print("\nTable booking percentage:")
    print(table_booking_pct)

    table_avg = df.groupby("Has_Table_booking")["Aggregate_rating"].mean()
    print("\nAverage rating by table booking:")
    print(table_avg)

if "Has_Online_delivery" in df.columns:
    delivery_pct = (
        df["Has_Online_delivery"].value_counts(normalize=True) * 100
    ).round(2)

    print("\nOnline delivery percentage:")
    print(delivery_pct)

if {"Price_range", "Has_Online_delivery"}.issubset(df.columns):
    delivery_by_price = pd.crosstab(
        df["Price_range"],
        df["Has_Online_delivery"],
        normalize="index"
    ) * 100

    print("\nOnline delivery availability by price range (%):")
    print(delivery_by_price.round(2))

    delivery_by_price.plot(
        kind="bar",
        figsize=(9, 5)
    )
    plt.title("Online Delivery Availability by Price Range")
    plt.xlabel("Price Range")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# -----------------------------
# TASK 2: PRICE RANGE ANALYSIS
# -----------------------------
print("\n" + "=" * 60)
print("LEVEL 2 - TASK 2: PRICE RANGE ANALYSIS")
print("=" * 60)

if "Price_range" in df.columns:
    common_price = df["Price_range"].mode()
    print("\nMost common price range:")
    print(common_price.iloc[0] if len(common_price) else "Not available")

    price_counts = df["Price_range"].value_counts().sort_index()
    print("\nRestaurant count by price range:")
    print(price_counts)

    avg_rating_price = (
        df.groupby("Price_range")["Aggregate_rating"]
        .mean()
        .sort_index()
    )

    print("\nAverage rating for each price range:")
    print(avg_rating_price.round(3))

    highest_price_rating = avg_rating_price.idxmax()
    highest_rating = avg_rating_price.max()

    print(
        f"\nHighest average rating: Price Range {highest_price_rating} "
        f"with average rating {highest_rating:.3f}"
    )

    plt.figure(figsize=(8, 5))
    avg_rating_price.plot(kind="bar")
    plt.title("Average Rating by Price Range")
    plt.xlabel("Price Range")
    plt.ylabel("Average Aggregate Rating")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# -----------------------------
# TASK 3: FEATURE ENGINEERING
# -----------------------------
print("\n" + "=" * 60)
print("LEVEL 2 - TASK 3: FEATURE ENGINEERING")
print("=" * 60)

# Restaurant name length
if "Restaurant_Name" in df.columns:
    df["Restaurant_Name_Length"] = (
        df["Restaurant_Name"].fillna("").astype(str).str.len()
    )

# Address length
if "Address" in df.columns:
    df["Address_Length"] = (
        df["Address"].fillna("").astype(str).str.len()
    )

# Cuisine count
if "Cuisines" in df.columns:
    df["Cuisine_Count"] = (
        df["Cuisines"]
        .fillna("")
        .astype(str)
        .apply(lambda x: len([c for c in x.split(",") if c.strip()]))
    )

# Binary encoding
binary_map = {
    "Yes": 1,
    "No": 0,
    "yes": 1,
    "no": 0
}

if "Has_Table_booking" in df.columns:
    df["Has_Table_Booking_Encoded"] = (
        df["Has_Table_booking"].map(binary_map)
    )

if "Has_Online_delivery" in df.columns:
    df["Has_Online_Delivery_Encoded"] = (
        df["Has_Online_delivery"].map(binary_map)
    )

# Votes per rating point (useful engineered feature)
if {"Votes", "Aggregate_rating"}.issubset(df.columns):
    df["Votes_per_Rating_Point"] = (
        df["Votes"] / df["Aggregate_rating"].replace(0, np.nan)
    )

print("\nNew engineered columns:")
new_cols = [
    c for c in [
        "Restaurant_Name_Length",
        "Address_Length",
        "Cuisine_Count",
        "Has_Table_Booking_Encoded",
        "Has_Online_Delivery_Encoded",
        "Votes_per_Rating_Point"
    ] if c in df.columns]

print(new_cols)

print("\nSample of engineered data:")
print(df[new_cols].head())

print("\nData types after feature engineering:")
print(df[new_cols].dtypes)

# Save engineered dataset
df.to_csv("Level_2_Engineered_Dataset.csv", index=False)
print("\nSaved: Level_2_Engineered_Dataset.csv")

print("\nLEVEL 2 COMPLETE.")
