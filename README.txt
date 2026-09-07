# Cognifyz Technologies - Data Science Internship Project

## Files

- `Level_1.py` -> Data Exploration & Preprocessing, Descriptive Analysis, Geospatial Analysis
- `Level_2.py` -> Table Booking & Online Delivery, Price Range Analysis, Feature Engineering
- `Level_3.py` -> Customer Preference Analysis, Data Visualization, Predictive Modeling

## Dataset

Put the restaurant CSV in the same folder and name it:

`Dataset.csv`

If your file has another name, change the `FILE = "Dataset.csv"` line in each script.

## Install packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn folium
```

## Run

```bash
python Level_1.py
python Level_2.py
python Level_3.py
```

The scripts print the requested results and generate plots.

Level 1 may generate:
- `restaurant_locations.html`

Level 2 generates:
- `Level_2_Engineered_Dataset.csv`

Level 3 generates:
- `Level_3_Model_Results.csv`

## Important

The exact numerical answers are calculated from the CSV when the scripts run. Do not manually enter or invent dataset-specific values.
