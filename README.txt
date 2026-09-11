# 🤖 Cognifyz Technologies Data Science Internship Projects

# 📌 Project Overview

This repository contains multiple **Data Science and Machine Learning projects** completed as part of the **Cognifyz Technologies Data Science Internship**.

The projects are organized into **Level 1, Level 2, and Level 3**, using the **Zomato Restaurant Dataset** to perform data analysis, visualization, feature engineering, and Machine Learning.

The projects demonstrate the complete Data Science workflow, including:

* Data Collection
* Data Cleaning
* Data Preprocessing
* Exploratory Data Analysis (EDA)
* Data Visualization
* Feature Engineering
* Geographical Analysis
* Statistical Analysis
* Machine Learning
* Model Building
* Model Evaluation
* Prediction

Each level focuses on different aspects of Data Science and Machine Learning using real-world restaurant data.

---

# 📚 Repository Contents

## 🟢 Level 1 – Data Exploration & Analysis

Level 1 focuses on understanding, cleaning, and exploring the Zomato restaurant dataset.

### 🔹 Tasks Covered

* Data loading and inspection
* Dataset shape and information
* Column analysis
* Data type checking
* Missing value analysis
* Duplicate value checking
* Data cleaning
* Restaurant analysis
* Rating analysis
* Price range analysis
* Online delivery analysis
* Table booking analysis
* Basic data visualization

### Dataset

* Zomato Restaurant Dataset

---

## 🟡 Level 2 – Advanced Data Analysis

Level 2 focuses on deeper analysis of restaurant characteristics and customer behavior.

### 🔹 Tasks Covered

* Cuisine analysis
* Restaurant rating analysis
* Price range analysis
* Customer voting analysis
* Online delivery trends
* Table booking trends
* Relationship between different restaurant features
* Statistical analysis
* Advanced data visualization
* Geographical insights

### Dataset

* Zomato Restaurant Dataset

---

## 🔴 Level 3 – Machine Learning & Predictive Analysis

Level 3 focuses on applying Machine Learning techniques to the processed Zomato dataset.

### 🔹 Tasks Covered

* Feature selection
* Feature engineering
* Numerical feature preprocessing
* Categorical feature encoding
* Train-test splitting
* Machine Learning pipeline
* Model training
* Prediction
* Model evaluation
* Performance analysis

### Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
```

---

# 🎯 Objectives

* Load and understand the Zomato dataset
* Perform data cleaning and preprocessing
* Handle missing and duplicate values
* Perform Exploratory Data Analysis (EDA)
* Analyze restaurant ratings
* Analyze cuisines and restaurant categories
* Study price range and customer behavior
* Analyze online delivery and table booking
* Perform geographical analysis
* Visualize relationships among variables
* Perform feature engineering
* Build Machine Learning models
* Compare model performance
* Evaluate Machine Learning models
* Generate meaningful predictions and insights

---

# 🛠️ Technologies Used

* Python 3.x
* Jupyter Notebook
* Visual Studio Code
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Folium
* Git
* GitHub

---

# 📚 Python Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import folium

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

---

# 📊 Exploratory Data Analysis (EDA)

The following analyses were performed throughout the different levels:

* Dataset Shape
* Dataset Information
* Column Analysis
* Data Types
* Null Value Check
* Duplicate Value Check
* Statistical Summary
* Unique Value Analysis
* Rating Distribution
* Cuisine Analysis
* Price Range Analysis
* Restaurant Distribution
* Correlation Matrix
* Pair Plot
* Scatter Plot
* Bar Plot
* Histogram
* Distribution Plot
* Box Plot
* Heatmap
* Geographical Visualization

---

# 🗺️ Geographical Analysis

Geographical information from the Zomato dataset was analyzed using latitude and longitude.

The analysis includes:

* Restaurant locations
* Restaurant concentration
* Location-based restaurant distribution
* Latitude and longitude analysis
* Interactive geographical visualization

**Folium** was used where applicable to create interactive restaurant maps.

---

# 🤖 Machine Learning

Level 3 includes Machine Learning-based predictive analysis.

## 🔹 Data Preprocessing

The Machine Learning workflow includes:

* Handling missing values
* Selecting relevant features
* Encoding categorical variables
* Scaling numerical features where required
* Splitting data into training and testing sets
* Creating preprocessing pipelines

## 🔹 Machine Learning Pipeline

```text
Raw Data
   ↓
Cleaning
   ↓
Feature Selection
   ↓
Preprocessing
   ↓
Encoding / Scaling
   ↓
Train-Test Split
   ↓
Machine Learning Model
   ↓
Prediction
   ↓
Evaluation
```

---

# 📈 Model Evaluation

Depending on the Machine Learning task, appropriate evaluation metrics were used.

## Classification Metrics

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

## Regression Metrics

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

# 📁 Repository Structure

```text
Cognifyz-Data-Science-Internship/
│
├── Dataset.csv
│
├── Level 1/
│   ├── main.ipynb
│   │   └── 
│   │
│   ├── Dataset.csv
│   │
│   │
│   └── restaurant_locations.html
│       
│
├── Level 2/
│   ├── main.ipynb
│   │
│   │
│   ├── Dataset.csv
│   │
│   │
│   └── Level_2_Engineered_Dataset.csv
│
│
├── Level 3/
│   ├── main.ipynb
│   │
│   │
│   ├── Dataset.csv
│   │
│   │
│   └── Level_3_Model_Results.csv
│
│
├── README.md
│
└── requirements.txt
```

> Update the folder names if your actual GitHub repository uses different names.

---

# ▶️ How to Run

## 1️⃣ Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-LINK>
```

---

## 2️⃣ Navigate to the Repository

```bash
cd Cognifyz-Data-Science-Internship
```

---

## 3️⃣ Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn folium jupyter
```

---

## 4️⃣ Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## 5️⃣ Open Any Project

Open the required notebook from:

```text
Level 1/
Level 2/
Level 3/
```

Run all notebook cells sequentially.

---

# 📊 Features

✔ Data Cleaning

✔ Data Preprocessing

✔ Exploratory Data Analysis (EDA)

✔ Data Visualization

✔ Restaurant Analysis

✔ Cuisine Analysis

✔ Rating Analysis

✔ Price Analysis

✔ Customer Behavior Analysis

✔ Geographical Analysis

✔ Feature Engineering

✔ Machine Learning

✔ Model Training

✔ Model Evaluation

✔ Prediction

---

# 📷 Visualizations

The projects contain multiple visualizations, including:

* Bar Plot
* Count Plot
* Scatter Plot
* Histogram
* Distribution Plot
* Box Plot
* Correlation Heatmap
* Pair Plot
* Rating Distribution
* Cuisine Distribution
* Price Range Visualization
* Geographical Map
* Model Evaluation Visualizations
* Feature Analysis

---

# 🚀 Results

The projects provide practical insights into the Zomato restaurant dataset through data analysis and visualization.

Key outcomes include:

* Better understanding of restaurant data
* Identification of popular cuisines
* Analysis of restaurant ratings
* Understanding of price range patterns
* Analysis of customer votes
* Understanding of online delivery trends
* Analysis of table booking availability
* Geographical understanding of restaurants
* Identification of relationships between important features
* Application of Machine Learning for predictive analysis

---

# 📖 Learning Outcomes

During this internship, I gained practical experience in:

* Python Programming
* Data Analysis
* Data Cleaning
* Data Preprocessing
* Exploratory Data Analysis
* Data Visualization
* Statistical Analysis
* Feature Engineering
* Machine Learning
* Model Training
* Model Evaluation
* Predictive Analysis
* Jupyter Notebook
* Visual Studio Code
* Git & GitHub

---

# 🔮 Future Improvements

* Hyperparameter Tuning
* Cross Validation
* Advanced Feature Engineering
* Advanced Machine Learning Models
* Ensemble Learning
* Model Optimization
* Streamlit Web Application
* Flask/FastAPI Deployment
* Docker Containerization
* Cloud Deployment
* Interactive Data Analytics Dashboard

---

# 👨‍💻 Author

**Kartik Tripathi**

**B.Tech – Computer Science & Engineering**

### 🔗 GitHub

https://github.com/panditkartiktripathig

### 🔗 LinkedIn

https://www.linkedin.com/in/PanditKartikTripathiG/

---

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

Feel free to fork this repository, create a new branch, and submit a Pull Request.

---

# 📄 License

This repository is created for **educational and internship purposes** as part of the **Cognifyz Technologies Data Science Internship**.

---

# ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.

Your support motivates me to continue learning and creating Data Science and Machine Learning projects.

---

# 🙏 Acknowledgement

I would like to thank **Cognifyz Technologies** for providing the opportunity to work on practical Data Science projects and gain hands-on experience with real-world datasets.

---

## Thank You! 😊

Thank you for visiting this repository!
