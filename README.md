# F1 Championship Prediction – Data Engineering & Machine Learning Pipeline

This project implements a complete data engineering and machine learning pipeline to analyze historical Formula 1 data and estimate **each driver’s likelihood of becoming World Champion at different points in the season**, based on performance patterns from past champions.

In simple terms:

> **The model evaluates how closely each driver's performance resembles that of past World Champions, producing a score between 0 and 1 that represents their championship likelihood at that moment.**


![Alt Text](figures\future_bar_race.gif)
![Alt Text](figures\future_top5.png)



## 📌 Project Overview

The pipeline is structured into several layers that follow modern data architecture patterns:

1. **Raw Layer**  
   - Contains the original CSV files exactly as downloaded.

2. **Bronze Layer (Normalized Raw Data)**  
   - Converts raw CSVs into Delta Lake tables.  
   - Ensures consistent schemas and efficient columnar storage.

3. **Silver Layer (Cleaned & Enriched Data)**  
   - Contains curated datasets derived from SQL transformations.
   - Includes multi-season performance summaries and race-level stats.

4. **Feature Store**  
   - Contains **driver-level, time-aware** features such as:
     - rolling performance metrics  
     - season-to-date performance  
     - sprint and race averages  
     - gain/loss indicators  
     - historical form vs. current form  

5. **ABT (Analytical Base Table)**  
   - Consolidates all drivers, features, and labels into a training-ready dataset.

6. **Machine Learning Model**  
   - Trains a **Random Forest classifier** to estimate championship likelihood.  
   - Generates:
     - OOT (Out-of-Time) evaluation
     - Future season predictions
     - Line charts and **bar chart race GIFs** showing probability evolution.

## 📁 Project Structure

```
F1_DATA_ENGINEERING/
│
├── data/
│   ├── raw/                     # Original CSV files
│   ├── bronze/
│   │   └── results/             # Delta table (Bronze)
│   ├── silver/
│   │   ├── abt_champions/       # ABT (Analytical Base Table)
│   │   ├── champions/           # Silver table
│   │   └── feature_store_drivers/ # Driver-level feature store
│
├── figures/
│   ├── combined_history.png
│   ├── future_bar_race.gif
│   ├── future_top5.png
│   ├── oot_bar_race.gif
│   └── oot_top_drivers.png
│
├── scripts/
│   ├── 01_raw.py
│   ├── 02_bronze.py
│   ├── 03_feature_store.py
│   ├── 04_silver.py
│   └── 05_ml_model.py
│   └── spark_ops.py
│
├── sql/
│   ├── abt_champions.sql
│   ├── champions.sql
│   └── feature_store_drivers.sql
│
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```

## ⚙️ Technology Stack

### Data Engineering
- PySpark  
- Delta Lake  
- SparkSQL  
- Rich CLI

### Machine Learning
- Scikit-learn  
- feature_engine  
- Pandas / NumPy  

### Visualization
- Matplotlib, Seaborn  
- bar_chart_race (GIF animation)

## 🚀 Running the Pipeline

### 1. Get Raw Data
```bash
uv run scripts/01_raw.py --start 1990 --stop 2025
```

### 2. Build Bronze Layer
```bash
uv run scripts/02_bronze.py
```

### 3. Build Feature Store
```bash
uv run scripts/03_feature_store.py --query sql/feature_store_drivers.sql --start 1990-01-01 --stop 2026-01-01
```

### 4. Build Silver Layer
```bash
uv run scripts/04_silver.py --query sql/champions.sql

uv run scripts/04_silver.py --query sql/abt_champions.sql 
```

### 5. Train ML Model
```bash
uv run scripts/05_ml_model.py
```

Outputs are saved to the `figures/` directory.

## 🔍 What the Model Predicts

### Simple explanation

> **It estimates how strongly each driver’s current season performance resembles the profile of a typical F1 World Champion.**

### Technical explanation

> **Given a driver’s historical performance up to any point in the season, the model assigns a score between 0 and 1 representing how likely that driver is to end the year as World Champion — based on patterns extracted from past champions.**

**Important:** Probabilities for different drivers do **not** have to sum to 1.

## 📈 Outputs

- Probability curves  
- OOT predictions  
- Future season forecasts  
- Top-5 trends  
- Animated bar chart race GIF  

## 🧠 High-Level Modeling Approach

- Binary classification (`champion` vs `not champion`)  
- Out-of-time validation  
- Driver-year sampling  
- Rolling windows  
- Season aggregates  
- Sprint & race metrics  
- Consistency indicators 

---

