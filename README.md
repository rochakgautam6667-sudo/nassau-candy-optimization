# Nassau Candy Distributor — Factory Reallocation & Shipping Optimization

> **Unified Mentor Internship — Data Analyst Intern**  
> **Author:** Rochak Gautam  
> **Domain:** Supply Chain / Transportation Analytics  

---

## What This Project Does

Nassau Candy currently assigns every product to one fixed factory, regardless of where customers actually are. This project builds a data-driven system to answer: **which factory should make each product to make shipping faster and more profitable?**

It does this by:
1. Predicting shipping lead time using Machine Learning (no raw Ship Date used — see below)
2. Clustering factory-region routes to spot slow ones
3. Simulating every product against every factory (75 scenarios)
4. Producing a ranked, risk-flagged list of which products should move — and where

---

## Results

| Metric | Value |
|---|---|
| **Total profit upside identified** | $46,387 |
| **Average delivery speed improvement** | 17.4% |
| **Products recommended for reassignment** | 6 of 15 |
| **Capital investment required** | None |

All 6 recommended products should move to **Secret Factory** (Illinois) — the most centrally located of the 5 factories relative to the actual customer base.

---

## Repository Structure

```
├── Nassau_Candy_Factory_Optimization_Rochak_Gautam_Project1UM.ipynb   ← main notebook
├── Nassau_Candy_Distributor.csv                                        ← input dataset
├── processed_orders.csv       ← cleaned data + engineered features
├── route_clusters.csv         ← factory × region cluster table
├── recommendations.csv        ← all 75 simulation scenarios, ranked
├── model_results.csv          ← RMSE / MAE / R² for all 3 models
├── app.py                     ← Streamlit dashboard
├── requirements.txt           ← Python dependencies
└── README.md
```

---

## How to Run the Notebook

1. Clone the repo  
2. Put `Nassau_Candy_Distributor.csv` in the same folder as the notebook  
3. Open the notebook in Jupyter and run all cells top to bottom

**Dependencies** — only standard packages, no `geopy` or unusual installs:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run the Streamlit Dashboard

```bash
pip install streamlit plotly
streamlit run app.py
```

---

## Notebook Walkthrough (9 Sections)

| Section | What it does |
|---|---|
| 1. Import data | Load CSV, check shape/types/nulls/duplicates |
| 2. EDA | Distributions, sales by division/region/ship mode, IQR outlier removal |
| 3. Ship Date problem | Documents why `Ship Date` column can't be used (3.6-year average gap) |
| 4. Feature engineering | Haversine distance (factory → customer state), synthetic `Lead_Time_Days` |
| 5. ML models | Linear Regression, Random Forest, Gradient Boosting — compared with RMSE/MAE/R² |
| 6. Route clustering | KMeans on Factory+Region combos → Fast / Moderate / Slow labels |
| 7. Scenario simulation | All 15 products × 5 factories → predicted lead time + profit |
| 8. Recommendations | Ranked by 50/50 score (speed + profit); low-volume products flagged |
| 9. Save results | Export CSVs for reuse |

---

## Important: Ship Date Data Quality Issue

The `Ship Date` column in the source spreadsheet is corrupted — the average gap between Order Date and Ship Date is **1,320 days (3.6 years)**, which is physically impossible for a candy shipment. This was investigated and documented rather than silently patched.

**Fix:** `Lead_Time_Days` was built from scratch using:
- Great-circle distance between factory and customer state (haversine formula)
- Ship mode speed (Same Day fastest → Standard Class slowest)
- Small random noise to simulate real-world variation

This matches the project brief's exact modeling objective: *predict lead time given product, factory, region, and ship mode.*

---

## Key Finding

**Secret Factory** (Moline, Illinois) is the most central of the 5 factories — average 843 miles to customers, versus 1,456 miles for the furthest factory. Moving the 5 Chocolate Wonka Bar SKUs and Kazookles there is worth **$46,387 in combined profit improvement and 17.4% faster delivery** with zero capital outlay.

---

## Links

| Resource | Link |
|---|---|
| 📓 Notebook | This repository |
| 📄 Research Paper | [Add your Zenodo link here] |
| 🌐 Streamlit App | [Add your Streamlit Cloud link here] |
| 🎥 Project Video | [Add your video link here] |
