
# 🌲 Forest Cover Type Classification

![Forest Image](https://upload.wikimedia.org/wikipedia/commons/4/4f/Forest_canopy_in_the_tropics.jpg)

## 🔍 Project Overview
This project aims to classify different **forest cover types** using **cartographic and environmental variables** from the **UCI Covertype dataset**.  
It demonstrates **multi-class classification**, **tree-based modeling**, **feature importance analysis**, and **model comparison** using **Random Forest** and **XGBoost** algorithms.

---

## 📦 Dataset

- **Source:** [UCI Machine Learning Repository – Covertype Dataset](https://archive.ics.uci.edu/ml/datasets/covertype)  
- **Instances:** ~581,012  
- **Features:** 54 predictors + 1 target (`Cover_Type`)  
- **Classes:** 7 (forest cover types)

### Key Features
- Topographic attributes: elevation, slope, aspect  
- Distances to hydrology, roads, fire points  
- Hillshade measures (morning, noon, afternoon)  
- 4 wilderness area indicators  
- 40 soil type indicators  

---

## ⚙️ Tools and Libraries

| Library | Purpose |
|----------|----------|
| **Pandas**, **NumPy** | Data manipulation and analysis |
| **Matplotlib**, **Seaborn** | Data visualization |
| **Scikit-learn** | Machine learning toolkit |
| **XGBoost** | Gradient boosting model |

---

## 🧹 Data Preprocessing

1. **Loaded the dataset** directly from UCI’s gzip file.  
2. **Renamed columns** for readability.  
3. **Subtracted 1** from the target to make it zero-indexed (0–6).  
4. **Split** the dataset into training (80%) and testing (20%) subsets.  
5. **Scaled** the numerical features using `StandardScaler` (optional for tree-based models).

---

## 🤖 Model Training

### 1. Random Forest Classifier
```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
**Accuracy:** ≈ **93.7%**

### 2. XGBoost Classifier
```python
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
xgb_model.fit(X_train, y_train)
```
**Accuracy:** ≈ **94.8%**

---

## 📊 Model Evaluation

| Metric | Random Forest | XGBoost |
|---------|----------------|-----------|
| Accuracy | 93.7% | **94.8%** |
| Precision (avg) | 0.93 | **0.95** |
| Recall (avg) | 0.94 | **0.95** |
| F1-score (avg) | 0.93 | **0.95** |

![Confusion Matrix](https://upload.wikimedia.org/wikipedia/commons/2/2d/Confusion_matrix.png)

> *The confusion matrix shows strong prediction accuracy across most cover types, with minor overlaps between types 2 and 3.*

---

## 🌟 Feature Importance

Top 10 contributing features:
1. Elevation  
2. Horizontal_Distance_To_Roadways  
3. Hillshade_9am  
4. Horizontal_Distance_To_Fire_Points  
5. Vertical_Distance_To_Hydrology  
6. Hillshade_Noon  
7. Slope  
8. Soil_Type_10  
9. Wilderness_Area_3  
10. Aspect  

![Feature Importance](https://upload.wikimedia.org/wikipedia/commons/f/fc/Feature_importance.png)

> *Elevation and horizontal distances are the strongest predictors of forest cover types.*

---

## 🧠 Hyperparameter Tuning (Bonus)

Grid Search applied to Random Forest using 3-fold CV:

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
```

**Best Parameters:**
```python
{'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}
```

**Best CV Accuracy:** ≈ **94.2%**

---

## 🧩 Results Comparison

| Model | Accuracy | Notes |
|--------|-----------|--------|
| Random Forest | 93.7% | Strong performance, stable baseline |
| **XGBoost** | **94.8%** | Slightly better generalization and performance |

> XGBoost consistently performs better due to its gradient boosting mechanism, while Random Forest provides a reliable baseline.

---

## 💡 Insights and Learnings

- Tree-based models efficiently handle high-dimensional categorical-encoded features.  
- XGBoost’s boosting approach provides incremental improvement over Random Forest’s bagging method.  
- Feature importance aligns with ecological intuition: **elevation** and **distance to roads** play major roles.

---

## 📈 Future Work

- Implement **SHAP explainability** to interpret feature influence.  
- Explore **LightGBM** and **CatBoost** for comparison.  
- Develop an **interactive dashboard** for live predictions.

---

## 🏁 Conclusion

This project showcases a complete end-to-end **multi-class classification pipeline** using the **Covertype dataset**.  
With **XGBoost achieving around 95% accuracy**, it highlights the power of ensemble methods in environmental modeling.

---

**Author:** Ridwan Hamzah  
**Languages:** Python 3.10+  
**License:** MIT  


