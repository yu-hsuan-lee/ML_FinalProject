# **Project 1: Wine Quality Classification using SVMs**

This project uses Support Vector Machines (SVMs) to classify the quality of white wines based on physicochemical properties. A grid search with cross-validation optimizes hyperparameters for improved model performance, while visualizations provide insight into the results.

---

## **Project Overview**

- **Author:** Emily Lee  
- **Description:** This Python program analyzes a wine dataset, identifies the most influential features, and classifies wine quality using SVMs. The model's performance is evaluated using accuracy and a confusion matrix.

---

## **Features**

- **Feature Selection:**
  - Identifies the top 4 most correlated features with wine quality.
  - Selected features: Derived from the dataset using correlation analysis.

- **Model Training and Optimization:**
  - Implements a grid search with cross-validation to optimize SVM hyperparameters (`C`, `gamma`, and `kernel`).
  - Trains the best SVM model and evaluates its performance on a test set.

- **Visualization:**
  - Plots accuracy scores for each hyperparameter configuration.
  - Displays the confusion matrix with accuracy for model evaluation.

---

## **Dataset**

The dataset is **Wine Quality - White Wines**, obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). It contains physicochemical properties of white wine samples and their respective quality scores.

### **Key Features**
- **Physicochemical properties**: Acidity, residual sugar, alcohol, etc.
- **Target variable**: `Quality` (scores ranging from 3 to 9).

---

### **Results**

- **Best Hyperparameters:**
   ```python
   {'C': 10, 'kernel': 'rbf', 'gamma': 1}

- **Test Accuracy:**
   - The test accuracy of the best model is **53%**.

- **Confusion Matrix:**
   - The confusion matrix illustrates the model's ability to classify different wine quality scores:
![Screenshot 2024-11-21 at 4 02 30 PM](https://github.com/user-attachments/assets/2a0204b5-558b-46a7-af7d-c322bf25bce4)
---
### **Key Insights**

- **Correlations:** The top 4 most correlated features with wine quality are selected for training, simplifying the dataset without significant loss of information.
- **Model Optimization:** A grid search over 32 hyperparameter combinations (varying `C`, `kernel`, and `gamma`) was performed to achieve the best results.
- **Performance:** The model achieves an accuracy of **53%** on the test set, reflecting the challenge of classifying wine quality due to overlapping quality scores in the dataset.

---

### **Future Improvements**

- Improve data preprocessing to handle overlapping wine quality scores.
- Experiment with advanced ML models like **Random Forest** or **XGBoost**.
- Use **SMOTE** or other techniques to balance the dataset if class imbalance is significant.
- Include more features or combine them to improve classification accuracy.

---

### **Dependencies**

- **Python 3.7+**
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

---
# **Project 2: Mushroom Classification with Optimized Decision Tree**

This project utilizes a Decision Tree Classifier to predict whether a mushroom is **edible** or **poisonous** based on its physical characteristics. A randomized search with cross-validation is employed to optimize the model’s hyperparameters, and the results are visualized through a confusion matrix and a decision tree plot.

---

## **Project Overview**

- **Author:** Emily Lee 
- **Description:** The project uses a labeled mushroom dataset to classify mushrooms as edible or poisonous. It includes:
  - Preprocessing of categorical data.
  - Randomized hyperparameter tuning for the Decision Tree model.
  - Visualization of the confusion matrix and optimized decision tree.

---

## **Features**

- **Data Preprocessing:**
  - All categorical features are encoded using `LabelEncoder`.
  - The dataset is split into training and testing sets (70% train, 30% test).

- **Model Training:**
  - A Decision Tree Classifier is optimized using `RandomizedSearchCV` to search for the best combination of:
    - `criterion`: ['entropy', 'gini']
    - `max_depth`
    - `min_samples_split`
    - `min_samples_leaf`
  - The best model is used for predictions and evaluation.

- **Visualizations:**
  - A **Confusion Matrix** shows the classification performance with accuracy.
  - A **Decision Tree Plot** illustrates the structure of the trained classifier.

---

## **Dataset**

The dataset used is **Mushroom Classification** (available on [Kaggle](https://www.kaggle.com/uciml/mushroom-classification)). It contains data on 8,124 mushrooms with 22 categorical features describing their physical characteristics.

### **Target Variable:**
- `class`:
  - `0` = Edible
  - `1` = Poisonous

---

## **Results**

### **Best Hyperparameters:**
```python
{
  'min_samples_split': 9,
  'min_samples_leaf': 2,
  'max_depth': 66,
  'criterion': 'entropy'
}
```
### **Prediction for a New Mushroom**

- **Input Mushroom Characteristics:**
  ```python
  [5, 2, 4, 1, 8, 1, 0, 1, 4, 0, 2, 2, 2, 7, 7, 0, 2, 1, 4, 7, 3, 5]
```
- **Prediction:** Poisonous (`1`)

---

### **Model Accuracy**

- The model achieves **100% accuracy** on the test set.

---

### **Visualizations**

- **Confusion Matrix and Decision Tree Plot:**
  - The decision tree structure is visualized alongside the confusion matrix.
![output](https://github.com/user-attachments/assets/11a42331-ede1-4b98-b2b6-d172e2625edb)

---

### **Future Improvements**

- Use additional classifiers like **Random Forest** or **Gradient Boosted Trees** to compare performance.
- Perform feature importance analysis to identify the most significant predictors.
- Optimize preprocessing by handling imbalanced classes (if applicable).

---

### **Dependencies**

- **Python 3.7+**
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`



