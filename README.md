# **Project 1: Wine Quality Classification using SVMs**

This project uses Support Vector Machines (SVMs) to classify the quality of white wines based on physicochemical properties. A grid search with cross-validation optimizes hyperparameters for improved model performance, while visualizations provide insight into the results.

---

## **Project Overview**

- **Author:** Emily Lee  
- **Course:** ITP-449  
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
