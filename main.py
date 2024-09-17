""" Emily Lee
    ITP-449
    Final Project
    Description: This python program uses SVMs and a grid search with cross-validation to classify wine quality based on a set of features.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

def main():
    # Load dataset from local file
    data = pd.read_csv('winequality-white.csv', skiprows=1, sep=';')

    # Find the four most correlated features
    correlations = data.corr()['quality'].apply(abs)
    correlations = correlations.drop(labels=['quality']).reset_index().rename(columns={'index': 'Feature', 'quality': 'Correlation'})
    correlations.sort_values('Correlation', ascending=False, inplace=True)
    selected_features = correlations['Feature'].head(4).tolist()

    # Split dataset into features and target
    X = data[selected_features]
    y = data['quality']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)

    # Define hyperparameters
    param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.1, 1, 10, 100]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(SVC(), param_grid, cv=3, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print('Best hyperparameters:', best_params)

    # Plot the accuracy score vs hyperparameter setup index
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(range(len(grid_search.cv_results_['mean_test_score'])), grid_search.cv_results_['mean_test_score'])
    ax1.set_xlabel('Hyperparam Setup Index')
    ax1.set_ylabel('Accuracy Score')
    ax1.set_title('Accuracy Score vs Hyperparam Setup Index')

    # Train and predict using the best model
    best_model = SVC(**best_params)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', cm)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy Score:', accuracy)

    # Plot and save confusion matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot(ax=ax2, cmap=plt.cm.Blues)
    ax2.set_title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}')  # title with the accuracy score

    plt.tight_layout()
    plt.savefig('combined_figure.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
