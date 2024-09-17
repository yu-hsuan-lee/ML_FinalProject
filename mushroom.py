""" Emily Lee
    ITP-449
    Final Project
    Description: This Python program uses a decision tree classifier and a randomized search with cross-validation to predict the edibility of mushrooms based on their characteristics.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def plot_cm(cm, ax):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_aspect('auto')

    # Add the colorbar to the plot
    cbar = ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=['edible', 'poisonous'], yticklabels=['edible', 'poisonous'],
           ylabel='True label',
           xlabel='Predicted label')

    # Calculate accuracy
    accuracy = np.trace(cm) / float(np.sum(cm))
    title_text = f'Confusion Matrix\nAccuracy: {accuracy:.2f}'

    # Set the title with a custom font size
    ax.set_title(title_text, fontsize=16)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

def main():
    # Load the dataset
    data = pd.read_csv('mushrooms.csv')

    # Encode categorical variables
    le = LabelEncoder()
    for col in data.columns:
        data[col] = le.fit_transform(data[col])

    # Prepare the features and target variables
    X = data.drop('class', axis=1)
    y = data['class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Set up the parameter search space
    param_dist = {
        'criterion': ['entropy', 'gini'],
        'max_depth': np.arange(2, int(np.sqrt(len(X_train)))),
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(2, 11),
    }

    # Set up the randomized search with cross-validation
    clf = DecisionTreeClassifier()
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=100, cv=5, random_state=42)

    # Fit the model to the training data
    random_search.fit(X_train, y_train)

    # Print the best parameters
    print('Best parameters:', random_search.best_params_)

    # Predict mushroom edibility
    mushroom = np.array([
        5, 2, 4, 1, 8, 1, 0, 1, 4, 0, 2, 2, 2, 7, 7, 0, 2, 1, 4, 7, 3, 5
    ]).reshape(1, -1)
    mushroom_df = pd.DataFrame(mushroom, columns=X.columns)  # Convert the input array to a DataFrame with feature names
    prediction = random_search.predict(mushroom_df)
    print('Mushroom edibility (0 = edible, 1 = poisonous):', prediction[0])

    # Save Confusion Matrix and Decision Tree Visualization as a single PNG
    y_pred = random_search.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, (ax_cm, ax_tree) = plt.subplots(1, 2, figsize=(24, 12))

    plot_cm(cm, ax_cm)

    plot_tree(random_search.best_estimator_, ax=ax_tree, filled=True, feature_names=X.columns, fontsize=6.5)

    # Move the tree plot to the left
    pos = ax_tree.get_position()
    pos.x0 -= 0.07
    ax_tree.set_position(pos)

    # Add a title to the tree plot
    ax_tree.set_title('Optimized Decision Tree', fontsize=16)

    # Save png
    fig.savefig('combined_visualizations.png', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    main()

