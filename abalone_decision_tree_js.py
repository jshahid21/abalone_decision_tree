# Step 1 - Import all required binaries for applying decision tree classifier on abalone data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz

# Step 2 - Read data into a data frame
names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
df = pd.read_csv('/content/abalone.data', names=names, sep=',')

df.head()

# Step 3 - Preprocess the data
df.dtypes

df.isna().sum()

df = df.drop('Sex', axis=1)

df.columns

# Step 4 - Define X and y
X = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight']]
y = df['Rings']

# Step 5 - Split X and y into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=45)

# Step 6 - Create an object of the algorithm
dt = DecisionTreeRegressor()

# Step 7 - Fit the data to the object
dt.fit(X_train, y_train)

dt.score(X_train,y_train)

tree1 = tree.export_text(clf,feature_names=list(X_train.columns))
print(tree1)

n_nodes = tree_structure.node_count
n_nodes

children_left = tree_structure.children_left
print(f"Childrens left :{children_left}")
children_right = tree_structure.children_right
print(f"Children right :{children_right}")
feature = tree_structure.feature
print(f"Features:{feature}")
threshold = tree_structure.threshold
print(f"Threshold :{threshold}")

y_pred = dt.predict(X_test)

accuracy_score(y_test,y_pred)



"""# Abalone Age Prediction using Decision Tree Regression

This project explores the use of a Decision Tree Regressor to predict the age of abalone based on physical measurements.

## Model Performance

The Decision Tree Regressor achieved an accuracy score of approximately **20.2%** on the test set.

## Interpretation of Accuracy Score

An accuracy score of 20.2% is very low for this regression task. This suggests that the model is not performing well in predicting the number of rings (age) of the abalone. There could be several reasons for this:

*   **Data complexity**: The relationship between the physical measurements and the number of rings might be more complex than a single decision tree can effectively capture.
*   **Feature limitations**: The chosen features might not be sufficiently predictive of the abalone's age.
*   **Model limitations**: Decision Tree Regressors can be prone to overfitting, especially on datasets with a wide range of target values like this one.

## Next Steps

Further steps could involve:

*   Exploring the data distribution and relationships between features and the target variable.
*   Trying different regression models (e.g., Random Forest, Gradient Boosting, Support Vector Regression).
*   Performing feature engineering to create new features that might be more predictive.
*   Hyperparameter tuning of the Decision Tree Regressor or other chosen models.
"""
