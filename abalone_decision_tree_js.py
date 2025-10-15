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
