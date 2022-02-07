# Import list 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
import numpy as np
import pandas as pd

# Load dataset
Data = pd.read_csv("../[01]data_set/diabetes.csv")

# Split dataset in features and target variable

X = Data[['Pregnancies','Insulin','BMI','Age','Glucose','BloodPressure','DiabetesPedigreeFunction']]
Y = Data.Outcome

# Split dataset into train set and val set
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=1) 

# Create Bagging Decision Tree classifier object.
model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10 , bootstrap=True)

# Model train.
model.fit(X_train,y_train)

# Create Decision Tree classifer object (default = gini)

decision_tree = DecisionTreeClassifier()

# Train Decision Tree classifer

decision_tree = decision_tree.fit(X_train,y_train)

print("decision_tree_model score(train_set): {}".format(decision_tree.score(X_train, y_train)))

print("decision_tree_model score(val_set): {}".format(decision_tree.score(X_val,y_val)))

# Print Bagging_model score.

print("Bagging_model score(train_set): {}".format(model.score(X_train, y_train)))

print("Bagging_model score(val_set): {}".format(model.score(X_val,y_val)))

# Make prediction test data.
test_data = np.array([[5,183,50,175,30.1,0.398,32]])

# Prediction of test data.
bagging_test_result = model.predict(test_data)
decision_tree_test_result = decision_tree.predict(test_data)

# Test Result
print("test data result : \n bagging : {}\n decision_tree : {}".format(bagging_test_result,decision_tree_test_result))
