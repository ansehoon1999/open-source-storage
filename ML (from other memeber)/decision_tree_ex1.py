# import list
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz 

# First you need to download data set file. check [01]data_set/diabetes.csv download.txt

# Load dataset
Data = pd.read_csv("../[01]data_set/diabetes.csv")

# Split dataset in features and target variable

X = Data[['Pregnancies','Insulin','BMI','Age','Glucose','BloodPressure','DiabetesPedigreeFunction']]
Y = Data.Outcome

# Split dataset into train set and val set
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=1) 

# Create Decision Tree classifer object (default = gini)

decision_tree = DecisionTreeClassifier()

# Train Decision Tree classifer

decision_tree = decision_tree.fit(X_train,y_train)

# Predict of val data.

y_pred_gini = decision_tree.predict(X_val)

# Create Decision Tree classifer object ( criterion="entropy" )

decision_tree_entropy = DecisionTreeClassifier( criterion="entropy" , max_depth = 4)

# Train Decision Tree classifer

decision_tree_entropy = decision_tree_entropy.fit(X_train,y_train)

# Predict of val data.

y_pred_entropy = decision_tree_entropy.predict(X_val)

# Model Accuracy check

print("Accuracy(Gini):",metrics.accuracy_score(y_val, y_pred_gini))

print("Accuracy(entropy):",metrics.accuracy_score(y_val, y_pred_entropy))

# def for draw decision_tree
def show_trees(tree):
    dot_data = export_graphviz(tree, out_file=None, class_names=["1", "0"],
                    feature_names = ['Pregnancies','Insulin','BMI','Age','Glucose','BloodPressure','DiabetesPedigreeFunction'],
                    precision=3, filled=True, rounded=True, special_characters=True)

    pred = tree.predict(X_val)
    print('Accuracy : {:.2f} %'.format(accuracy_score(y_val, pred) * 100))

    graph = graphviz.Source(dot_data)
    return graph

gini_graph = show_trees(decision_tree)

entropy_graph = show_trees(decision_tree_entropy)

# Show decision_tree in JUPYTER

# gini_graph
entropy_graph

# make decision_tree pdf

gini_graph.render("Gini")
entropy_graph.render("Entropy")

