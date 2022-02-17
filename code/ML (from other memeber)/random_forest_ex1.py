# Import list

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score # accuracy_check function

# load data
data = pd.read_csv("../[01]data_set/Housing.csv")

# Check data.head
print(data.head(2))

# Converting some string-type data, such as yes no inside the data, into an integer type.
mapping_dict_bool = {'yes' : True,'no' : False,}
mapping_dict_furnished = {'furnished' : 1, 'semi-furnished' : 0, 'unfurnished' : -1}

bool_list = ['mainroad','guestroom','basement','hotwaterheating','airconditioning']

# Change yes or no string data to 1 or 0
for x in bool_list:
    data[x] = data[x].apply(lambda x : mapping_dict_bool[x])

# Change furnishingstatus data to 1 , 0 , -1
data['furnishingstatus'] = data['furnishingstatus'].apply(lambda x : mapping_dict_furnished[x])

# Set X, Y data.
X = data[['area', 'price', 'bathrooms', 'stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','furnishingstatus']]
Y = data['bedrooms']

# Split dataset into train set and val set.
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size = 0.2,random_state = 42)

'''

Make clf_20,clf_100, clf_20_e,clf_100_e RandomForestClassifier
    ex):
    clf_20 : RandomForestClassifier(criterion = 'gini', n_estimators = 20 , max_depth = 4)
    clf_100_e : RandomForestClassifier(criterion = 'entropy', n_estimators = 100 , max_depth = 4)

'''

# Make RandomForestClassifier and Train model with Train dataset.
clf_20 = RandomForestClassifier(criterion='gini', n_estimators=20, max_depth=4,random_state=0)
clf_20.fit(train_x,train_y)

# Make predict set of model with val dataset. and check the Acccuracy of the predict set.
predict20 = clf_20.predict(test_x)
print("gini : estimators = 10 : {}".format(accuracy_score(test_y,predict20)))

# Make RandomForestClassifier and Train model with Train dataset.
clf_100 = RandomForestClassifier(criterion='gini', n_estimators=100, max_depth=4,random_state=0)
clf_100.fit(train_x,train_y)

# Make predict set of model with val dataset. and check the Acccuracy of the predict set.
predict100 = clf_100.predict(test_x)
print("gini : estimators = 100 : {}".format(accuracy_score(test_y,predict100)))

# Make RandomForestClassifier and Train model with Train dataset.
clf_20_e = RandomForestClassifier(criterion='entropy', n_estimators=20, max_depth=4,random_state=0)
clf_20_e.fit(train_x,train_y)

# Make predict set of model with val dataset. and check the Acccuracy of the predict set.
predict20_e = clf_20_e.predict(test_x)
print("entropy : estimators = 10 : {}".format(accuracy_score(test_y,predict20_e)))

# Make RandomForestClassifier and Train model with Train dataset.
clf_100_e = RandomForestClassifier(criterion='entropy', n_estimators=100, max_depth=4,random_state=0)
clf_100_e.fit(train_x,train_y)

# Make predict set of model with val dataset. and check the Acccuracy of the predict set.
predict100_e = clf_100_e.predict(test_x)
print("entropy : estimators = 100 : {}".format(accuracy_score(test_y,predict100_e)))
