#import list
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load data_set
data = pd.read_csv("../[01]data_set/imports-85.data",na_values = '?')
# Set columns
data.columns = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']

# option 0 : delete all nan values row.
'''data = data.dropna(axis=0)'''
# option 1 : replace all nan values with mean values of columns
df = data[["make","engine-size","highway-mpg","city-mpg","price"]]

df = df.groupby("make").transform(lambda x: x.fillna(x.mean()))
# In this sample code use option 1

# Set X,Y data_set
X = df[["engine-size","highway-mpg","city-mpg"]]
y = df["price"]

# def funtion : draw Learning_curves in this Data set.
def plot_learning_curves(model, X, y, X_train, X_val, y_train, y_val):
    train_errors, val_errors = [], []
    for m in range (1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        # Train model with x_train[0~m],y_train[0~m]
        y_train_predict = model.predict(X_train[:m])
        # Check model with x_train[0~m]
        y_val_predict = model.predict(X_val)
        # Check model with x_val[0~m]
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
        # add train and val error in list.
    # Draw learning curves.
    plt.plot(np.sqrt(train_errors), "r-+", linewidth = 2, label = "train set")
    plt.plot(np.sqrt(val_errors), "b-", linewidth = 3, label = "validation set") 
    plt.xlabel("size of train set")
    plt.ylabel("RMSE")
    plt.legend()

# New linear model
model = LinearRegression()

# Split Dataset into Trainset and val_set
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2,random_state= 0)

# Draw learning curves
plot_learning_curves(model,X,y,X_train, X_val, y_train, y_val)


# New linear model
model_1 = LinearRegression()

# Train model
model_1.fit(X_train,y_train)

# Make y_pred list with x_val list
y_pred = model_1.predict(X_val)

# Check Suitable of y_pred and y_val
score = r2_score(y_val,y_pred)
print(score)
