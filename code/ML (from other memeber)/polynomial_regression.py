# Import list
import numpy as np 
import pandas as pd
import datetime as dt
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Function define

def f(size):
    x = np.linspace(0,4.5,size)
    y = 2 * np.sin(x * 1.5)
    # Make sin graph
    return (x,y)

def sample(size):
    x = np.linspace(0,4.5,size)
    y = 2 * np.sin(x * 1.5) + pl.randn(x.size)
    # Sample data generation based on sin graph.
    return (x,y)

def fit_polynomial(x,y,degree):
    model = LinearRegression()
    model.fit(np.vander(x,degree+1),y)
    # Modeling input data with degree. 

    return model

def apply_polynomial(model,x):
    degree = model.coef_.size -1

    y = model.predict(np.vander(x,degree + 1))
    # Generating a prediction graph based on the model.

    return y

def plot_learning_curves(model, X, y, X_train, X_val, y_train, y_val,degree):
    train_errors, val_errors = [], []
    for m in range (1, len(X_train)):
        model.fit(np.vander(X_train[:m],degree+1), y_train[:m])
        y_train_predict = model.predict(np.vander(X_train[:m],degree+1))
        y_val_predict = model.predict(np.vander(X_val,degree+1))
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth = 2, label = "train set")
    plt.plot(np.sqrt(val_errors), "b-", linewidth = 3, label = "validation set") 
    plt.xlabel("size of train set")
    plt.ylabel("RMSE (degree {})".format(degree + 1))
    plt.legend()
    #Draw Learning curves.


# Make sample Data.
f_x,f_y = f(1000)
s_x,s_y = sample(1000)

# Split sameple Data.
X_train, X_val, y_train, y_val = train_test_split(s_x,s_y, test_size=0.2,random_state= 0)

# Draw sample Data.
plt.subplot(2,3,1)
plt.plot(s_x,s_y,'k.')

# Make model and prediction value(degree 1) 
model = fit_polynomial(X_train,y_train,1)
p_y = apply_polynomial(model,X_val)

# Draw prediction graph (degree 1) 
plt.subplot(2,3,2)
plt.plot(X_val,p_y,'g')

# Make model and prediction value(degree 3) 
model_3 = fit_polynomial(X_train,y_train,3)
p_y_3 = apply_polynomial(model_3,X_val)

# Draw prediction graph (degree 3) 
plt.subplot(2,3,3)
plt.plot(X_val,p_y_3,'b')

# Draw learning_curves (degree 1)
plt.subplot(2,3,4)
model_curve = LinearRegression()
plot_learning_curves(model_curve,s_x,s_y,X_train,X_val,y_train,y_val,1)

# Check r2_score of prediction value (degree 1)
score_1 = r2_score(y_val,p_y)
print("r2_score of sample model (degree 1) : {}".format(score_1))


# Draw learning_curves (degree 3)
plt.subplot(2,3,5)
model_curve = LinearRegression()
plot_learning_curves(model_curve,s_x,s_y,X_train,X_val,y_train,y_val,3)

# Check r2_score of prediction value (degree 3)
score_2 = r2_score(y_val,p_y_3)
print("r2_score of sample model (degree 3) : {}".format(score_2))
