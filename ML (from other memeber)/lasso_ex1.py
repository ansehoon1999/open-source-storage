# Import list
import numpy as np
import pandas as pd
from matplotlib.pylab import rcParams
rcParams['figure.figsize']= 8,5
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Make X,y data set
x = np.array([i*np.pi/180 for i in range(60, 300, 4)])
np.random.seed(20)
y = np.sin(x) + np.random.normal(0,0.15,len(x))
data = pd.DataFrame(np.column_stack([x,y]), columns=['x','y'])

# Draw plot.
plt.plot(data['x'],data['y'],'.')
for i in range(2,16):
    colname = 'x_%d'%i
    data[colname] = data['x']**i

predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

# Split alpha value (parameters of Ridge)
alpha_ridge = [1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]
models_to_plot = {1e-15:231,1e-10:232,1e-4:233,1e-3:234,1e-2:235,5:236}

# Make table to save result.
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index = ind, columns = col)

#def function of Lasso_Regression and Draw Lasso_regression graph.
def lasso_regression(data,predictors,alpha ,models_to_plot):
    # Model train and predict
    lassoreg = Lasso(alpha= alpha,normalize = True)
    lassoreg.fit(data[predictors],data['y'])
    y_pred = lassoreg.predict(data[predictors])

    # Draw plot of RIDGE regression
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha : %.3g'%alpha)

    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret

# Make Lasso_Regression.
for i in range(10):
    coef_matrix_ridge.iloc[i,] = lasso_regression(data,predictors,alpha_ridge[i],models_to_plot)

# print result table.
coef_matrix_ridge
