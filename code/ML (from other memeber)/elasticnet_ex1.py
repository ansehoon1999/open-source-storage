# Import list 
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Global array.
my_predictions = {}
my_pred = {}
my_actual = {}
my_name = {}

# def for draw graph.
def plot_predictions(name_s,pred_s,actual_s):
    
    for name in name_s:
        df = pd.DataFrame({'prediction':pred_s[name],'actual':actual_s[name]})
        df = df.sort_values(by='actual').reset_index(drop=True)

        plt.figure(figsize=(11,8))
        plt.scatter(df.index, df['prediction'],marker='x',color='r')
        plt.scatter(df.index,df['actual'],alpha=0.7,marker='o',color='black')
        plt.title(name,fontsize=15)
        plt.legend(['prediction','actual'],fontsize=12)
        plt.show()

    
# def for add model name, test_data, prediction_data to Global array and check mse.
def add_model(name_,pred,actual):
    global my_predictions,my_pred,my_actual,my_name
    my_name[name_] = name_
    my_pred[name_] = pred
    my_actual[name_] = actual

    mse = mean_squared_error(pred,actual)
    my_predictions[name_] = mse

# def for check the result.
def plot_all():
    global my_predictions,my_pred,my_actual,my_name

    plot_predictions(my_name,my_pred,my_actual)
    y_value = sorted(my_predictions.items(),key=lambda x: x[1],reverse=True)

    df = pd.DataFrame(y_value,columns=['model','mse'])
    print(df)
    min_ = df['mse'].min() - 10
    max_ = df['mse'].max() + 10

    length = len(df)

    plt.figure(figsize=(9,length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model'],fontsize=12)
    bars = ax.barh(np.arange(len(df)),df['mse'],height=0.3)

    for i, v in enumerate(df['mse']):
        ax.text(v+2,i,str(round(v,3)),color='k',fontsize=12,fontweight='bold',verticalalignment='center')

    plt.title('mse error',fontsize=16)
    plt.xlim(min_,max_)

    plt.show()

# RandomSeed
SEED = 30

# Load dataset.
data = pd.read_csv('winequality-red.csv')

x = data[['fixed acidity','volatile acidity','citric acid','residual sugar','alcohol','free sulfur dioxide','density','chlorides','pH','sulphates']]
y = data['total sulfur dioxide']

# Split sameple Data (test,train set).
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=SEED)

# alpha and l1_ratio list.
alphas = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
ratios = [0.2,0.5,0.8]

# Make ElasticNet model and test the model.
for ratio in ratios:
    for alpha in alphas:
        elasticnet = ElasticNet(alpha = alpha, l1_ratio=ratio,random_state=SEED)
        elasticnet.fit(x_train,y_train)
        pred = elasticnet.predict(x_test)
        # Add test result in Global array.
        add_model('ElasticNet(l1_ratio = {},alpha = {})'.format(ratio,alpha),pred,y_test)

# Check the Result all.
plot_all()
