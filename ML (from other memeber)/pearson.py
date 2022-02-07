import pandas as pd
import numpy as np

from sklearn import datasets
data = datasets.load_diabetes()

df = pd.DataFrame(data['data'], index=data['target'], columns=data['feature_names'])

X = df.bmi.values
Y = df.index.values

# BMI(체질량지수)와 당뇨병수치(Target)의 산점도 그래프
import matplotlib.pyplot as plt
plt.scatter(X, Y, alpha=0.5)
plt.title('TARGET ~ BMI')
plt.xlabel('BMI')
plt.ylabel('TARGET')
plt.show()

cov = (np.sum(X*Y) - len(X)*np.mean(X)*np.mean(Y)) / len(X)
cov
corr = cov / (np.std(X) * np.std(Y))
corr

np.cov(X,Y)[0,1]
np.corrcoef(X,Y)[0,1]
import scipy.stats as stats
stats.pearsonr(X,Y)
#(상관계수, p-value)
