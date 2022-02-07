import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# 데이터 셋 로드
cancer = load_breast_cancer()

# 데이터 프레임 만들기
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df['target'] = cancer['target']
df.head()

# standardization
scaler = StandardScaler()
scaled = scaler.fit_transform(df.drop('target', axis=1))

# train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(scaled, df['target'],  random_state=10)

# logistic regression
lr_clf = LogisticRegression()
lr_clf.fit(x_train, y_train)
pred = lr_clf.predict(x_valid)

# 정확도 측정
accuracy_score(y_valid, pred)

from sklearn.metrics import confusion_matrix
from IPython.display import Image

cm = confusion_matrix(y_valid, pred)

import seaborn as sns

sns.heatmap(cm, annot=True, annot_kws={"size": 20}, cmap='YlOrBr')
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Actual', fontsize=20)

TN, FP, FN, TP = cm.ravel()

tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()

precision = TP / (FP + TP)

from sklearn.metrics import precision_score

precision_score(y_valid, pred)

# 검증
recall = TP / (FN + TP)
2 * (precision * recall) / (precision + recall)
from sklearn.metrics import f1_score
f1_score(y_valid, pred)
