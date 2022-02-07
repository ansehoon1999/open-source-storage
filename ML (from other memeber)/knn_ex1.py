import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 사이킷런에서 제공하는 심장병 데이터 불러오기
breast_cancer = load_breast_cancer()

# 데이터 키 확인
breast_cancer.keys()

# 데이터 분류
Xd_train, Xd_test, yd_train, yd_test = train_test_split(breast_cancer['data'],breast_cancer['target'],
test_size=0.3,random_state=40)

# 리스트 형태로 1~11까지의 train, test 정확도 그래프
train_acc, test_acc = [],[]

for i in range(1,12):
    clf3 = KNeighborsClassifier(n_neighbors=i)
    clf3.fit(Xd_train, yd_train)
    predict_label = clf3.predict(Xd_train)
    train_acc.append(clf3.score(Xd_train, yd_train))
    test_acc.append(clf3.score(Xd_test, yd_test))
    
plt.plot(range(1,12), train_acc, label="train")
plt.plot(range(1,12), test_acc, label="test")
plt.show()
