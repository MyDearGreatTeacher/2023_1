# [不靠框架硬功夫 - Scikit-learn 手刻機器學習每行程式碼](https://www.tenlong.com.tw/products/9786267146736?list_name=lv)
```
01 機器學習概述

1.1 什麼是機器學習

1.2 機器學習的作用領域

1.3 機器學習的分類

1.4 機器學習理論基礎

1.5 機器學習應用程式開發的典型步驟

1.6 本章小結

1.7 複習題

 

02 機器學習之資料特徵

2.1 資料的分佈特徵

2.2 資料的相關性

2.3 資料的聚類性

2.4 資料主成分分析

2.5 資料動態性及其分析模型

2.6 資料視覺化

2.7 本章小結

2.8 複習題

 

03 用scikit-learn 估計器分類

3.1 scikit-learn 基礎

3.2 scikit-learn 估計器

3.3 本章小結

3.4 複習題

 

04 單純貝氏分類

4.1 演算法原理

4.2 單純貝氏分類

4.3 單純貝氏分類實例

4.4 單純貝氏連續值的處理

4.5 本章小結

4.6 複習題

 

05 線性回歸

5.1 簡單線性回歸模型

5.2 分割資料集

5.3 用簡單線性回歸模型預測考試成績

5.4 本章小結

5.5 複習題

 

06 用 k 近鄰演算法分類和回歸

6.1 k 近鄰演算法模型

6.2 用 k 近鄰演算法處理分類問題

6.3 用 k 近鄰演算法對鳶尾花進行分類

6.4 用 k 近鄰演算法進行回歸擬合

6.5 本章小結

6.6 複習題

 

07 從簡單線性回歸到多元線性回歸

7.1 多變數的線性模型

7.2 模型的最佳化

7.3 用多元線性回歸模型預測波士頓房價

7.4 本章小結

7.5 複習題

 

08 從線性回歸到邏輯回歸

8.1 邏輯回歸模型

8.2 多元分類問題

8.3 正則化項

8.4 模型最佳化

8.5 用邏輯回歸演算法處理二分類問題

8.6 辨識手寫數字的多元分類問題

8.7 本章小結

8.8 複習題

 

09 非線性分類和決策樹回歸

9.1 決策樹的特點

9.2 決策樹分類

9.3 決策樹回歸

9.4 決策樹的複雜度及使用技巧

9.5 決策樹演算法：ID3、C4.5 和CART

9.6 本章小結

9.7 複習題

 

10 整合方法：從決策樹到隨機森林

10.1 Bagging 元估計器

10.2 由隨機樹組成的森林

10.3 AdaBoost

10.4 梯度提升回歸樹

10.5 本章小結

10.6 複習題

 

11 從感知機到支援向量機

11.1 線性支援向量機分類

11.2 非線性支援向量機分類

11.3 支援向量機回歸

11.4 本章小結

11.5 複習題

 

12 從感知機到類神經網路

12.1 從神經元到類神經元

12.2 感知機

12.3 多層感知機
12.4 本章小結
12.5 複習題

 

13 主成分分析降維
13.1 資料的向量表示及降維問題
13.2 向量的表示及基變換
13.3 協方差矩陣及最佳化目標
13.4 PCA 演算法流程
13.5 PCA 實例
13.6 scikit-learn PCA 降維實例
13.7 核心主成分分析KPCA 簡介
13.8 本章小結
13.9 複習題


A 參考文獻
```
## ch3-2
```python
# -*- coding: utf-8 -*-
print(__doc__)
"""
Created on Wed Jul  7 10:25:49 2021

@author: liguo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()
```
