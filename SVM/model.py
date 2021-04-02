#2 Apr 2021

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

xtrain , xtest , ytrain , ytest = \
    sklearn.model_selection.train_test_split(x ,y , test_size=0.1)

classes = ["malignant" , "benign"]

clf = svm.SVC(kernel ="linear" , C=1)
clf.fit(xtrain , ytrain)

y_pred = clf.predict(xtest)

acc = metrics.accuracy_score(ytest , y_pred)
print(acc*100)
