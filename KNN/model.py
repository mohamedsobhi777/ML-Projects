#2 Apr 2021
#dataset : http://techwithtim.net/wp-content/uploads/2019/01/Car-Data-Set.zip
import pandas as pd
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model , preprocessing

data = pd.read_csv("car.data" , sep = ",")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform( list(data["buying"]))
maint = le.fit_transform( list(data["maint"]))
door = le.fit_transform( list(data["door"]))
persons = le.fit_transform( list(data["persons"]))
lug_boot = le.fit_transform( list(data["lug_boot"]))
safety = le.fit_transform( list(data["safety"]))
cls = le.fit_transform( list(data["class"]))

predict = "class"


x = list(zip(buying , maint , door , persons , lug_boot , safety))
y = list(cls)

xtrain , xtest , ytrain , ytest\
    = sklearn.model_selection.train_test_split(x , y , test_size= 0.1)
model = KNeighborsClassifier(n_neighbors=8)
model.fit(xtrain , ytrain)

acc = model.score(xtest, ytest)
predicted = model.predict(xtest)

names = ["unacc" , "acc" , "good" , "vgood"]

for x in range(len(xtest)):
    print("Predicted " , names[predicted[x]] , " Data "
          , xtest[x] , " actual " , names[ytest[x]] )
    n = model.kneighbors([xtest[x]] , 8 , True)
    # print(n)
# print(acc*100)

