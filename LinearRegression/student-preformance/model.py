# Date 1 Apr 2021
# data set : https://archive.ics.uci.edu/ml/datasets/Student+Performance
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model

# Load the Data
data = pd.read_csv("student-mat.csv" , sep = ";")

# select the features of input and output
predict = "G3"
data = data[["G1" , "G2"  , "G3" , "studytime" , "failures" , "absences" ]]

# storing the data in arrays
x = np.array(data.drop(["G3"] , 1))
y = np.array(data[predict])

# splitting the data
xtrain ,xtest,ytrain,ytest = sklearn.model_selection.train_test_split(x , y ,test_size= 0.2)

# training the model
linear = linear_model.LinearRegression()
linear.fit(xtrain , ytrain)
acc = linear.score(xtest , ytest)

# print("Co \n" ,  linear.coef_)
# print("Intercept \n" , linear.intercept_)

# printing accuracy and predictions
print("accuracy =" ,acc*100 , "%")

predictions = linear.predict(xtest)

for i , G3 in enumerate(predictions):
    print("student " , i + 1 , " whose G1 and G2 (respectively) are "
          , data["G1"][i], ',' , data["G2"][i]
          , " is predicted to score " , G3 , " in their G3 and they got " , ytest[i])

