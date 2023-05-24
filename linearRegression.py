import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
#select parameters
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict ="G3"

X = np.array(data.drop(predict, axis=1))
y = np.array(data["G3"])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()
#create fit line
linear.fit(x_train, y_train)
#test accuracy
acc = linear.score(x_test, y_test)
print(acc)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)
#Predictions loop
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])