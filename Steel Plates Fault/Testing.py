
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv("train.csv")
dft = pd.read_csv("test.csv")
#importing both train and test datasets


df["X"] = df["X_Maximum"] - df["X_Minimum"]
df["Y"] = df["Y_Maximum"] - df["Y_Minimum"]
df["Luminosity"] = df["Maximum_of_Luminosity"] - df["Minimum_of_Luminosity"]
del df["TypeOfSteel_A400"]

dft["X"] = dft["X_Maximum"] - dft["X_Minimum"]
dft["Y"] = dft["Y_Maximum"] - dft["Y_Minimum"]
dft["Luminosity"] = dft["Maximum_of_Luminosity"] - dft["Minimum_of_Luminosity"]
del dft["TypeOfSteel_A400"]

#modifying features as we did before in Training_models.py


del df["X_Maximum"]
del df["Y_Maximum"]
del df["Maximum_of_Luminosity"]

del dft["X_Maximum"]
del dft["Y_Maximum"]
del dft["Maximum_of_Luminosity"]

del df["X_Perimeter"]
del df["Y_Perimeter"]
del df["Minimum_of_Luminosity"]
del df["Outside_Global_Index"]
del df["Orientation_Index"]
del df["Luminosity_Index"]
del df["Y"]

del dft["X_Perimeter"]
del dft["Y_Perimeter"]
del dft["Minimum_of_Luminosity"]
del dft["Outside_Global_Index"]
del dft["Orientation_Index"]
del dft["Luminosity_Index"]
del dft["Y"]
#Deleting features as we did before

#Selecting our class label
y = df["Faults"]
del df["Faults"]
X = df

#Xt is the input of test dataset
Xt = dft


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


#Fitting XGBoost to the training data
from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth = 50, objective = 'binary:logistic')
classifier.fit(X, y)


y_pred_test = classifier.predict(Xt)
#predicting the ouptput for test data


print(y_pred_test)

output = pd.DataFrame(columns = ["Faults"])
output["Faults"] = y_pred_test
output.to_csv("Output.csv")
#saving the output to Output.csv

