from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np


# read data
dataframe = read_csv('datasets-uci-breast-cancer.csv', header = None)
print(dataframe.head)
dataset = dataframe.values
print(type(dataset))
print(dataset.shape)
print(dataset.shape)
X = dataset[:, 0:9]
Y = dataset[:, 9]


# Encode string class values as integers
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(Y)
print(encoded_y.shape)
print(type(encoded_y))

# Encode string input values as integers


features = []
for i in range(0, X.shape[1]):
    label_encoder = LabelEncoder()
    features_i = label_encoder.fit_transform(X[:, i])
    features_i = np.array([features_i]).T
    onehotencoder = OneHotEncoder(sparse = False)
    features_i = onehotencoder.fit_transform(features_i)
    features.append(features_i)

encoded_x = np.column_stack(features)

seed = 7
test_size = 0.33 
X_train, X_test, y_train, y_test = train_test_split(encoded_x, encoded_y, test_size = test_size, random_state = seed)

model = XGBClassifier()
model.fit(X_train, y_train)

print(model)

predictions = model.predict(X_test)

accuracy= accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" %(accuracy*100.0))

