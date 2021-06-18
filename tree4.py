from pandas import read_csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

dataframe = read_csv('horse-colic.csv', delim_whitespace = True, header = None)
print(dataframe.head)

dataset = np.array(dataframe)
print(dataset.shape)

X = dataset[:, 0:27]
y = dataset[:, 27]

labelencoder = LabelEncoder()
y_encoded = labelencoder.fit_transform(y)

X[ X == '?'] = np.nan
X = X.astype(float)
seed = 42
size = 0.33

X_train, X_test, y_train, y_test = train_test_split(X,y_encoded, random_state = seed, test_size = size)

model = XGBClassifier()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print(accuracy)






