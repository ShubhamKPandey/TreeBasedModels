import numpy as np
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

df = read_csv('pima-indians-diabetes.txt')

dataset = np.array(df)
X = dataset[:,0:8]
y = dataset[:,8]

seed = 5

model = XGBClassifier()

kfold = KFold(n_splits = 10, random_state = 7)
results = cross_val_score(model, X, y, cv = kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
