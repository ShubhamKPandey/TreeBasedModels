import numpy as np
from numpy import loadtxt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

dataset = loadtxt('pima-indians-diabetes.txt', delimiter = ',')
print(dataset.shape)
print(dataset)

X = dataset[:,0:8]
y = dataset[:,8]

model = XGBClassifier()
seed = 7

kfold = StratifiedKFold(n_splits = 10 , random_state = seed)
results = cross_val_score( model ,X, y, cv = kfold )

print('Accuracy : %.2f%%' %(results.mean()*100))