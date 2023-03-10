import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.model_selection import RepeatedKFold
import pickle

df_all = pd.read_csv('data/preg_kikai.csv', encoding="Shift-JIS")
f = open('data/type_stallions.txt', 'rb')
type_stallions = pickle.load(f)

f = open('data/name_stallions.txt', 'rb')
name_stallions = pickle.load(f)

print(df_all.head())
print(name_stallions)
print(df_all.columns)


def categorize_type(type):
    if type in type_stallions:
        return int(type_stallions.index(type))
    else:
        return np.nan


def categorize_name(name):
    if name in name_stallions:
        return int(name_stallions.index(name))
    else:
        return np.nan


def categorize_money(money):
    if money == 0:
        return 0
    elif 500 >= money > 0:
        return 1
    elif 1000 >= money >= 501:
        return 2
    elif 1600 >= money >= 1001:
        return 3
    elif money > 1600:
        return 4


for e in ['母父タイプ名', '母母父タイプ名', '父タイプ名', '父母父タイプ名']:
    df_all[e] = df_all[e].apply(categorize_type)

for e in ['母父名', '母の母の父名']:
    df_all[e] = df_all[e].apply(categorize_name)

df_all['クラス'] = df_all['収得賞金'].apply(categorize_money)

print(df_all.head())

X = df_all.loc[:, ~df_all.columns.isin(['本賞金', '収得賞金', '母年齢', '種牡馬', '父の母の父名', 'クラス'])]
y = df_all['クラス']

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)

model = xgb.XGBRegressor(verbosity=0)

params = {'booster': ['gbtree'], 'n_estimators': [10, 30, 50, 80, 100], 'max_depth': [2, 3, 4, 5, 6],
          'learning_rate': [0.01, 0.03, 0.1, 0.3], 'colsample_bytree': [0.1, 0.25, 0.5, 0.75, 1.0],
          'random_state': [0], "objective": ["multi:softmax"], "num_class": [5], 'tree_method': ['gpu_hist']
          }

gs = GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring='f1_micro')
gs.fit(X_train, y_train)

print('best score: {:0.3f}'.format(gs.score(X_test, y_test)))
print('best params: {}'.format(gs.best_params_))
print('best val score:  {:0.3f}'.format(gs.best_score_))

'''
best score: 0.611
best params: {'booster': 'gbtree', 'colsample_bytree': 0.5,
 'learning_rate': 0.3, 'max_depth': 2, 'n_estimators': 80, 
 'num_class': 5, 'objective': 'multi:softmax', 'random_state': 0, 
 'tree_method': 'gpu_hist'}
best val score:  0.624
'''