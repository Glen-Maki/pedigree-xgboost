import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

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

'''
best score: 0.611
best params: {'booster': 'gbtree', 'colsample_bytree': 0.5,
 'learning_rate': 0.3, 'max_depth': 2, 'n_estimators': 80, 
 'num_class': 5, 'objective': 'multi:softmax', 'random_state': 0, 
 'tree_method': 'gpu_hist'}
best val score:  0.624
'''

model = xgb.XGBRegressor(verbosity=0, booster='gbtree', colsample_bytree=0.5,
                         learning_rate=0.3, max_depth=2, n_estimators=80, num_class=5,
                         objective='multi:softmax', random_state=0, tree_method='gpu_hist')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)

print(classification_report(y_test, y_pred))

'''
file_path = 'model/pedigree.pkl'

pickle.dump(model, open(file_path, 'wb'))
'''


