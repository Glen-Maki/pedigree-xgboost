import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

df_all = pd.read_csv('data/preg_kikai.csv', encoding="Shift-JIS")
print(df_all.head())
print(df_all.columns)

# 系統
type_stallions = []
params_type = ['母父タイプ名', '母母父タイプ名', '父タイプ名', '父母父タイプ名']

for param_type in params_type:
    for type_stallion in df_all[param_type]:
        if type_stallion in type_stallions:
            pass
        else:
            type_stallions.append(type_stallion)

'''
df_type_stallions = pd.DataFrame(type_stallions, columns=['系統'])
df_type_stallions['index'] = [i for i in range(len(type_stallions))]

print(df_type_stallions.head())
'''

# 種牡馬
name_stallions = []
params_name = ['母父名', '母の母の父名', '種牡馬', '父の母の父名']

for param_name in params_name:
    for name_stallion in df_all[param_name]:
        if name_stallion in name_stallions:
            pass
        else:
            name_stallions.append(name_stallion)

'''
df_name_stallions = pd.DataFrame(name_stallions, columns=['種牡馬'])
df_name_stallions['index'] = [i for i in range(len(name_stallions))]
print(df_name_stallions.head())

df_type_stallions.to_csv('data/type_stallions.csv', index=False)
df_name_stallions.to_csv('data/name_stallions.csv', index=False)
'''

f = open('data/name_stallions.txt', 'wb')
pickle.dump(name_stallions, f)

f = open('data/type_stallions.txt', 'wb')
pickle.dump(type_stallions, f)
