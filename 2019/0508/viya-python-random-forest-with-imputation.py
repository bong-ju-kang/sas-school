# Bong Ju Kang
# for for random forest sample with missing imputation
# 5/8/2019
 
# 필요한 패키지 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
 
 
# 목표 변수 분포 확인 
print(dm_inputdf[dm_dec_target].value_counts()) 
# 0    5430 
# 1    2001 
 
base_dist = dm_inputdf[dm_dec_target].value_counts()/dm_inputdf.shape[0] 
print(base_dist) 
# 0    0.730268 
# 1    0.269732 
 
# 범주형 데이터를 숫자형 데이터로 전환 
df_X_onehot = pd.get_dummies(dm_inputdf[dm_class_input], prefix_sep='_') 
df_y_onehot = pd.get_dummies(dm_inputdf[dm_dec_target], drop_first=True) 
 
# 범주형 데이터와 숫자형 데이터 결합 
train_flag = (dm_inputdf[dm_partitionvar].values == dm_partition_train_val) 
X = np.c_[dm_inputdf[dm_interval_input].values, df_X_onehot.values] 
y = df_y_onehot.values.ravel() 
 
# 데이터 분할 
X_train = X[train_flag] 
y_train = y[train_flag] 
X_test = X[~train_flag] 
y_test = y[~train_flag] 
 
# 모델 구성: 나무는 100개 사용하며, 분기 변수의 개수는 자동 
clf = RandomForestClassifier(random_state=123, n_estimators=100, min_samples_leaf=5, 


