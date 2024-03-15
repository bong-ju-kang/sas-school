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
                             max_depth=20, n_jobs=-1)
print(clf)
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=20, max_features='auto', max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=5,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             n_estimators=100, n_jobs=-1, oob_score=False, random_state=123,
#             verbose=0, warm_start=False)

# 모델 적합
clf.fit(X_train, y_train)

## 모델 평가: 정확도
# 훈련 데이터
print('훈련데이터 정확도:', clf.score(X_train, y_train))
# 평가 데이터
print('평가데이터 정확도:', clf.score(X_test, y_test))


## SAS에서 지정한 형태로 출력물 정의
##  범주형인 경우
tmp = clf.predict_proba(X)

# 목표변수의 유일한 값별로 변수 구성
target_values = sorted(dm_inputdf[dm_dec_target].unique())
target_variables = ["P_"+dm_dec_target+str(x) for x in target_values]

# 추정된 각 점수(확률)에 대한 값별 변수이름 지정
dm_scoreddf = pd.DataFrame(tmp, columns=target_variables)
