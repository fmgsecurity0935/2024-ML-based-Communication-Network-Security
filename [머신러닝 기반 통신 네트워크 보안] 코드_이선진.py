# 필요한 패키지 Import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
# 스텝별 정확도 측정용


# 제안모델 정확도 측정
path1 = 'Step/step1_bin1.csv' # 데이터 불러오기
path2 = 'Step/step1_bin2.csv' # 데이터 불러오기
path3 = 'Step/step1_bin3.csv' # 데이터 불러오기

data1 = pd.read_csv(path1)
data2 = pd.read_csv(path2)
data3 = pd.read_csv(path3)

df1 = data1.copy()
df2 = data2.copy()
df3 = data3.copy()

# Data preprocessing phase
Embarked_list = ['process', 'network', 'winevtlog', 'syscheck-registry', 'rootcheck', 'syscheck', 'browser',
                     'ossec-logcollector']
for embarked in Embarked_list:
    df1['detection_type_' + f'{embarked}'] = df1['detection_type'] == embarked
del df1['detection_type']
Embarked_list = ['running', 'NONE', 'LISTEN', 'TIME_WAIT', 'ESTABLISHED', 'stopped', 'CLOSE_WAIT']
for embarked in Embarked_list:
    df1['status_' + f'{embarked}'] = df1['status'] == embarked
del df1['status']
df_for_sample1 = df1.loc[:]

# Data preprocessing phase
Embarked_list = ['process', 'network', 'winevtlog', 'syscheck-registry', 'rootcheck', 'syscheck', 'browser',
                     'ossec-logcollector']
for embarked in Embarked_list:
    df2['detection_type_' + f'{embarked}'] = df2['detection_type'] == embarked
del df2['detection_type']
Embarked_list = ['running', 'NONE', 'LISTEN', 'TIME_WAIT', 'ESTABLISHED', 'stopped', 'CLOSE_WAIT']
for embarked in Embarked_list:
    df2['status_' + f'{embarked}'] = df2['status'] == embarked
del df2['status']
df_for_sample2 = df1.loc[:]

# Data preprocessing phase
Embarked_list = ['process', 'network', 'winevtlog', 'syscheck-registry', 'rootcheck', 'syscheck', 'browser',
                     'ossec-logcollector']
for embarked in Embarked_list:
    df3['detection_type_' + f'{embarked}'] = df3['detection_type'] == embarked
del df3['detection_type']
Embarked_list = ['running', 'NONE', 'LISTEN', 'TIME_WAIT', 'ESTABLISHED', 'stopped', 'CLOSE_WAIT']
for embarked in Embarked_list:
    df3['status_' + f'{embarked}'] = df3['status'] == embarked
del df3['status']
df_for_sample3 = df3.loc[:]

# Learning phase -Decision Tree
# train data 선출 위함(전체 데이터에서 70% 선출)
Y1 = df1[['label']]
df1.drop(df1.columns[0], axis=1, inplace= True)
X1= df1.drop('label', axis=1)
xtrain1, xtest1, ytrain1, ytest1 = train_test_split(X1,Y1, test_size=0.3, random_state=25, shuffle=True)  # train, test 데이터의 7:3 비율을 맞춰주기 위해 test size 통일

Y2 = df2[['label']]
df2.drop(df2.columns[0], axis=1, inplace= True)
X2= df2.drop('label', axis=1)
xtrain2, xtest2, ytrain2, ytest2 = train_test_split(X2,Y2, test_size=0.3, random_state=25, shuffle=True)  # train, test 데이터의 7:3 비율을 맞춰주기 위해 test size 통일


Y3 = df3[['label']]
df3.drop(df3.columns[0], axis=1, inplace= True)
X3= df3.drop('label', axis=1)
xtrain3, xtest3, ytrain3, ytest3 = train_test_split(X3,Y3, test_size=0.3, random_state=25, shuffle=True)  # train, test 데이터의 7:3 비율을 맞춰주기 위해 test size 통일


clf_dt1 = DecisionTreeClassifier(random_state=100)
clf_dt1.fit(xtrain1, ytrain1)
dt_pred1 = clf_dt1.predict(xtest1)    # apt 공격만 탐지하는 경우 xtest2, ytest2로 수정하기
dt_accuracy1 = accuracy_score(ytest1, dt_pred1)
print("step 0-3: {}".format(dt_accuracy1))

clf_dt2 = DecisionTreeClassifier(random_state=100)
clf_dt2.fit(xtrain2, ytrain2)
dt_pred2 = clf_dt2.predict(xtest2)    # apt 공격만 탐지하는 경우 xtest2, ytest2로 수정하기
dt_accuracy2 = accuracy_score(ytest2, dt_pred2)
print("step 4-6: {}".format(dt_accuracy2))

clf_dt3 = DecisionTreeClassifier(random_state=100)
clf_dt3.fit(xtrain3, ytrain3)
dt_pred3 = clf_dt3.predict(xtest3)    # apt 공격만 탐지하는 경우 xtest2, ytest2로 수정하기
dt_accuracy3 = accuracy_score(ytest3, dt_pred3)
print("step 7-9: {}".format(dt_accuracy3))


import sklearn.ensemble as ske
classifier1 = ske.RandomForestClassifier(n_estimators=50)
classifier1.fit(xtrain1, ytrain1.values.ravel())
rf_pred1 = classifier1.predict(xtest1)
rf_accuracy1 = accuracy_score(ytest1, rf_pred1)
print("step 0-3: {}".format(rf_accuracy1))

classifier2 = ske.RandomForestClassifier(n_estimators=50)
classifier2.fit(xtrain2, ytrain2.values.ravel())
rf_pred2 = classifier2.predict(xtest2)
rf_accuracy2 = accuracy_score(ytest2, rf_pred2)
print("step 4-6: {}".format(rf_accuracy2))

classifier3 = ske.RandomForestClassifier(n_estimators=50)
classifier3.fit(xtrain3, ytrain3.values.ravel())
rf_pred3 = classifier3.predict(xtest3)
rf_accuracy3 = accuracy_score(ytest3, rf_pred3)
print("step 7-9: {}".format(rf_accuracy3))

# # LinearSVM 활용
# from sklearn import svm
# clf_svm1 = svm.LinearSVC(C=1, max_iter = 10000) # 학습 반복횟수 10000
# clf_svm1.fit(xtrain1, ytrain1)
# svm_pred1 = clf_svm1.predict(xtest1)
# svm_acc1 = accuracy_score(ytest1, svm_pred1)
# print("step 0-3: {}".format(svm_acc1))
#
# clf_svm2 = svm.LinearSVC(C=1, max_iter = 10000) # 학습 반복횟수 10000
# clf_svm2 .fit(xtrain2, ytrain2)
# svm_pred2 = clf_svm2.predict(xtest2)
# svm_acc2 = accuracy_score(ytest2, svm_pred2)
# print("step 4-6: {}".format(svm_acc2))
#
# clf_svm3 = svm.LinearSVC(C=1, max_iter = 10000) # 학습 반복횟수 10000
# clf_svm3.fit(xtrain3, ytrain3)
# svm_pred3 = clf_svm3.predict(xtest3)
# svm_acc3 = accuracy_score(ytest3, svm_pred3)
# print("step 7-9: {}".format(svm_acc3))

import keras
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()

model.add(Dense(30,input_shape=(30,),activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#import tensorflow as tf
#tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

X1=X1.astype(float)
Y1=Y1.astype(float)
X2=X2.astype(float)
Y2=Y2.astype(float)
X3=X3.astype(float)
Y3=Y3.astype(float)

model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
a = model.fit(X1,Y1,epochs=10)
b = model.fit(X2,Y2,epochs=10)
c = model.fit(X3, Y3, epochs=10)

DL_accuracy1 = model.evaluate(X1, Y1)
DL_accuracy1 = DL_accuracy1[1]

DL_accuracy2 = model.evaluate(X2, Y2)
DL_accuracy2 = DL_accuracy2[1]

DL_accuracy3 = model.evaluate(X3, Y3)
DL_accuracy3 = DL_accuracy3[1]



print(DL_accuracy1, DL_accuracy2, DL_accuracy3)
