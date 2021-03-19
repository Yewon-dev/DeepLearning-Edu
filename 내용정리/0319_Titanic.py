#!/usr/bin/env python
# coding: utf-8

# ## 타이타닉 생존자 예측
# - 데이터 읽기
# - 데이터 전처리
# - 학습용 데이터 만들기
# - 모델생성
# - 모델검증
# - 모델향상

# In[149]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder #라벨링
from sklearn.model_selection import train_test_split # train/test데이터 나누기
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


# In[2]:


raw_df = pd.read_csv('titanic_train.csv')


# In[3]:


raw_df.info()
# Age도 확인해야하고
# Cabin이 204개네.. 687개가 null...?
# raw data는 건들지 말자 -> cl_df
# PassengerId 지워도 됨


# In[7]:


cleansing_feature = ['PassengerId', 'Name']
cl_df = raw_df.drop(cleansing_feature, axis=1)


# In[8]:


# Embarked 정리
cl_df.Embarked.fillna('N', inplace=True)


# In[10]:


cl_df.info()


# In[19]:


# cabin 정리
cl_df.Cabin.fillna('N', inplace=True)


# In[21]:


cl_df['Cabin_tmp'] = cl_df.Cabin.str[:1]


# In[17]:


# Age 정리
cl_df.Age.fillna(round(cl_df.Age.mean(),1), inplace=True) # Nan을 평균값으로 대체, 


# In[18]:


cl_df.info()


# In[23]:


#
gender_encoder = LabelEncoder()
gender_encoder.fit(cl_df.Sex)

gender = gender_encoder.transform(cl_df.Sex) # 0과 1로 라벨링


# In[24]:


gender


# In[25]:


cl_df['gender'] = gender # 대입


# In[26]:


cl_df.info()


# In[27]:


Cabin_encoder = LabelEncoder()
Cabin_encoder.fit(cl_df.Cabin_tmp)
Cabin_class = Cabin_encoder.transform(cl_df.Cabin_tmp)

Embarked_encoder = LabelEncoder()
Embarked_encoder.fit(cl_df.Embarked)
Embarked_class = Embarked_encoder.transform(cl_df.Embarked)


# In[28]:


cl_df['Cabin_class'] = Cabin_class
cl_df['Embarked_class'] = Embarked_class


# In[31]:


cl_df.info()


# ## 최종 정리

# In[33]:


feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'gender', 'Cabin_class', 'Embarked_class']
y_col = ['Survived']

X = cl_df[feature_cols]
y = cl_df[y_col]


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[37]:


tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)


# In[38]:


tree_pred = tree_model.predict(X_test)


# In[39]:


accuracy_score(y_test, tree_pred)


# In[41]:


svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_pred = svc_model.predict(X_test)
accuracy_score(y_test, svc_pred)


# In[43]:


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
accuracy_score(y_test, rf_pred)


# In[50]:


lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
accuracy_score(y_test, lr_pred)


# In[51]:


kn_model = KNeighborsClassifier()
kn_model.fit(X_train, y_train)
kn_pred = kn_model.predict(X_test)
accuracy_score(y_test, kn_pred)


# In[58]:


lr_model.predict(np.array(X_test.iloc[0,:]).reshape(1,-1))


# In[59]:


lr_model.predict_proba(np.array(X_test.iloc[0,:]).reshape(1,-1))


# In[52]:


y_test


# ## HARD VOTING
# - 각 분류기마다 모델생성
# - 분류기부터 예측결과 취합
# - 분류기가 예측한 투표 결과를 다수결로 하여 예측결과 산정
# - accuracy_score(y_test,예측결과)을 이용하여 결과값 산출

# In[120]:


v1 = RandomForestClassifier()
v2 = LogisticRegression()
v3 = KNeighborsClassifier()


# In[121]:


v1.fit(X_train, y_train)
v2.fit(X_train, y_train)
v3.fit(X_train, y_train)


# In[122]:


v1_pred = v1.predict(X_test)
v2_pred = v2.predict(X_test)
v3_pred = v3.predict(X_test)


# In[123]:


v1_pred


# In[124]:


#result = sum(v1_pred.reshape(1,-1),(sum(v2_pred.reshape(1,-1), v3_pred.reshape(1,-1))))
result = v1_pred + v2_pred + v3_pred
hv = np.where(result >=2, 1, 0)


# In[125]:


accuracy_score(y_test, hv)


# ### SOFT VOTING
# - 각 분류기마다 모델생성
# - 분류기부터 예측결과 취합(확률값)
# - 분류기가 예측한 투표 결과를 확률의 기대값으로 예측결과 산정
# - accuracy_score(y_test,예측결과)을 이용하여 결과값 산출

# In[126]:


v1 = RandomForestClassifier()
v2 = LogisticRegression()
v3 = KNeighborsClassifier()

v1.fit(X_train, y_train)
v2.fit(X_train, y_train)
v3.fit(X_train, y_train)


# In[127]:


v1_pred = v1.predict_proba(X_test)
v2_pred = v2.predict_proba(X_test)
v3_pred = v3.predict_proba(X_test)


# In[128]:


v1_pred


# In[129]:


(v1_pred + v2_pred + v3_pred).shape


# In[131]:


# 죽을 확률에서 살 확률 빼기 
sv = np.where((v1_pred + v2_pred + v3_pred)[:,0] - (v1_pred + v2_pred + v3_pred)[:,1] > 0
        ,0
        ,1)


# In[132]:


accuracy_score(y_test, sv)


# In[ ]:





# In[167]:


c1 = RandomForestClassifier()
c2 = LogisticRegression()
c3 = KNeighborsClassifier()
c4 = RandomForestClassifier()
c5 = KNeighborsClassifier()


# In[168]:


c1.fit(X_train, y_train)
c2.fit(X_train, y_train)
c3.fit(X_train, y_train)
c4.fit(X_train, y_train)
c5.fit(X_train, y_train)


# In[169]:


c1_pred = c1.predict(X_test)
c2_pred = c2.predict(X_test)
c3_pred = c3.predict(X_test)
c4_pred = c4.predict(X_test)
c5_pred = c5.predict(X_test)


# In[170]:


result = c1_pred + c2_pred + c3_pred + c4_pred + c5_pred
hv = np.where(result >=3, 1, 0)


# In[171]:


accuracy_score(y_test, hv)


# In[172]:


c1_pred = c1.predict_proba(X_test)
c2_pred = c2.predict_proba(X_test)
c3_pred = c3.predict_proba(X_test)
c4_pred = c4.predict_proba(X_test)
c5_pred = c5.predict_proba(X_test)


# In[173]:


sv = np.where((c1_pred + c2_pred + c3_pred + c4_pred + c5_pred)[:,0] - (c1_pred + c2_pred + c3_pred + c4_pred + c5_pred)[:,1] > 0
        ,0
        ,1)


# In[174]:


accuracy_score(y_test, sv)


# In[ ]:


#######################################


# In[189]:


rf = RandomForestClassifier()
lr = LogisticRegression()
kn = KNeighborsClassifier()
meta = DecisionTreeClassifier()


# In[193]:


rf.fit(X_train, y_train)
lr.fit(X_train, y_train)
kn.fit(X_train, y_train)
rf_result = rf.predict(X_train)
lr_result = lr.predict(X_train)
kn_result = kn.predict(X_train)


# In[194]:


temp = pd.DataFrame()
temp['rf_result'] = rf_result
temp['lr_result'] = lr_result
temp['kn_result'] = kn_result


# In[195]:


temp


# In[196]:


meta.fit(temp.values, y_train.values)


# In[ ]:


meta.predict(temp.values)


# In[ ]:


#########


# In[198]:


rf_result = rf.predict(X_test)
lr_result = lr.predict(X_test)
kn_result = kn.predict(X_test)


# In[199]:


temp = pd.DataFrame()
temp['rf_result'] = rf_result
temp['lr_result'] = lr_result
temp['kn_result'] = kn_result


# In[200]:


st_result = meta.predict(temp.values)


# In[220]:


accuracy_score(y_test,st_result)


# In[ ]:





# In[ ]:





# In[219]:


#########################################


# In[ ]:





# In[ ]:





# In[209]:


rf = RandomForestClassifier()
tree = DecisionTreeClassifier()
kn = KNeighborsClassifier()
meta = LogisticRegression()


# In[210]:


rf.fit(X_train, y_train)
tree.fit(X_train, y_train)
kn.fit(X_train, y_train)
rf_result = rf.predict(X_train)
tree_result = tree.predict(X_train)
kn_result = kn.predict(X_train)


# In[217]:


temp = pd.DataFrame()
temp['rf_result'] = rf_result
temp['tree_result'] = tree_result
temp['kn_result'] = kn_result


# In[218]:


temp


# In[213]:


meta.fit(temp.values, y_train.values)


# In[214]:


meta.predict(temp.values)


# In[221]:


rf_result = rf.predict(X_test)
tree_result = tree.predict(X_test)
kn_result = kn.predict(X_test)

temp = pd.DataFrame()
temp['rf_result'] = rf_result
temp['tree_result'] = tree_result
temp['kn_result'] = kn_result

st_result = meta.predict(temp.values)
accuracy_score(y_test,st_result)


# In[ ]:





# In[ ]:


############################ proba


# In[222]:


rf.fit(X_train, y_train)
tree.fit(X_train, y_train)
kn.fit(X_train, y_train)
rf_result = rf.predict_proba(X_train)[:, 0] -  rf.predict_proba(X_train)[:, 1]
tree_result = tree.predict_proba(X_train)[:, 0] - tree.predict_proba(X_train)[:, 1]
kn_result = kn.predict_proba(X_train)[:, 0] - kn.predict_proba(X_train)[:, 1]


# In[223]:


temp = pd.DataFrame()
temp['rf_result'] = rf_result
temp['tree_result'] = tree_result
temp['kn_result'] = kn_result


# In[224]:


temp


# In[225]:


meta.fit(temp.values, y_train.values)


# In[226]:


rf_result = rf.predict_proba(X_test)[:, 0] -  rf.predict_proba(X_test)[:, 1]
tree_result = tree.predict_proba(X_test)[:, 0] - tree.predict_proba(X_test)[:, 1]
kn_result = kn.predict_proba(X_test)[:, 0] - kn.predict_proba(X_test)[:, 1]


# In[227]:


temp = pd.DataFrame()
temp['rf_result'] = rf_result
temp['tree_result'] = tree_result
temp['kn_result'] = kn_result

temp


# In[228]:


st_result = meta.predict(temp.values)
accuracy_score(y_test, st_result)

