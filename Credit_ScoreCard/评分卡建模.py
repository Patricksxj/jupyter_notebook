#!/usr/bin/env python
# coding: utf-8

# In[11]:


import seaborn as sns
import plotly
import numpy as np
import pandas as pd
import xgboost


# ### 导入数据

# In[12]:


df=pd.read_csv('.\data\cs-training.csv')


# ### 看数据分布

# In[22]:


df=df[['SeriousDlqin2yrs','RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']]


# In[23]:


df.describe()


# In[ ]:





# ## 1.重命名

# In[26]:


df.rename(columns={'SeriousDlqin2yrs':'target',
'RevolvingUtilizationOfUnsecuredLines':'credit_used_rate',
'age':'age',
'NumberOfTime30-59DaysPastDueNotWorse':'has_30_59_due_times',
'DebtRatio':'debt_ratio',
'MonthlyIncome':'income',
'NumberOfOpenCreditLinesAndLoans':'credit_loan_num',
'NumberOfTimes90DaysLate':'has_90_plus_due_times',
'NumberRealEstateLoansOrLines':'realestate_loan_num',
'NumberOfTime60-89DaysPastDueNotWorse':'has_60_89_due_times',
'NumberOfDependents':'relatives_num'},inplace=True)


# ### 2.看数据分布

# In[29]:


df.describe()


# In[32]:


print("bad_rate:",df['target'].sum()*1.00/df['target'].count())


# ### 3.寻找缺失变量
# - relatives_num
# - income

# In[40]:


df.isna().sum()


# In[63]:


## 如下方法不对，应该count不会统计缺失值，所以应该用len
df.isna().sum()/df.count()


# In[62]:


df.isna().sum()/len(df)


# ### 4. 数据填充

# In[54]:


df['relatives_num_fillna']=df.relatives_num.fillna(df.relatives_num.median())


# In[56]:


df['income_fillna']=df.income.fillna(df.income.median())


# In[57]:


df.isna().sum()/df.count()


# In[71]:


var_list=['credit_used_rate',
'age',
'has_30_59_due_times',
'debt_ratio',
'income_fillna',
'credit_loan_num',
'has_90_plus_due_times',
'realestate_loan_num',
'has_60_89_due_times',
'relatives_num_fillna']


# ### 5.寻找异常值

# #### 异常值结果：

# In[ ]:


'''
credit_used_rate>15
debt_ratio 超过98分位数

'''


# In[65]:


df[var_list].describe()


# In[82]:


sns.distplot(df['credit_used_rate'])


# In[99]:


#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', 200)

#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)


# In[100]:


### 使用率超过15倍的定义为异常值，需要剔除
df[df['credit_used_rate']>15]['credit_used_rate']


# In[74]:


sns.distplot(df['age'])


# In[89]:


df[df['age']>100]['age']


# ### debt_ratio 字段

# In[109]:


len(df[df['debt_ratio']>500])/len(df)


# In[107]:


df["debt_ratio"].quantile(0.95)


# In[93]:


df.select_dtypes(include=['number']).columns


# In[167]:


df["age"].value_counts().sort_index()


# ### 6.排除异常区间的数据

# In[110]:


df = df.loc[df["debt_ratio"] <= df["debt_ratio"].quantile(0.98)]
df = df.loc[(df["credit_used_rate"] >= 0) & (df["credit_used_rate"] <= 15)]


# In[111]:


df.describe()


# ## 二、特征工程

# ### 7.分桶

# In[113]:


import math
age_bins = [-math.inf, 25, 40, 50, 60, 70, math.inf]
df['bin_age'] = pd.cut(df['age'],bins=age_bins).astype(str)
relatives_bin = [-math.inf,2,4,6,8,10,math.inf]
df['bin_relatives_num_fillna'] = pd.cut(df['relatives_num_fillna'],bins=relatives_bin).astype(str)
dpd_bins = [-math.inf,1,2,3,4,5,6,7,8,9,math.inf]
df['bin_has_90_plus_due_times'] = pd.cut(df['has_90_plus_due_times'],bins=dpd_bins)
df['bin_has_30_59_due_times'] = pd.cut(df['has_30_59_due_times'], bins=dpd_bins)
df['bin_has_60_89_due_times'] = pd.cut(df['has_60_89_due_times'], bins=dpd_bins)

df['bin_credit_used_rate'] = pd.qcut(df['credit_used_rate'],q=5,duplicates='drop').astype(str)
df['bin_debt_ratio'] = pd.qcut(df['debt_ratio'],q=5,duplicates='drop').astype(str)
df['bin_income_fillna'] = pd.qcut(df['income_fillna'],q=5,duplicates='drop').astype(str)
df['bin_credit_loan_num'] = pd.qcut(df['credit_loan_num'],q=5,duplicates='drop').astype(str)
df['bin_realestate_loan_num'] = pd.qcut(df['realestate_loan_num'],q=5,duplicates='drop').astype(str)


# In[117]:


bin_var_list=[c for c in df.columns.values if c.startswith('bin')]


# In[118]:


bin_var_list


# In[119]:


df[bin_var_list].head()


# ### 8.对变量筛选，计算iv值，用iv大于0.2的变量，并进行woe编码

# In[130]:


def cal_IV(df, feature, target):
    lst = []
    cols=['Variable', 'Value', 'All', 'Bad']
    for i in range(df[feature].nunique()):
        
        val = list(df[feature].unique())[i]
        #变量名字，val表示对应的分类值，比如'(40.0, 50.0]',第3个参数是某一变量为特定分桶的数据行数，第四个参数是某一变量为特定分桶且target为1的数量
        lst.append([feature, val, df[df[feature] == val].count()[feature], df[(df[feature] == val) & (df[target] == 1)].count()[feature]])
#     print(lst)
    data = pd.DataFrame(lst, columns=cols)
    data = data[data['Bad'] > 0]

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    data['IV'] = (data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])).sum()

    data = data.sort_values(by=['Variable', 'Value'], ascending=True)

    return data['IV'].values[0]


# In[132]:


for f in bin_var_list:
    print("var_name:",f,cal_IV(df,f,'target'), 1 if cal_IV(df,f,'target')>=0.2 else 0)


# In[133]:


feature=['bin_age','bin_has_90_plus_due_times','bin_has_30_59_due_times','bin_has_60_89_due_times','bin_credit_used_rate']


# In[127]:


df[df['bin_age']=='(40.0, 50.0]'].count()['bin_age']


# In[129]:


df[(df['bin_age']=='(40.0, 50.0]')&(df['target']==1)].count()['bin_age']


# ### 计算woe

# In[134]:


def cal_WOE(df,features,target):
    df_new = df
    for f in features:
        df_woe = df_new.groupby(f).agg({target:['sum','count']})
        df_woe.columns = list(map(''.join, df_woe.columns.values))
        df_woe = df_woe.reset_index()
        df_woe = df_woe.rename(columns = {target+'sum':'bad'})
        df_woe = df_woe.rename(columns = {target+'count':'all'})
        df_woe['good'] = df_woe['all']-df_woe['bad']
        df_woe = df_woe[[f,'good','bad']]
        df_woe['bad_rate'] = df_woe['bad']/df_woe['bad'].sum()
        df_woe['good_rate'] = df_woe['good']/df_woe['good'].sum()
        df_woe['woe'] = df_woe['bad_rate'].divide(df_woe['good_rate'],fill_value=1)
        df_woe.columns = [c if c==f else c+'_'+f for c in list(df_woe.columns.values)]
        df_new = df_new.merge(df_woe,on=f,how='left')
    return df_new


# In[135]:


# 我们根据IV选出来的特征
df_woe = cal_WOE(df,feature,'target')
woe_cols = [c for c in list(df_woe.columns.values) if 'woe' in c]
df_woe[woe_cols]


# In[166]:


df_woe.head(10)


# In[ ]:





# In[137]:


df_woe[woe_cols]


# In[ ]:


'''
df_bin_to_woe = pd.DataFrame(columns = ['features','bin','woe'])
for f in feature_cols:
    b = 'bin_'+f
    w = 'woe_bin_'+f
    df = df_woe[[w,b]].drop_duplicates()
    df.columns = ['woe','bin']
    df['features'] = f
    df=df[['features','bin','woe']]
    df_bin_to_woe = pd.concat([df_bin_to_woe,df])
    df_bin_to_woe
'''


# In[ ]:





# ## 三、建模

# ### 9.开始建模

# In[138]:


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


# In[139]:


X_train, X_test, y_train, y_test = train_test_split(df_woe[woe_cols], df_woe['target'], test_size=0.3, random_state=42)


# In[142]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42).fit(X_train,y_train)


# In[140]:


TRAIN_CONFIGS = {
    "model_params": {
        'min_child_weight': list(range(1,10,2)),
        'gamma':[i/10.0 for i in range(0,5)],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree':[i/10.0 for i in range(6,10)],
        'max_depth':range(3,10,2),
        'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
        'reg_lambda': [0, 0.001, 0.005, 0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 1],
        'learning_rate': [0.001, 0.002, 0.005, 0.006, 0.01, 0.02, 0.05, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2],
        'n_estimators': [50, 100, 150, 200, 250, 300,350,400,450,500, 550, 600, 650, 700, 750],
        "booster": ["gbtree", "gblinear", "dart"]
        }
}


# In[ ]:


# Train Model
print("Training Model...")
xgb = XGBClassifier(random_state=0)
model = RandomizedSearchCV(xgb, param_distributions=TRAIN_CONFIGS["model_params"], n_iter=400, scoring='roc_auc', n_jobs=-1, cv=StratifiedKFold(n_splits=5, shuffle = True, random_state = 0), verbose=3, random_state=0)
model.fit(X_train, y_train, eval_metric="auc")
print("Done...\n")


# ### 10.保存模型

# In[147]:


import os
os.getcwd()


# In[149]:


# Save Model
import joblib
print("Saving Model...")
joblib.dump(model, open('./model_output/model.p', "wb"))
print("Completed!")


# In[ ]:





# ## 四、对测试数据评估

# In[150]:


#在验证集上看性能
model.score(X_test,y_test)


# In[151]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(X_test)

preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
ks=max(tpr-fpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:





# In[154]:


#混淆矩阵
y_pred = model.predict(X_test)


# In[155]:


metrics.confusion_matrix(y_test,y_pred)


# In[156]:


model.coef_


# In[ ]:





# In[ ]:





# In[ ]:





# ### 对训练数据集作平衡处理,暂时先不处理再平衡

# In[160]:


#!pip install imblearn
from imblearn.over_sampling import SMOTE
over_samples = SMOTE(random_state=1234) 
over_samples_X,over_samples_y = over_samples.fit_resample(X_train, y_train)


# In[161]:


over_samples_model = LogisticRegression(random_state=42).fit(over_samples_X,over_samples_y)


# In[163]:


#在验证集上看性能
over_samples_model.score(X_test,y_test)


# In[164]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = over_samples_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:





# ### 需要计算A和B的话，我们设置两个参数
# 1. 基准分。我们设θ_0为20：1时的多数风控策略基准分都设置为650，我们就试试650吧，基准分为A-B\theta_0
# 2. PDO（point of double），比率翻番时分数的变动值。假设我们设置为当odds翻倍时，分值减少30。
# 
# 而A、B计算公式（推导过程详见我的知乎专栏 https://zhuanlan.zhihu.com/p/148102950）
# 
# ![image.png](attachment:image.png)
# 

# In[ ]:





# In[ ]:


for index,row in df.iterrows():
    print(row)

