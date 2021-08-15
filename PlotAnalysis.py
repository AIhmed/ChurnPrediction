import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as fn 
from torch.utils.data import Dataset , DataLoader

trainSet=pd.read_csv('sample_data/churn-bigml-80.csv')
testSet=pd.read_csv('sample_data/churn-bigml-20.csv')
print(testSet.columns)
group=trainSet.groupby('Churn')

x_axis=group['State'].value_counts().loc[True].keys().values
churn_rate=group['State'].value_counts().loc[True].values
plt.title('Churn rate by State')
plt.xlabel('State')
plt.ylabel('Churned')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.bar(x_axis,churn_rate)
fig.savefig('sample_data/Churned_byState.png')

x_axis=group['State'].value_counts().loc[False].keys().values
churn_rate=group['State'].value_counts().loc[False].values
plt.title('Churn rate by State')
plt.xlabel('State')
plt.ylabel('not Churned')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.bar(x_axis,churn_rate)
fig.savefig('sample_data/fig/NoneChurned_byState.png')

x_axis=group['International plan'].value_counts().loc[True].keys().values
churned=group['International plan'].value_counts().loc[True].values
notChurned=group['International plan'].value_counts().loc[False].values

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)

ax1.bar(x_axis,churned)
ax2.bar(x_axis,notChurned)
fig.set_size_inches(15, 10)
ax1.set_title('Churn rate by international plan')
ax1.set_xlabel('International plan')
ax1.set_ylabel('Churned')


ax2.set_title('Churn rate by international plan')
ax2.set_xlabel('International plan')
ax2.set_ylabel('not Churned')
plt.show()
fig.savefig('sample_data/churn_rate_by_international_plan.png')
print(churned)
print(notChurned)

x_axis=group['Voice mail plan'].value_counts().loc[True].keys().values
churned=group['Voice mail plan'].value_counts().loc[True].values
notChurned=group['Voice mail plan'].value_counts().loc[False].values

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharex=True)

ax1.bar(x_axis,churned)
ax2.bar(x_axis,notChurned)
fig.set_size_inches(15, 10)
ax1.set_title('Churn rate by Voice mail plan')
ax1.set_xlabel('Voice mail plan')
ax1.set_ylabel('Churned')



ax2.set_title('Churn rate by Voice mail plan')
ax2.set_xlabel('Voice mail plan')
ax2.set_ylabel('not Churned')
plt.show()
fig.savefig('sample_data/fig/churn_rate_byVoiceMailPlan.png')

x1_axis=group['Customer service calls'].value_counts().loc[True].keys().values
x2_axis=group['Customer service calls'].value_counts().loc[False].keys().values
churned=group['Customer service calls'].value_counts().loc[True].values
notChurned=group['Customer service calls'].value_counts().loc[False].values

fig , (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(15,10)
print(x1_axis,x2_axis,churned,notChurned)
ax1.bar(x1_axis,churned)
ax1.set_title('Churn rate by Customer service calls')
ax1.set_xlabel('Customer service calls')
ax1.set_ylabel('Churned')

ax2.bar(x2_axis,notChurned)
ax2.set_title('Churn rate by Customer service calls')
ax2.set_xlabel('Customer service calls')
ax2.set_ylabel('not Churned')
fig.savefig('sample_data/fig/churn_rate_byCustomerServiceCalls.png')

group['Account length'].describe()

group['Area code'].describe()

group['Number vmail messages'].describe()

group['Total day minutes','Total day calls', 'Total day charge'].describe()
group['Total eve minutes','Total eve calls', 'Total eve charge'].describe()
group['Total night minutes','Total night calls', 'Total night charge'].describe()
group['Total intl minutes','Total intl calls', 'Total intl charge'].describe()

