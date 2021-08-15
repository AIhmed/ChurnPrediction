import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as fn 
from torch.utils.data import Dataset , DataLoader


trainSet=pd.read_csv('sample_data/churn-bigml-80.csv')
testSet=pd.read_csv('sample_data/churn-bigml-20.csv')
print(testSet.columns)
valc=trainSet.loc[trainSet['Churn']==True,'International plan'].value_counts().values
valn=trainSet.loc[trainSet['Churn']==False,'International plan'].value_counts().values
np.array([valc,valn])

group=trainSet.groupby('Churn')
group['Account length', 'Area code'].describe()
group['Total night charge','Total night minutes'].describe()
group['Total eve charge','Total eve minutes'].describe()
group['Total intl charge','Total intl calls'].describe()

def churn_to_num(churn):
  if churn==False:
    return torch.tensor(0,dtype=torch.long)
  else:
    return torch.tensor(1,dtype=torch.long)

def category_to_num(val):
  if val=='No':
    return torch.tensor(0,dtype=torch.int8)
  else:
    return torch.tensor(1,dtype=torch.int8)
def get_correct_pred(pred,target):
  return pred.argmax(dim=1).eq(target).sum()

night=trainSet[['Total night minutes','Total night calls','Total night charge']]
day=trainSet[['Total day minutes','Total day calls','Total day charge']]
evening=trainSet[['Total eve minutes','Total eve calls','Total eve charge']]
intl=trainSet[['Total intl minutes','Total intl calls','Total intl charge']]

day_tensor=torch.tensor([day['Total day minutes'].values, day['Total day calls'].values ,  day['Total day charge'].values])
night_tensor=torch.tensor([night['Total night minutes'].values, night['Total night calls'].values , night['Total night charge'].values])
evening_tensor=torch.tensor([evening['Total eve minutes'].values, evening['Total eve calls'].values , evening['Total eve charge'].values])
intl_tensor=torch.tensor([intl['Total intl minutes'].values, intl['Total intl calls'].values ,  intl['Total intl charge'].values])
target=trainSet['Churn'].apply(churn_to_num)
print(target)


churners=trainSet[trainSet['Churn']==True]
noneChurners=trainSet[trainSet['Churn']==False]

churners['Customer service calls'].value_counts(normalize=True)
noneChurners['Customer service calls'].value_counts(normalize=True)

class ChurnClassifierDataset(Dataset):
  def __init__(self,day,evening,night,intl,target):
    self.day=day
    self.evening=evening
    self.night=night
    self.intl=intl
    self.churn=target

  def __len__(self):
    return len(self.churn)

  def __getitem__(self,index):
    return {
        'input_features':torch.tensor([ self.day[0][index], self.day[1][index], self.day[2][index],
                                       self.evening[0][index],self.evening[1][index],self.evening[2][index],
                                       self.night[0][index],self.night[1][index],self.night[2][index],
                                       self.intl[0][index],self.intl[1][index],self.intl[2][index]]),
        'target':self.churn[index]
    }
train_set=ChurnClassifierDataset(day_tensor,evening_tensor,night_tensor,intl_tensor,target)
dataLoader=DataLoader(train_set,shuffle=True)
first=next(iter(dataLoader))
shape=first['input_features'].shape
input=first['input_features']
input_target=first['target']
print(shape)

layer1=nn.Linear(shape[0]*shape[1],24)
fn.leaky_relu(layer1(input.reshape(1,-1).float()))

class ChurnPrediction(nn.Module):
  def __init__(self,shape,nbr_classes):
    super(ChurnPrediction,self).__init__()
    self.layer1=nn.Linear(shape[0]*shape[1],shape[0]*24)
    self.layer2=nn.Linear(shape[0]*24,shape[0]*48)
    self.layer3=nn.Linear(shape[0]*48,shape[0]*shape[1])
    self.layer4=nn.Linear(shape[0]*shape[1],shape[0]*nbr_classes)
    self.softmax=nn.Softmax(dim=1)

  def forward(self,t):
    #print('message in here bro \t', t)
    t=fn.leaky_relu(self.layer1(t.reshape(1,-1).float()))
    #print('message in here too \t', t)
    t=fn.leaky_relu(self.layer2(t.float()))
    t=fn.leaky_relu(self.layer3(t.float()))
    t=fn.leaky_relu(self.layer4(t.float()))
    return self.softmax(t.reshape(shape[0],2))

classifier=ChurnPrediction(shape,2)
lossfn=nn.NLLLoss()
optimizer= torch.optim.Adam(classifier.parameters(),0.001)

preds=classifier(input)
optimizer.zero_grad()
classifier.layer1.weight.grad

loss=lossfn(preds,input_target)
loss.item()
loss.backward()

optimizer.step()

classifier.layer1.weight.grad

for epoch in range(2):
  correct_pred=0.0
  for sample in dataLoader:
      input=sample['input_features']
      print(input.shape)
      #print(input)
      target=sample['target']
      #print(target)
      preds=classifier(input)
      print(preds)
      correct_pred=get_correct_pred(preds,target)+correct_pred
      print(f'number of correct prediction is { correct_pred} \n\n\n out {len(trainSet)}')
      loss=lossfn(preds,target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  print(f'{epoch} is done \n\n')
  torch.save(classifier.state_dict(),'sample_data/saved_params.pth')
print('done with the training')

tnight=testSet[['Total night minutes','Total night calls','Total night charge']]
tday=testSet[['Total day minutes','Total day calls','Total day charge']]
tevening=testSet[['Total eve minutes','Total eve calls','Total eve charge']]
tintl=testSet[['Total intl minutes','Total intl calls','Total intl charge']]

tday_tensor=torch.tensor([tday['Total day minutes'].values, tday['Total day calls'].values ,  tday['Total day charge'].values])
tnight_tensor=torch.tensor([tnight['Total night minutes'].values, tnight['Total night calls'].values , tnight['Total night charge'].values])
tevening_tensor=torch.tensor([tevening['Total eve minutes'].values, tevening['Total eve calls'].values , tevening['Total eve charge'].values])
tintl_tensor=torch.tensor([tintl['Total intl minutes'].values, tintl['Total intl calls'].values ,  tintl['Total intl charge'].values])
ttarget=testSet['Churn'].apply(churn_to_num)

test_set=ChurnClassifierDataset(tday_tensor,tevening_tensor,tnight_tensor,tintl_tensor,ttarget)

test_set[0]['input_features'].reshape(1,-1).shape


testLoader=DataLoader(test_set,shuffle=True)

correct_pred=0.0
for sample in testLoader:
  input=sample['input_features'].reshape(1,-1)
  print(input)

  print(f'the dimensions the the input {input.shape}\n\n')
  target=sample['target']
  print(input.shape)
  preds=classifier(input)
  correct_pred=get_correct_pred(preds,target)+correct_pred
  print(f'numbre of correct prediction {correct_pred} out of {len(test_set)}')





