import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as fn 
from torch.utils.data import Dataset , DataLoader

def get_correct_prediction(predictions,labels):
    return predictions.argmax(dim=1).eq(labels).sum()


tensor=torch.tensor([234.3422,413.342],dtype=torch.float64)
layer1=nn.Linear(2,4)
real=fn.relu(layer1(tensor.float()))
pred=torch.sigmoid(real)
#fn.binary_cross_entropy(pred,torch.tensor([1.0,1.0,1.0,1.0]))
#print(get_correct_prediction(pred,torch.tensor([1.0,1.0,1.0,1.0])))
pred.argmax().eq(2)


df=pd.read_csv('sample_data/telecomDataset.csv')
sample=df[['InternetService','Churn','TotalCharges']]
empty_str_index=df[df['TotalCharges']==' '].index.values
sample.drop(index=empty_str_index,inplace=True)
sample['index']=torch.arange(0,sample.shape[0],dtype=torch.int32)
sample.set_index('index',inplace=True)


def churn_to_num(churn):
  if churn=='No':
    return torch.tensor(0,dtype=torch.int8)
  else:
    return torch.tensor(1,dtype=torch.int8)


def net_number(service):
  if service=='No':
    return torch.tensor(0,dtype=torch.float16)
  else:
    if service=='DSL':
      return torch.tensor(1,dtype=torch.float16)
    else:
      return torch.tensor(2,dtype=torch.float16)


class ChurnClassifierDataset(Dataset):
  def __init__(self,field1,field2,field3):
    self.netService=field1
    self.totalCharges=field2
    self.churn=field3


  def __len__(self):
    return len(self.churn)

  def __getitem__(self,index):
    return {
        'input_features':torch.tensor([self.netService[index],self.totalCharges[index]]),
        'target':self.churn[index]
    }



field1=sample['InternetService'].apply(net_number)
field2=torch.tensor(sample['TotalCharges'].astype(np.float16),dtype=torch.float16)
field3=sample['Churn'].apply(churn_to_num)


train_set=ChurnClassifierDataset(field1,field2,field3)
train_set[488]
dataLoader=DataLoader(train_set)
first=next(iter(dataLoader))



class ChurnClassier(nn.Module):
  def __init__(self):
    super(ChurnClassier,self).__init__()
    self.layer1=nn.Linear(2,4)
    self.layer2=nn.Linear(4,2)
    self.softmax=nn.Softmax(dim=1)

  def forward(self,t):
    #print('message in here bro \t', t)
    t=fn.relu(self.layer1(t.float()))
    #print('message in here too \t', t)
    t=fn.relu(self.layer2(t.float()))
    return self.softmax(t)

classifier=ChurnClassier()
lossfn=nn.NLLLoss()
optimizer= torch.optim.Adam(classifier.parameters(),0.001)


correct_pred=0.0
for sample in dataLoader:
  input=sample['input_features']
  #print(input)
  target=torch.tensor(sample['target']).type(torch.LongTensor)
  #print(target)
  preds=classifier(input)
  print(preds)
  correct_pred=get_correct_prediction(preds,target)+correct_pred
  print('number of correct prediction is ', correct_pred, '\n\n\n')
  loss=lossfn(preds,target)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
print('done with the training')


print(correct_pred / len(train_set))
