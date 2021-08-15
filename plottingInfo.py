import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.available
plt.style.use('seaborn-notebook')
df=pd.read_csv('sample_data/california_housing_train.csv')

x=df['longitude'].head(50)
y=df['median_house_value'].head(50)

plt.plot(x.values,y.values)
plt.show()


plt.savefig('housingPriceByLongitude.png')
