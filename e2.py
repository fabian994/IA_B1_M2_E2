import pandas as pd
import numpy as np
import math
from sklearn.linear_model import SGDRegressor

cols = ['Sex','length','Diameter','Height','Whole_weight','Shucked_weight',
        'Viscera_weight','Shell_weight','Rings']
df = pd.read_csv('abalone/abalone.data',names=cols)
df = pd.get_dummies(df,columns=['Sex'],dtype='int')


newdf = df.copy()
newdf = newdf.sample(frac=1)
newdf_x = newdf[['Sex_F','Sex_I','Sex_M','length','Diameter','Height','Whole_weight','Shucked_weight',
        'Viscera_weight','Shell_weight']]
newdf_y = newdf[['Rings']]

dflen = len(newdf)
trainS = math.floor(80*dflen/100)
testS = math.floor(20*dflen/100)
train_x = newdf_x[:][:trainS]
train_y = newdf_y[:][:trainS]

test_x = newdf_x[:][-testS:]
test_y = newdf_y[:][-testS:]


model = SGDRegressor(fit_intercept=True)
