#A01367585 | Fabian Gonzalez Vera
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt 
import seaborn as sns

from mlxtend.evaluate import bias_variance_decomp
import warnings
warnings.filterwarnings('ignore') # Ignore pairplot warings

# ETL
print('------Started ETL process------')
cols = ['Sex','length','Diameter','Height','Whole_weight','Shucked_weight',
        'Viscera_weight','Shell_weight','Rings']
df = pd.read_csv('abalone/abalone.data',names=cols)
df = pd.get_dummies(df,columns=['Sex'],dtype='int') #Transform Sex into onehot encoding

newdf = df.copy()
newdf = newdf.sample(frac=1) #Shuffle the dataframe
newdf_x = newdf[['Sex_I','length','Diameter','Height','Whole_weight','Shell_weight']] #Select the chosen variables
newdf_y = newdf[['Rings']] #Target

train_x, test_x, train_y, test_y =  train_test_split(newdf_x, newdf_y, test_size=0.30)#split into 70 train 30 test
print('------Finished ETL process------')


# MODEL
print('--------Started Random Forest Model Tuning--------')
params = { # Random Forest Parameters
    'n_estimators': [50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
}

grid_search = GridSearchCV(RandomForestRegressor(), # GridSearchCV performs an exhaustive search over specified parameter values for an estimator.
                           param_grid= params,
                           scoring= 'neg_root_mean_squared_error', # What is going to be used to evaluate Cross validation
                           refit=True,
                           cv= RepeatedKFold(n_splits=5, n_repeats=3),#Repeated K-Fold cross validator.
                           n_jobs=-1 #Use all available processors
                           )
grid_search.fit(train_x, train_y)
print('--------Finished Random Forest Model Tuning--------\n\n')

print('\n Best parameters for the tree: ',grid_search.best_params_)
print('\n Model with the best parameters: ',grid_search.best_estimator_) # Trained Model with best parameters is stored in best_estimator_
print('\n Mean cross-validated score of the Model with the best parameters: ',grid_search.best_score_)


randForestReg = grid_search.best_estimator_ 
y_preds = randForestReg.predict(test_x)

print("Root Mean Squared Error", mean_squared_error(test_y, y_preds, squared=False))
print('R2 score',randForestReg.score(test_x,test_y))

print('\n\n--------Calculating Bias & Variance--------\n\n')
# estimate bias and variance
mse, bias, var = bias_variance_decomp(randForestReg, train_x.values, train_y.values,
                                       test_x.values, test_y.values, loss='mse', num_rounds=200, random_seed=1)
# summarize results
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)

plt.scatter(test_x['Whole_weight'], test_y, color = 'green')
plt.scatter(test_x['Whole_weight'], y_preds, color = 'red')
plt.title('Random Forest Regression')
plt.xlabel('Whole_weight')
plt.ylabel('Rings')
plt.show() 
