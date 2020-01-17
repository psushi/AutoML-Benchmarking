import openml as oml 
import pandas as pd  
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score,r2_score,roc_auc_score,accuracy_score 
import mlbox as mlb
import time
import random
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


columns = ['did','name','NumberOfInstances', 'NumberOfFeatures','NumberOfClasses']
oml.config.apikey = 'f7b559f93de31b58e136a7a6ca02c3e9'
oml.config.server = 'https://www.openml.org/api/v1/xml' 
data_dict = oml.datasets.list_datasets()
data_list = pd.DataFrame.from_dict(data_dict,orient='index')

data_ids = {
                'classification':[1464],#[1464,40701,1046,1461], 
                'regression':[196]#[196,308,537,344]
            }

s=","
r=mlb.preprocessing.Reader(s)
data = oml.datasets.get_dataset(1464)
X,y,categorical_indicator,attribute_names = data.get_data(target=data.default_target_attribute)
#X['target'] = y
X_train, X_test, y_train, y_test =train_test_split(X, y, random_state=1,train_size=0.75,test_size=0.25)
X_train['target'] = y_train
#X_test['target'] = y_test
X_train.to_csv('mlb_data/train_'+str(1416)+'.csv')
X_test.to_csv('mlb_data/test_'+str(1416)+'.csv')

#with warnings.catch_warnings():
	#warnings.simplefilter("ignore", category=FutureWarning)
data = r.train_test_split(['mlb_data/train_'+str(1416)+'.csv','mlb_data/test_'+str(1416)+'.csv'],target_name='target')

best = mlb.preprocessing.Drift_thresholder().fit_transform(data)


space = {'ne__numerical_strategy':{'search':'choice','space':['mean','median']},
'ne__categorical_strategy':{'search':'choice','space':[np.NaN]},
'ce__strategy':{'search':'choice','space':['label_encoding','entity_embedding','random_projection']},
'fs__strategy':{'search':'choice','space':['l1','variance','rf_feature_importance']},
'fs__threshold':{'search':'uniform','space':[0.01,0.3]},
'est__max_depth':{'search':'choice','space':[3,5,7,9]},
'est__n_estimators':{'search':'choice','space':[250,500,700,1000]}}

opt = mlb.optimisation.Optimiser(scoring='f1',n_folds=5)
best = opt.optimise(space,data,40)


#best = mlb.optimisation.Optimiser().evaluate(None,data)
mlb.prediction.Predictor().fit_predict(best,data)





