import openml as oml 
import pandas as pd  
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score,r2_score,roc_auc_score,accuracy_score 
warnings.simplefilter(action="ignore", category=DeprecationWarning)
import time
import random
import auto_ml
from auto_ml import Predictor

columns = ['did','name','NumberOfInstances', 'NumberOfFeatures','NumberOfClasses']
oml.config.apikey = 'f7b559f93de31b58e136a7a6ca02c3e9'
oml.config.server = 'https://www.openml.org/api/v1/xml' 
data_dict = oml.datasets.list_datasets()
data_list = pd.DataFrame.from_dict(data_dict,orient='index')

data_ids = {
                'classification':[1464],#[1464,40701,1046]
                'regression':[196]#[196,308,537,344]
            }


data = oml.datasets.get_dataset(1464)
X,y,categorical_indicator,attribute_names = data.get_data(target=data.default_target_attribute)
		
df1 = pd.DataFrame(X,columns=attribute_names)
vectorizer = DictVectorizer(sparse=False)

df2=  vectorizer.fit_transform(df1[df1.columns[0:]].to_dict('records'))
df = pd.DataFrame(df2)
#df['target'] = y
le = LabelEncoder()
y = le.fit_transform(y)


column_descriptions = {
	'target':'output'
}

X_train,X_test,y_train,y_test = train_test_split(df,y,train_size=0.75,test_size=0.25,random_state=1)
X_train['target'] = y_train
X_test['target'] = y_test


ml_predictor  = Predictor(type_of_estimator='classifier',column_descriptions=column_descriptions)
ml_predictor.train(X_train)

#test_score = ml_predictor.score(X_test,X_test.target)
#print(test_score)