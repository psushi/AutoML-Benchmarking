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

from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset

df_train,df_test = get_boston_dataset()

column_descriptions = {
    'MEDV': 'output',
    'CHAS': 'categorical'
}
print(df_train)