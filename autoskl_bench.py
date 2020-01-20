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
import autosklearn.classification
import autosklearn.regression
import time
import random


columns = ['did','name','NumberOfInstances', 'NumberOfFeatures','NumberOfClasses']
oml.config.apikey = 'f7b559f93de31b58e136a7a6ca02c3e9'
oml.config.server = 'https://www.openml.org/api/v1/xml' 
data_dict = oml.datasets.list_datasets()
data_list = pd.DataFrame.from_dict(data_dict,orient='index')

data_ids = {
                'classification':[1464,40701,1046,1461], #1461 showing error while evaluating together,until then training it separately. 
                'regression':[196,308,537,344]
            }




def autoskl_benchmarking(data_id,problem_type):

    data = oml.datasets.get_dataset(data_id)

    if problem_type == 'classification':
        
        X,y,categorical_indicator,attribute_names = data.get_data(target=data.default_target_attribute)
        df1 = pd.DataFrame(X,columns=attribute_names)
       
        feat_type = ['Categorical' if ci else 'Numerical' for ci in categorical_indicator]
       
        f1_scores = []
        time_to_predict=[]
        for feature in X.columns:
            if str(X.dtypes[feature])=='category':
                le = LabelEncoder()
                X[feature] = le.fit_transform(X[feature])
        


        for seed in random.sample(range(1,100),2):

            X_train, X_test, y_train, y_test =train_test_split(X, y, random_state=1)

            automl = autosklearn.classification.AutoSklearnClassifier(
                                        time_left_for_this_task=60, # sec., how long should this seed fit process run
                                        per_run_time_limit=15, # sec., each model may only take this long before it's killed
                                        ml_memory_limit=1024, # MB, memory limit imposed on each call to a ML algorithm
                                       
                                        )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',category=RuntimeWarning)
                automl.fit(X_train, y_train,feat_type=feat_type,metric=autosklearn.metrics.f1)
                start_time = time.time()
                y_hat = automl.predict(X_test)
                time_to_predict.append(time.time() - start_time)
                y_hat1 = [str(int(i)) for i in y_hat]
                y_test = np.array(y_test)
                le = LabelEncoder()
                y_hat = le.fit_transform(y_hat1)
                y_test= le.transform(y_test)
                f1_scores.append(f1_score(y_hat,y_test))
                

       
        return np.mean(f1_scores),np.mean(time_to_predict)

    if problem_type == 'regression':
        X,y,categorical_indicator,attribute_names = data.get_data(target=data.default_target_attribute)
        
        feat_type = ['Categorical' if ci else 'Numerical' for ci in categorical_indicator]
        for feature in X.columns:
            if str(X.dtypes[feature])=='category':
                le = LabelEncoder()
                X[feature] = le.fit_transform(X[feature])

        
        r2_scores = []
        time_to_predict=[]

        for seed in random.sample(range(1,100),2):

            X_train, X_test, y_train, y_test =train_test_split(X, y, random_state=1)

            automl = autosklearn.regression.AutoSklearnRegressor(
                                        time_left_for_this_task=120, # sec., how long should this seed fit process run
                                        per_run_time_limit=30, # sec., each model may only take this long before it's killed
                                        ml_memory_limit=1024, # MB, memory limit imposed on each call to a ML algorithm
                                        
                                        )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                automl.fit(X_train, y_train,dataset_name=str(data_id),feat_type=feat_type,metric=autosklearn.metrics.r2)
                start_time = time.time()
                y_hat = automl.predict(X_test)
                time_to_predict.append(time.time() - start_time)
                r2_scores.append(r2_score(y_hat,y_test))
                return np.mean(r2_scores),np.mean(time_to_predict)



        
        

def bench_scoring(data_ids):
    clf_scores_dict ={}
    reg_scores_dict = {}
    prediction_time={}

    for ID in data_ids['classification']:
        score,time_to_predict = autoskl_benchmarking(ID,'classification')
        print('score for {} is :{}'.format(ID,score))
        clf_scores_dict[ID] = score
        prediction_time[ID] = time_to_predict

    for ID in data_ids['regression']:
        score = autoskl_benchmarking(ID,'regression')
        print('score for {} is :{}'.format(ID,score))
        reg_scores_dict[ID] = score
        prediction_time[ID] = time_to_predict

    return clf_scores_dict,reg_scores_dict,prediction_time

    
    
    
    
    
    



clf_dict,reg_dict,time_dict = bench_scoring(data_ids)
reg = pd.Series(reg_dict,index=reg_dict.keys())
clf = pd.Series(clf_dict,index=clf_dict.keys())
timings = pd.Series(time_dict,index=time_dict.keys())

reg.to_csv('autoskl_benchmarking1/reg_autoskl.csv')
clf.to_csv('autoskl_benchmarking1/clf_autoskl.csv')
timings.to_csv('autoskl_benchmarking1/time_to_predict_autoskl.csv')



#score = autoskl_benchmarking(1464,'classification')
#print('score is:{}'.format(score))

clf_dict,reg_dict,time_dict = bench_scoring(data_ids)
reg = pd.Series(reg_dict,index=reg_dict.keys())
clf = pd.Series(clf_dict,index=clf_dict.keys())
timings = pd.Series(time_dict,index=time_dict.keys())

reg.to_csv('autoskl_benchmarking/reg_autoskl.csv')
clf.to_csv('autoskl_benchmarking/clf_autoskl.csv')
timings.to_csv('autoskl_benchmarking/time_to_predict_autoskl.csv')

