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
import h2o
from h2o.automl import H2OAutoML
import time
import random
h2o.init()

columns = ['did','name','NumberOfInstances', 'NumberOfFeatures','NumberOfClasses']
oml.config.apikey = 'f7b559f93de31b58e136a7a6ca02c3e9'
oml.config.server = 'https://www.openml.org/api/v1/xml' 
data_dict = oml.datasets.list_datasets()
data_list = pd.DataFrame.from_dict(data_dict,orient='index')

data_ids = {
                'classification': [1464,40701,1046,1461], 
                'regression':[196,308,537,344]
            }



def H2O_benchmarking(data_id,problem_type):
	data = oml.datasets.get_dataset(data_id)
	X,y,categorical_indicator,attribute_names = data.get_data(target=data.default_target_attribute)
	X['target'] = y
	hf = h2o.H2OFrame(X)


	train,test = hf.split_frame(ratios=[0.8],seed=1)
	x = hf.columns
	x.remove('target')
	y='target'

	for seed in random.sample(range(0,100),2):

		aml = H2OAutoML(max_runtime_secs=120,seed=1)
		aml.train(x=x,y=y,training_frame=train)
		lb = aml.leaderboard

		f1_scores=[]
		time_to_predict=[]

		start_time = time.time()
		pred = aml.leader.predict(test)
		time_to_predict.append(time.time() - start_time)
		
		pred = pred.as_data_frame()
		test1 = test.as_data_frame()

		if problem_type =='classification':
			predr = np.round(pred)
			f1_scores.append(f1_score(predr,test1['target']))

		if problem_type == 'regression':
			r2_scores.append(r2_score(pred,test1['target']))

	if problem_type=='classification':
		return np.mean(f1_scores),np.mean(time_to_predict)
	else:
		return np.mean(r2_scores),np.mean(time_to_predict)



def bench_scoring(data_ids):
	clf_scores_dict ={}
	reg_scores_dict = {}
	prediction_time ={}

	for ID in data_ids['classification']:
	    scores,time_to_predict = H2O_benchmarking(ID,'classification')
	    print('score for {} is :{}'.format(ID,scores))
	    clf_scores_dict[ID] = scores
	    prediction_time[ID] = time_to_predict

	#for ID in data_ids['regression']:
	 #   score,time_to_predict = H2O_benchmarking(ID,'regression')
	  #  print('score for {} is :{}'.format(ID,score))
	   # reg_scores_dict[ID] = score
	    #prediction_time[ID] = time_to_predict

	return clf_scores_dict,reg_scores_dict,prediction_time

clf_dict,reg_dict,time_dict = bench_scoring(data_ids)
print('classification:{} , regression:{}'.format(clf_dict,reg_dict))
reg = pd.Series(reg_dict,index=reg_dict.keys())
clf = pd.Series(clf_dict,index=clf_dict.keys())
timings = pd.Series(time_dict,index=time_dict.keys())

reg.to_csv('H2O_benchmarking/reg_h2o.csv')
clf.to_csv('H2O_benchmarking/clf_h2o.csv')
timings.to_csv('H2O_benchmarking/time_to_predict_h2o.csv')



