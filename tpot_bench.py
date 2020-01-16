#from mlbox.preprocessing import *
#import mlbox
import openml as oml 
import pandas as pd  
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import warnings
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction import DictVectorizer
warnings.simplefilter(action="ignore", category=DeprecationWarning)
import random



columns = ['did','name','NumberOfInstances', 'NumberOfFeatures','NumberOfClasses']
oml.config.apikey = 'f7b559f93de31b58e136a7a6ca02c3e9'
oml.config.server = 'https://www.openml.org/api/v1/xml' 
data_dict = oml.datasets.list_datasets()
data_list = pd.DataFrame.from_dict(data_dict,orient='index')

#print(data_list.query('name.str.contains("blood-transfusion")',engine='python')[columns])

#blood = oml.datasets.get_dataset(1464)
#speed = oml.datasets.get_dataset(40536)
#dataframes = [blood,speed]
#print(blood.description[:500])
#abalone  = oml.datasets.get_dataset(183)


def tpot_benchmarking(data_id,problem_type):

	data = oml.datasets.get_dataset(data_id)



	if problem_type =='classification':
		X,y,categorical_indicator,attribute_names = data.get_data(target=data.default_target_attribute)
		df1 = pd.DataFrame(X,columns=attribute_names)
		vectorizer = DictVectorizer(sparse=False)

		df2=  vectorizer.fit_transform(df1[df1.columns[0:]].to_dict('records'))
		df = pd.DataFrame(df2)
		df['target'] = y
		le = LabelEncoder()
		df['target'] = le.fit_transform(df['target'])
		training_indices,testing_indices = training_indices,validation_indices = train_test_split(df.index,stratify=y,train_size=0.75,test_size=0.25)
		f1_score = []
		


		for seed in random.sample(range(1,100),2):
			tpot = TPOTClassifier(verbosity=2,max_eval_time_mins=0.04,max_time_mins=1,population_size=15,cv=5,random_state=seed,scoring='f1')
			tpot.fit(df.drop('target',axis=1).loc[training_indices],df.loc[training_indices,'target'])
			f1_score.append(tpot.score(df.drop('target',axis=1).loc[validation_indices].values, df.loc[validation_indices, 'target'].values))


		return np.mean(f1_score)

	if problem_type == 'regression':
		X,y,categorical_indicator,attribute_names = data.get_data(target=data.default_target_attribute)
		df1 = pd.DataFrame(X,columns=attribute_names)
		vectorizer = DictVectorizer(sparse=False)

		df2=  vectorizer.fit_transform(df1[df1.columns[0:]].to_dict('records'))
		df = pd.DataFrame(df2)
		df['target'] = y
		training_indices,testing_indices = training_indices,validation_indices = train_test_split(df.index,train_size=0.75,test_size=0.25)
		r2 = []

		for seed in random.sample(range(1,100),2):
			tpot = TPOTRegressor(verbosity=2,max_eval_time_mins=0.04,max_time_mins=1,population_size=15,cv=5,random_state=seed,scoring='r2')
			tpot.fit(df.drop('target',axis=1).loc[training_indices],df.loc[training_indices,'target'])
			r2.append(tpot.score(df.drop('target',axis=1).loc[validation_indices].values, df.loc[validation_indices, 'target'].values))

		return np.mean(r2)


#for oml_data in dataframes:
	#tpot_score = tpot_benchmarking(oml_data)
	#print('tpot scores : {}'.format(tpot_score))


data_ids = {
				'classification':[1464,40701,1046,1461],
				'regression':[196,308,537,344]
			}




def bench_scoring(data_ids):
	clf_scores_dict ={}
	reg_scores_dict = {}

	for ID in data_ids['classification']:
		score = tpot_benchmarking(ID,'classification')
		print('score for {} is :{}'.format(ID,score))
		clf_scores_dict[ID] = score

	for ID in data_ids['regression']:
		score = tpot_benchmarking(ID,'regression')
		print('score for {} is :{}'.format(ID,score))
		reg_scores_dict[ID] = score


	return clf_scores_dict,reg_scores_dict



clf_dict,reg_dict = bench_scoring(data_ids)

#score  =tpot_benchmarking(287,'regression')


#print(score)


print('classifiers: {}'.format(clf_dict))
print('regression:{}'.format(reg_dict))



