# AutoML-Benchmarking
Benchmarking popular open source Automatic Machine Learning Frameworks. Datasets from OpenML API.  

### OpenML [https://www.openml.org/]
OpenML is an online machine learning platform for sharing and organizing data, machine learning algorithms and experiments. It is designed to create a frictionless, networked ecosystem, that you can readily integrate into your existing processes/code/environments, allowing people all over the world to collaborate and build directly on each other’s latest ideas, data and results, irrespective of the tools and infrastructure they happen to use.

## Frameworks:
### H2O AutoML [https://www.h2o.ai/products/h2o/] 
H2O is a fully open-source, distributed in-memory machine learning platform with linear scalability. H2O supports the most widely used statistical & machine learning algorithms, including gradient boosted machines, generalized linear models, deep learning, and many more. H2OAutoML is H2O's automatic machine learning framework. 

### Auto-Sklearn [https://automl.github.io/auto-sklearn/master/]
Auto-sklearn provides out-of-the-box supervised machine learning. Built around the scikit-learn machine learning library, auto-sklearn automatically searches for the right learning algorithm for a new machine learning dataset and optimizes its hyperparameters. Thus, it frees the machine learning practitioner from these tedious tasks and allows them to focus on the real problem.

### TPOT [https://epistasislab.github.io/tpot/]
TPOT (tree-based pipeline optimisation tool) is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.TPOT will automate the most tedious part of machine learning by intelligently exploring thousands of possible pipelines to find the best one for your data.

Once TPOT is finished searching (or you get tired of waiting), it provides you with the Python code for the best pipeline it found so you can tinker with the pipeline from there.TPOT is built on top of scikit-learn, so all of the code it generates should look familiar if you're familiar with scikit-learn.

## Benchmarking process:
- Time to train 2 mins for each framework.
- F1 score for classification and R2 score for regression tasks
- 8 datasets used for recording results. (4-regression,4-classification)
- Default parameters wherever applicable.
- Average over two random seeds.
- Can be run on any number of datasets from OpenML, when provided with the OpenML data_id
