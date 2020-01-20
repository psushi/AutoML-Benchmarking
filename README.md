# AutoML-Benchmarking
Benchmarking popular open source Automatic Machine Learning Frameworks. Datasets from OpenML API.  

## Frameworks:
### H2O AutoML  
H2O is a fully open-source, distributed in-memory machine learning platform with linear scalability. H2O supports the most widely used statistical & machine learning algorithms, including gradient boosted machines, generalized linear models, deep learning, and many more.

### Auto-Sklearn 
Auto-sklearn provides out-of-the-box supervised machine learning. Built around the scikit-learn machine learning library, auto-sklearn automatically searches for the right learning algorithm for a new machine learning dataset and optimizes its hyperparameters. Thus, it frees the machine learning practitioner from these tedious tasks and allows her to focus on the real problem.

### TPOT
TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.TPOT will automate the most tedious part of machine learning by intelligently exploring thousands of possible pipelines to find the best one for your data.

Once TPOT is finished searching (or you get tired of waiting), it provides you with the Python code for the best pipeline it found so you can tinker with the pipeline from there.


TPOT is built on top of scikit-learn, so all of the code it generates should look familiar... if you're familiar with scikit-learn, anyway.

## Benchmarking process:
- Time to train 2 mins for each framework. 
- Default parameters wherever applicable.
- Average over two random seeds.
