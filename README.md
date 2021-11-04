# Predicting Wine Type using Azure ML
[![Azure](https://img.shields.io/badge/Azure-MLOps-blue)](https://www.credly.com/badges/2f897b9d-cd69-44af-a94a-511eb732b19c/linked_in)
[![Udacity](https://img.shields.io/badge/Udacity-Operationalizing%20ML-blue)](https://www.udacity.com/course/machine-learning-engineer-for-microsoft-azure-nanodegree--nd00333)

## Table of contents
* [Overview](#Overview)
* [Project Set Up and Installation](#Project-Set-Up-and-Installation)
* [Dataset](#Dataset)
* [Automated ML](#Automated-ML)
* [Hyperparameter Tuning](#Hyperparameter-Tuning)
* [Model Deployment](#Model-Deployment)
* [Screen Recording](#Screen-Recording)


## Overview
In this project, we build a machine learning model in Azure ML using two approaches. The first approach was utlizing AutoML in Azure ML. The second approach was using Python SDK and a Scikit-learn Logistic Regression model and tuning the hyperparamters using the Hyperdrive. The results from both approaches were compared and the best model was deployed as a service using ACI (Azure Container Instance). One of the goals of the project is to use an external datasource and register it in Azure for training the models. 

## Project Set Up and Installation
There are two main files for the two approaches: [```hyperparameter_tuning.ipynb```](https://github.com/ammarahmed93/Capstone--Azure-Machine-Learning-Engineer/blob/main/hyperparameter_tuning.ipynb) to run the SKLearn model and [```automl.ipynb```](https://github.com/ammarahmed93/Capstone--Azure-Machine-Learning-Engineer/blob/main/automl.ipynb) for running the automl. <br>
The following files are necessary for the project setup: 
- [```wine-classification.csv```](https://github.com/ammarahmed93/Capstone--Azure-Machine-Learning-Engineer/blob/main/wine-classification.csv): the dataset required for this project. Although, the automl.ipynb could directly read the data from this repo and regsiter the dataset in the blobstorage associated with the wrokspace, the dataset might be require to be uploaded manually along with the hyperparameter_tuning.ipynb file within the same directory. 
- [```train.py```](https://github.com/ammarahmed93/Capstone--Azure-Machine-Learning-Engineer/blob/main/train.py): used along with the ```hyperparameter_tuning.ipynb``` and must be in the same directory.
- [```env.yml```](https://github.com/ammarahmed93/Capstone--Azure-Machine-Learning-Engineer/blob/main/env.yml): needed for the deployment environment. 
- [```score.py```](https://github.com/ammarahmed93/Capstone--Azure-Machine-Learning-Engineer/blob/main/score.py): is the scoring script which is required to deploy the model. The file has to be placed in a folder "./source_dir" in the same parent directory as the automl.ipynb file.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
