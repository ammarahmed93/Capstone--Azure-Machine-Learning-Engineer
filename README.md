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
*TODO*: Explain about the data you are using and where you got it from. <br>
The dataset used for this project can be obtained from the [Wine Quality- UCI Repo](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/). The dataset contains two seprate csv files for each the white and red wine qualities. Both dataset were combined and the red wine was assigned a value of 1 and white wine as 0 in the target column "y". The final merged csv file was uploaded to github. The data contains 6,598 rows and 13 columns including the target column. 

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it. <br>
The goal is to predict the wine type (red or white wine) based on the given properties such as pH, alchohol ...etc. Therefore this is a binary classification task. A target of one stands for red winde and 0 for white wine.

### Access
*TODO*: Explain how you are accessing the data in your workspace. <br>
The data was made publicly available through github. To access the data in the workspace, it was registerd in the workspace by providing the full path of the data along with the name and workspace. The code snippet below shows how the data being accessed and registerd in the Azure workspace 
```
found = False
description_text = "Wine Quality DataSet for Udacity Capstone Project"
key = 'wine-classification'

if key in ws.datasets.keys(): 
    found = True
    dataset = ws.datasets[key] 

if not found:
    # Create AML Dataset and register it into Workspace
    whiteWine_data = 'https://github.com/ammarahmed93/Capstone--Azure-Machine-Learning-Engineer/wine-classification.csv'
    dataset = Dataset.Tabular.from_delimited_files(whiteWine_data, separator=',')        
      
    #Register Dataset in Workspace
    dataset = whiteWine_dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)


df = dataset.to_pandas_dataframe()
df.describe()
```
However, for the hyperdrive experiment, the data was manually uploaded and placed within the same directory as the train.py. 

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment <br>
The following snippet was the settings for the automl:
```
# automl settings
automl_settings = {
    "n_cross_validations": 3,
    "experiment_timeout_minutes": 20,
    "enable_early_stopping": True,
    "max_concurrent_iterations": 4,
    "primary_metric" : 'AUC_weighted'
}

#automl config 
automl_config = AutoMLConfig(compute_target=compute_target,
                             path = project_folder,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="y",   
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```


* *n_crossvalidations* : The cross validation was provided to perform k-fold cross-validation. This parameter sets how many cross validations to perfotm, based on the same number of folds. In this case, 3 folds are used for cross-validation. Therefore 3 different trainings are performed with each training using 2/3 of the data and each validation using 1/3 of the data with a different holdout fold each time. The metrics are calculated with the average of the five validation metrics. <br>
* *experiment_timeoutminutes*: In this case the maximum time that each iteration can run for is 20 minutes. if the iteration reaches the time limit then it will terminate. <br>
* *enable_earlystopping*: When enabling early termination, if the score is not improving in the short term after the first 20 iterations then the experiment will terminate. <br>
* *max_concurrentiterations*: The maximum number of iterations that would be execute in parallel, in this case 4 parallel iterations at max. This is dependent on the number of nodes provisioned and has to be one less than the provisioned nodes.<br>
* _primarymetric_: The metric that the AutoML will optimize for the model selection. The "AUC_weighted" was chosen in this case as its better metric for imblanced data.<br>
* *compute_target*: The compute cluster that the automl will be running on <br>
* _path_: The path to the Azure ML project folder. <br>
* _featurization_: performs featuratization automatically such as imputation, encoding categorical variables ...etc. <br>
* _debug_log_: The file to write the debug information. <br>
* _task_: "classification" since its a binary classification task.
<br>
[reference](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cross-validation-data-splits)
<br>

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it? <br>
The best model that was trained was the VotingEnsemble. The model Achieved an accuracy of 0.999 
* **Need a screenshot of the model with the actual metrics**
* **Screenshot of the Data Guardrails**

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
<br>
* **RunWidget**
![img](https://github.com/ammarahmed93/Capstone--Azure-Machine-Learning-Engineer/blob/main/img/Step%202-Automl%20RunWidget.PNG)
<br>

* **Parameters of the Best model trained**
![img](https://github.com/ammarahmed93/Capstone--Azure-Machine-Learning-Engineer/blob/main/img/Step%202%20-automl%20best%20model%20ID.png)

<br>
The results can be improved by enabling the DNN in the settings, however, this would require a higher computations. 

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search <br>

The model used for second approach was the logistic regression from Scikit-learn. The logistic regression model was implemented for simplicity reasons as the main focus of this project is operatinalizing ML in Azure and not building the most sophisticated and accurate model. The logistic regression is suited well for binary classification problem and can be setup and trained easily. The model has two main hyperparamters the C and max iterations. The Azure HyperDrive was used to tune both parameters. For the Hyperdrive to tune the parameters, the parameter sampler had to be defined first. For the logistic regression, the two parameters were defined in the parameter sampler as follows:

```
ps = RandomParameterSampling({
    "--C": uniform(0.03, 1),
    "--max_iter": choice(50, 100, 150, 200, 300)
})
```

<br>

There are three choices for the sampling methods: Random sampling, Grid sampling, and Bayesian sampling.
The grid sampling is the most expensive one as its an exhaustive search over the hyperparameter space. 
Bayesian sampling is based on Bayesian optimization algorithm and similar to Grid sampling, it is recommended if we have enough budget to explore the hyperparamter space.
The Random sampling was chosen as it results in faster hyperparemter tuning and it also supports early termination of low-performance runs. 
However, if the time and budget was not an issue, the Grid sampling would yield to the most optimal hyperparameters. For the search space, it can be discrete or continous. 
In the bove code snippet, the *choice* specific discrete values search and *uniform* specifies continous hyperparameters. 
More information regarding the parameter sample and search space can be found in the [Azure-documentation](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/).  
The *BanditPolicy* method was used to define early stopping based on the slack criteria and evaluation interval.
```
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)
```
* *evaluation_interval*: the frequency for applying the policy.
* *slack_factor*: the ratio used to calculate the allowed distance from the best performing experiment run.

Based on the defined parameters in the code snippet above, the early termination policy is applied at every other interval when metrics are reported. For instance, if the best performing run at interval 2 reported a primary metric is 0.8. If the policy specify a _slack_factor_ of 0.1, any training runs whose best metric at interval 2 is less than 0.73 (0.8/(1+_slack_factor_)) will be terminated.



### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it? <br>

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![img](https://github.com/ammarahmed93/Capstone--Azure-Machine-Learning-Engineer/blob/main/img/Step%202%20hyperdrive%20RunWidget.png?raw=true)

<br>
The best run hyperparameters for this experiment was C=0.884 and max_iter=1000 and the accuracy was 0.975 as shown in the screenshot below:
![diagram](https://github.com/ammarahmed93/Capstone--Azure-Machine-Learning-Engineer/blob/main/img/hyperdrive-%20best%20model%20with%20hyperparams%20jupyter.PNG?raw=true)

![diagram](https://github.com/ammarahmed93/Capstone--Azure-Machine-Learning-Engineer/blob/main/img/hyperdrive-%20best%20model%20with%20hyperparams.PNG?raw=true)

![diagram](https://github.com/ammarahmed93/Capstone--Azure-Machine-Learning-Engineer/blob/main/img/step%202-%20hyperdrive%20visualize%20the%20progress%20of%20runs.PNG)

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
