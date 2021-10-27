from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Datastore, Dataset

ws = Workspace.from_config()

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required

# create tabular dataset from a single file in datastore

key = "wine_classification"

if key in ws.datasets.keys(): 
        found = True
        dataset= ws.datasets[key] 

df = dataset.to_pandas_dataframe()

run = Run.get_context()

print(df)
    
# x, y = clean_data(ds)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# def main():
#     # Add arguments to script
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
#     parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

#     args = parser.parse_args()

#     run.log("Regularization Strength:", np.float(args.C))
#     run.log("Max iterations:", np.int(args.max_iter))

#     model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

#     accuracy = model.score(x_test, y_test)
#     run.log("Accuracy", np.float(accuracy))

#     os.makedirs('outputs', exist_ok=True)
#     joblib.dump(value=model, filename='outputs/model.joblib')

if __name__ == '__main__':
    main()