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



# ds = TabularDatasetFactory.from_delimited_files(path='https://ml.azure.com/fileexplorerAzNB?wsid=/subscriptions/9b72f9e6-56c5-4c16-991b-19c652994860/resourceGroups/aml-quickstarts-162678/providers/Microsoft.MachineLearningServices/workspaces/quick-starts-ws-162678&tid=660b3398-b80e-49d2-bc5b-ac1dc93b5254&activeFilePath=Users/odl_user_162678/wine-classification.csv')

# print(ds.to_pandas_dataframe().head())

run = Run.get_context()

# x_df = ds.to_pandas_dataframe().dropna()

x_df = pd.read_csv("./wine-classification.csv")
x_df.dropna(inplace=True)

x = x_df.iloc[:,1:-1]
y = x_df.pop('y')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/model.joblib')

if __name__ == '__main__':
    main()
