from azureml.core import Run

new_run = Run.get_context()

ws = new_run.experiment.workspace


import pandas as pd

# Read the input dataset
df = new_run.input_datasets['raw_data'].to_pandas_dataframe()

# Select relevant columns from the dataset
dataPrep = df.drop(["ID"], axis=1)

all_cols = dataPrep.columns

# Check the missing values
dataNull = dataPrep.isnull().sum()


# Replace the missing values of string variable with mode
mode = dataPrep.mode().iloc[0]
cols = dataPrep.select_dtypes(include='object').columns

dataPrep[cols] = dataPrep[cols].fillna(mode)


# Replace numerical columns with mean
mean = dataPrep.mean()
dataPrep = dataPrep.fillna(mean)


# Create Dummy variables - Not required in designer/Classic Studio
dataPrep = pd.get_dummies(dataPrep, drop_first=True)


# Normalise the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
columns = df.select_dtypes(include='number').columns
dataPrep[columns] = scaler.fit_transform(dataPrep[columns])

#data folder access

from argparse import ArgumentParser as AP
parser = AP()
parser.add_argument("--datafolder", type = str)
args = parser.parse_args()


#create the folder if exist
import os
os.makedirs(args.data, exist_ok=True)

path = os.path.join(args.datafolder, "prepared_data.csv")

dataPrep.to_csv(path, index = False)

# Log null values
for column in all_cols:
    new_run.log(column, dataNull[column])

new_run.complete()

