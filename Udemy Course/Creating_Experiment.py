from azureml.core import Workspace, Dataset, Datastore

ws = Workspace.from_config()
ds = Datastore.get(ws, "mydatastore")
df = Dataset.get_by_name(ws, "Loan Application")


#creating the Experiment
from azureml.core import Experiment

experiment = Experiment(workspace=ws,
name = "Loan-SDK-Exp01")

#Starting the run in the experiment
new_run = experiment.start_logging()

# <------------------------------------------we write the code and log the values to the workspace--------------------------------------->
df = df.to_pandas_dataframe()

total_rows = df.shape[0]

null_df = df.isnull().sum()

#logging the metrices
new_run.log("Total Observations", total_rows)

for column in df.columns:
    new_run.log(column, null_df[column])

#ending the run
new_run.complete()


