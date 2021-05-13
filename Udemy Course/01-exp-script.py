from azureml.core import Experiment, Dataset, Datastore, Workspace

ws = Workspace.from_config()
ds = Datastore.get(ws, "mydatastore")
df = Dataset.get_by_name(ws, "Loan Application")

from azureml.core import Run

new_run = Run.get_context()

df = df.to_pandas_dataframe()

new_df = df[["Married", "Gender", "Education", "Loan_Status"]]
new_df.to_csv("./outputs/loan_data.csv", index = False)

new_run.log("total rows", df.shape[0])

new_run.complete()
