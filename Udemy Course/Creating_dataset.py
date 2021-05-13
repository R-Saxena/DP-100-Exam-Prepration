from azureml.core import Workspace, Datastore, Dataset

ws = Workspace.from_config()

ds = Datastore(ws, "mydatastore")

#creating Dataset

#creating path, we can have multiple data path like this
dataset_path = [(ds, "loan.csv")]
loan_dataset = Dataset.Tabular.from_delimited_files(path = dataset_path)
dataset = loan_dataset.register(workspace=ws, name = "Loan Application")


# see all the datasets
for i in list(ws.datasets.keys()):
    print(i)


# get a dataset
df = Dataset.get_by_name(ws, "Loan Application")



