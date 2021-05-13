from azureml.core import Workspace,Dataset, Datastore

ws = Workspace.from_config()
az_ds = Datastore.get(ws, "mydatastore")
loan_dataset = Dataset.get_by_name(ws, "Loan Application")
az_default_ds = ws.get_default_datastore()

#now converting this dataset into a pandas dataframe
loan_df = loan_dataset.to_pandas_dataframe()

#we can also perform some actions on this dataset and register it as a new 
loan_df = loan_df[["Married", "Gender", "Loan_Status"]]

loan_new_df = Dataset.Tabular.register_pandas_dataframe(
    dataframe = loan_df, 
    target = az_ds,
    name = "Loan Application from Df"
)

#upload local files to storage account using datastore

files_list = ["./data/test1.csv", "./data/test2.csv"]
az_ds.upload_files(files = files_list,
                   target_path = "New/",
                   relative_root = "./data/",
                   overwrite=True)

#uploading the entire folder
az_ds.upload(src_dir = "./data", 
              target_path = "New/",
              overwrite = True)         
              




