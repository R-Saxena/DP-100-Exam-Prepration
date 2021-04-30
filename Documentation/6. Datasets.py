#there are two types of dataset 
# 1. tabular dataset in which we store the structured data
# 2. Files dataset in which we work with unstructured data like images.

import numpy as np 
import pandas as pd

#connecting to Workspace
from azureml.core import Workspace, Dataset, Datastore
ws= Workspace.from_config()

#we can create dataset via Andriod studio or Azure SDK.


#For Creating the Tabular dataset (from_delimited_files)
blob_ds = Datastore.get(ws, datastore_name="blob_data")
csv_path = (blob_ds, 'diabetes-data/*.csv')
tab_ds = Dataset.Tabular.from_delimited_files(path = csv_path)
tab_ds = tab_ds.register(workspace=ws, name = 'csv_table')


#for Creating the file dataset (from_files)
ds = ws.get_default_datastore()
file_ds = Dataset.File.from_files(path = (ds, 'path'))
file_ds = file_ds.register(ws, name = "file_dataset")



#retrieving the dataset
data = ws.datastores['file_dataset']
#or
data = Dataset.get_by_name(ws, "csv_table")



#for having same name and different version. when we register the dataset with same name, just need to add one more argument 
tab_ds.register(workspace=ws, name = "csv_table", create_new_version = True)


#retrieving different dataset with version and name

data = Dataset.get_by_name(ws, name = 'csv_table', version=2)



#to convert data into pandas dataframe
df = data.to_pandas_dataframe()
print(df.head())



#we can also use this in script in that we can pass the dataset id as the parameter and we can retrieve data in the script file.

#file can be retrieve in the script file via to ways to_download and to_mount way. 
#In to_download we sent the copy of data to the compute So if you have large dataset then we will not be able to use this for that we go with to_mount becuase in this streaming happens.


