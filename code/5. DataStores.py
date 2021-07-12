#Datastores are abstraction of cloud data sources.
#they encapsulate the info required to connect to data sources.

#there are different kinds of datastores like Azure Storage(Blob + file manager), Azure data Lake Stores, Azure SQL Database, Azure Databricks file System (DBFS)

#There are 2 built-in datastores (blob+file) that are used as a system storage.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#connecting the workspace
from azureml.core import Workspace, Datastore
ws = Workspace.from_config()

#Registering a new datastores

# for registration we need storage name and storage key in which we want to create this data source
blob_ds = Datastore.register_azure_blob_container(workspace=ws,
                                                  datastore_name = "blob_data",
                                                  container_name='azureml-blobstore-bf4e0c62-87d2-4233-920c-6870aa62cfc0',
                                                  account_name='rishabhmachine5989301776',
                                                  account_key = 'OCToPz0m8zQBNxIUL01aZDyhHDGK3fDuMXCE0NV/e28UW89q9YWfZimujAeGMS4dvGSOEbHE5YYFmZUFRrXaeA==')


#lets check all the datastores in the workspace 

for ds_name in ws.datastores:
    print(ds_name)     

#get a reference to any datastore

blob_store = Datastore.get(ws, datastore_name="blob_data")


#to get the by default datastore
default_store = ws.get_default_datastore()


#to set default datastore
ws.set_default_datastore('blob_data')
