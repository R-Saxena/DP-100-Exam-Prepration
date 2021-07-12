import numpy as np
import pandas as pd

from azureml.core import Workspace

ws = Workspace.from_config()

#creating the Datastores
from azureml.core import Datastore

az_store = Datastore.register_azure_blob_container(workspace = ws,
                                                   datastore_name = "MyDataStore",
                                                   account_name = "storageaccountsdk001",
                                                   container_name = "datasets",
                                                   account_key = "o5yLP1G5rtrqofqtJcYre0AiG9cfRtxuLPm8AS5eMDLyLj3VEGgjQPqvXCCkuHx8oFB2YENkoK73XXwME9wHrg=="
)

# for checking the list of all the Datastores 
for i in ws.datastores:
    print(i)

#for fetching a particular datastore
ds = Datastore.get(workspace = ws, datastore_name =  "mydatastore")


#to get the default datastore
ds = ws.get_default_datastore()


#to set any default datastore
ws.set_default_datastore("mydatastore")

