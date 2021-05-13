import numpy as np
import pandas as pd

from azureml.core import Workspace

#creating the workspace
ws = Workspace.create(name = "MyWorkSpace",
                      subscription_id= "1a165e41-dbb8-49a2-9b49-3030ea29f192",
                      resource_group= "MyResourceGrp",
                      create_resource_group=True,
                      location = "centralindia",
)

#saving the config file for future references
ws.write_config()

#to see all the Workspaces

ws_list = Workspace.list(subscription_id="1a165e41-dbb8-49a2-9b49-3030ea29f192")

ws_list = list(ws_list)

