# this code is used for connecting the existing workspace.

#before connecting, You have to create a config.json file in which you need to pass
# subscription_id, resource_grp, workspace_id.

#pip install azureml-sdk

from azureml.core import Workspace

#this function by defaults check the config.json in current directory. we can also give path if it is not here.
ws = Workspace.from_config()

#After connecting to it, we can see different compute targets which we have created in the workspace.

for compute_name in ws.compute_targets:
    cm = ws.compute_targets[compute_name]
    print(cm.name + " : " + cm.type)
    #to get the current the status of each compute
    print(str(cm.get_status()))

