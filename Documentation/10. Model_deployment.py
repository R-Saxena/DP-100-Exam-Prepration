import numpy as np
import pandas as pd

#we can deploy model as a real-time web service to several kinds of compute target like local compute, ACI(Azure Container instance), AKS, an azure function, IOT modules 

#AzureML uses containers as a deployment mechanism, packaging the model and the code to use it as an image that can be deployed to a container in your chosen compute target.

#for the deployment of the model these are the steps:

# 1. Register a trained model

from azureml.core import Model

classi_model = Model.register(workspace = ws, 
                            model_name='classi_model',
                            model_path = 'model.pkl', # localpath
                            description = "classification model")

#if we have a reference to the Run used to train the model

run.register_model(model_name = 'classi_model', 
                   model_path = 'outputs/model.pkl',
                   description = 'A classification Model')

#Now we need inference configuration in which 2 things will be there 1. script which will return the prediction and 2. Environment script in which 1 will run.

# Need to create Entry Script 

import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("classi_model")
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    pred = model.predict(data)
    return pred.tolist()


# Need to create a script for Environment

from azureml.core.conda_dependencies import CondaDependencies

my_env = CondaDependencies()
my_env.add_conda_package('scikit-learn')

env_file = 'service_files/env.yml'

with open(env_file, "w") as f:
    f.write(my_env.serialize_to_string())

print("Saved dependency info in", env_file)


#combining both of the script in inference config

from azureml.core.model import InferenceConfig

class_inference_config = InferenceConfig(runtime='python',
                                         source_directory='service_files',
                                         entry_script = 'score.py',
                                         conda_file='env.yml')


# now inference config is ready now So now we need to configure the compute to which the service will be deployed 
#if we are going for AKS cluster then we need to create the cluster and a compute target before deployment


#creating the AKS cluster(Azure kubernetes service)

from azureml.core.compute import ComputeTarget, AksCompute

cluster_name = 'aks-cluster'
compute_config = AksCompute.provisioning_configuration(location="eastus")
production_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
production_cluster.wait_for_completion(show_output=True)


from azureml.core.webservice import AksWebservice

classifier_deploy_config = AksWebservice.deploy_configuration(cpu_core = 1, memory_gb=1)


#finally deploting the model

from azureml.core.model import Model

model = ws.models['classification_model']
service = Model.deploy(workspace = ws,
                       name = 'classifier-service',
                       models = [model],
                       inference_config=class_inference_config,
                       deployment_config=classifier_deploy_config,
                       deployment_target=production_cluster)

service.wait_for_deployment(show_output=True)











