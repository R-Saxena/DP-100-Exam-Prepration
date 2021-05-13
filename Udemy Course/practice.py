# creating a workspace

from azureml.core import Workspace
ws = Workspace.create(name= "My_Workspace",
                      subscription_id="free_Trial",
                      resource_group="My_grp",
                      create_resource_group=True,
                      location="centralindia"
                      )


# we can also create workspace via azure cli 
az ml workspace create -w 'my_workspace' -g 'my_resources'


#Now we have created the workspace, lets access it 

#for accessing the workspace we have to create a config.json file in which we will include all the things related to workspace then we can easily connect the workspace.

#creating the JSON and save it within same directory

{
    "subscription_id" : " ",
    "resource_group": " ",
    "workspace_name": " "
} 

#connecting the workspace 

from azureml.core import Workspace
ws = Workspace.from_config()              # it fetches from the default path 

#Now we have accessed our workspace then lets try to see all the compute target on that workspace which are present.

for compute in ws.compute_targets:
    print(compute)


#Now lets define an experiment. 

# Experiment is nothing by the process used for running a script. Experiment consists of many runs and we can track all the runs. 

#lets create the Experiment

from azureml.core import Experiment

experiment = Experiment(ws, "my_exp")

new_run = experiment.start_logging()

#my code.

new_run.complete()

#Now you know we can also log all the metrics and can save the entire folder or any file.

import numpy as np
import pandas as pd
import json
from azureml.core import Workspace
ws = Workspace.from_config()

exp = Experiment(ws, "my_exp")
run = exp.start_logging()

df = pd.read_csv("data/input_data.csv")

rows = df.shape[0]
#logging the metrics
run.log("Observations", rows)

df = df.iloc[:3]

#data in output folder saving
df.to_csv("./Output/data.csv", index = False)

#reviewing the metrics logged
metrics = run.get_metrics()

print(json.dumps(metrics, indent = 2))

run.complete()

# now We can also use some other flexible way to to do the same thing and that is ScriptRunConfig. 

#for this method we have to create 2 files 
#1. for run script file in which all the logic will be written 
#2. for cretaing the experiment and creating the env and compute cluster and all


#1. process flow file

from azureml.core import Workspace, Experiment, ScriptRunConfig
ws = Workspace.from_config()

exp = Experiment(ws, "my_exp")

script_config = ScriptRunConfig(source_directory = ".",
                                script = "script_name.py",
                                )

run = exp.submit(script_config)

run.wait_for_completion(show_output=True)


#2. Script file

from azureml.core import Run, Workspace

ws = Workspace.from_config()

#to get the current run of the experiment 
new_run = Run.get_context()

#do your work 

#for completing the run
new_run.complete()


#now we have the experiment module part but lets check the how can we do training of the model with this script method

# 1. Script file

from azureml.core import Workspace,Run
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression

new_run = Run.get_context()

df = pd.read_csv("data/input_data.csv")

#we can do preprocessing, feature engineering and all

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=81)

reg = 0.1

classi_model = LogisticRegression(C = 1/reg, solver="liblinear").fit(X_train, Y_train)

y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log("accuracy", acc)

os.makedirs('outputs', exist_ok = True)

joblib.dump(value = classi_model, filename = 'outputs/model.pkl')

run.complete()


# 2 process file

#Now scikit package is not present in the default env so we need to provide the conda dependencies explicitly

from azureml.core import Experiment, ScriptRunConfig, Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()

my_env = Environment(ws, "My_env")

my_env.python.conda_dependencies = CondaDependencies.create(conda_packages= ['scikit-learn', 'pip'],
                                                            pip_packages=['azureml-defaults'])


script_config = ScriptRunConfig(source_directory = ".",
                                script = "script_name",
                                environment = my_env)


experiment = Experiment(ws, "my_training_exp")

new_run = experiment.submit(script_config)

new_run.wait_for_completion()


# you know some times we need to call some functions with arguments 


#1. changes in process file

script_file = ScriptRunConfig(source_directory = ".",
                              script = "script_name.py",
                              arguments = ['--reg-rate', 0.1],
                              environment = my_env)

        
#2. chnages in script file 

from argparse import ArgumentParser as AP
ap = AP()
ap.add_argument("--reg-rate", type = float, dest = 'reg_rate', default=0.01)
args = ap.parse_args()
reg = args.reg_rate


#Now whatever model we are saving if we want to use them in other experiments then what will we do, we can register the model 

from azureml.core import Model

model = Model.register(workspace=ws,
                       model_name = "my_classification_model",
                       model_path = 'model.pkl',
                       description="A..",
                       tags = {'data-format':'csv'},
                       model_framework=Model.Framework.SCIKITLEARN,
                       model_framework_version='0.20.3')



#list of all the register model
from azureml.core import Model

for model in Model.list(ws):
    print(model.name, 'version: ', model.version)


#Now We have done Lots of things So lets go with management of data

#in real time, we have data at so many different places like it can databrick, it can be azure SQL database
#and we want to get data from there So, we use datastores for that. it is like reference space in the storage account.


#adding Datasource in the workspace

from azureml.core import Workspace, Datastore

ds = Datastore.register_azure_blob_container(
    workspace = ws,
    datastore_name = 'blob_data',
    container_name = "container in storage",
    account_name= "Storage_account_name",
    account_key="key"
     )


#reviewing all the datastores
from ds in ws.datastores:
    print(ds)


#get any particular
ds = Datastore.get(ws, "My data Store")


#to get or set default datastore
default_ds = ws.get_default_datastore()
ws.set_default_datastore("My data store")


from azureml.core import Dataset
#dataset can be of two types tabular - structured data, File - unstructured dataset

# Tabular Dataset
blob_ds = ws.get_default_datastore()

tab_ds = Dataset.Tabular.from_delimited_files("csv_paths")

tab_ds = tab_ds.register(ws, "csv_table")

# File Dataset

file_ds = Dataset.File.from_files(path = (blob_ds, 'data/files/images/*.jpg'))
file_ds = file_ds.register(workspace=ws, name = "img_files")


#how to get dataset

df = ws.datastores['dataset_name']

df = Dataset.get_by_name(ws, "dataset_name")

#we can also pass data as the argument like we have passed the parameters and you know we can also give them name with the help or as_named_input() func.



#now we r done with the datasets, datastores then let's check about the cluster

# for creating the dependencies of environment we can also use conda.yml file 

#example of conda.yml file
name: py_env
dependencies:
    scikit-learn
    numpy
    pandas
    pip:
      - azureml-defaults

#we can easily create env from it as well

from azureml.core import Environment

env = Environment.from_conda_specification(name = "Training_env",
                                           file_path= './conda.yml')

#from existing conda env
env = Environment.from_conda_specification(name = "yoyo", conda_environment = 'py_env')

#lets talk about compute target

# compute targets are physical or virtual computers on which experiments are run.
# Type
# 1. Local Compute - used for develpemnt and testing
# 2. Compute cluster - used for high scalability, 
# 3. Attached Cluster - we can connect azure databrick compute as well
# 4. inferencing cluster - used at the deployment time also known as azure kubernetes cluster

#creation of compute cluster

from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute

ws = Workspace.from_config()

compute_name = 'My_cluster'

aml_config = AmlCompute.provisioning_configuration(vm_size="Standard_D11_V2",
                                                    max_nodes=2, min_nodes=0)

cluster = ComputeTarget.create(ws, compute_name, aml_config)

cluster.wait_for_completion()


#checking for existing compute target

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

compute_name = "my_cluster"

try:
    cl = ComputeTarget(ws, compute_name)
    print("Found existing")

except ComputeTargetException:
    config = AmlCompute.provisioning_configuration(vm_size="standard_d11_v2",max_nodes=2)
    cl = ComputeTarget.create(ws, compute_name, config)

cl.wait_for_completion(show_output=True)


#in Azureml we have flexibility to use different compute cluster on different compute-targets

from azureml.core import Environment, ComputeTarget, ScriptRunConfig

compute_name = "My_Cluster"

training_env = Environment(ws, name = "my_env")

compute = ComputeTarget(ws, name = compute_name)

script_config = ScriptRunConfig(source_directory = 'my_dir',
                                script = 'script.py',
                                environment = training_env,
                                compute_target = compute_name/compute) #any would work


#upto this point we know:-
# creating workspace and manage it
# creating experiments and logging the metrics 
# creating the datasets and data stores
# creating the custmized environment, defining the dependencies
# creating the compute targets and using them.

#Now the main thing is coming i.e Pipelining.

# pipelining is a workflow of machine learning tasks in which each task is implented as step
# different steps can use different compute target and datastores to store their output.
#we can publish a pipeline as a REST endpoint, enabling client applications to intiate the pipeline.

#pipeline can be started :- 
# -from client side
# -at periodic level
# -at some trigger point like added new data 

#in pipelining there n+1 py files. 1 is for process flow and n is number of steps and 1 file for each steps

#in pipeline, steps are like one step output is input of next step. So for that we need some reference data stores which can be refered by the steps.


#process flow file

from azureml.core import Experiment, Workspace

ws = Workspace.from_config()

from azureml.pipeline.steps import PythonScriptStep

from azureml.pipeline.core import PipelineData

df = Dataset(ws, "my_dataset")
ds = ws.get_default_datastore()

prepped_data = PipelineData("prepped", ds)

step1 = PythonScriptStep(name = "prepare_data",
                         source_directory="Scripts",
                         script_name = "prepare.py",
                         compute_target="my_cluster1",
                         arguments=['--raw_data', df.as_named_input('raw_data'),
                                    '--out_folder', prepped_data]
                         outputs=[prepped_data])


step2 = PythonScriptStep(name = "training_data",
                         source_directory="Scripts",
                         script_name = "training.py",
                         compute_target="my_cluster2",
                         arguments = ['--in_folder', prepped_data]
                         input = [prepped_data]
                         )


from azureml.pipeline.core import Pipeline

pipeline = Pipeline(ws , steps = [step1, step2])

exp = Experiment(ws, "pappu")

pipeline_run = exp.submit(pipeline)


#Script python file

from azure.core import Run

from argparse import ArgumentParser as AP

parser = AP()
parser.add_argument('--raw_data', type = str, dest = 'raw_data_id')
parser.add_argument('--out_folder', type = str, dest = 'folder')
args = parser.parse_args()

df_id = args.raw_data_id
folder = args.folder

raw_df = run.input_datasets['raw_data'].to_pandas_dataframe()


prepped_df = raw_df.iloc[:4]

os.makedirs(folder, exist_ok=True)

path = os.path.join(folder + 'data.csv')

prepped_df.to_csv(path, index = False)


#pipeline is very time consuming task So what azure does, they try to cache the results of pipeline steps.
# But in some scenario we don't want that So, for that we can use  
#scenarios like if we have changed something in data then it will not get reflected in the pipeline steps

allow_reuse = False in PythonScriptStep

# we can easily publish the pipeline 

published_pipe = pipeline_run.publish(name = "pip",
                     description = "dedo",
                     version = '1.0')


#we can easily get the pipeline endpoint

endpoint = published_pipe.endpoint


#we can also schedule the pipelines for some periodic interval

from azureml.pipeline.core import Schedule, ScheduleRecurrence

#for daily
daily = ScheduleRecurrence(frequency = 'day', interval=1)

pipeline_schedule = Schedule.create(ws, name = "Daily training",
                                    desciption = "",
                                    pipeline_id= published_pipe.id,
                                    experiment_name = 'training_pipe',
                                    recurrence=daily)

#for triggering

from azureml.core import Datastore
from azureml.pipeline.core import Schdule

trining_datastore = Datastore(ws, "blob_data")
pipeline_schedule = Schedule.create(ws, name, description=
                                    pipeline_id = published_pipeline_id,
                                    experiment_name = "training_pipeline",
                                    datastore=trining_datastore,
                                    path_on_datastore="data/training")


#Now we have developed all the things now its time for the development 

#steps:
- Register the trained model
- after that we need to define the inference configuration for this we also need to create a entryscript file 
- create the env
- deplyment code


#registering the model

from azureml.core import Model

classi_model = Model.register(ws, model_name = , model_path = , description=)


#creating the entry script

import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model = joblib.load(Model.get_model_path('classi_model'))

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    return list(model.predict(data))


from azureml.core.model import InferenceConfig

classifier_inference_config = InferenceConfig(runtine = "python",
                                              source_directory=".",
                                              entry_script = 'entry.py',
                                              conda_file='env.yml')


from azureml.core.compute import AksCompute, ComputeTarget

cluster_name = 'aks_cluster'
compute_config = AksCompute.provisioning_configuration(location = 'centralindia')
production_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
production_cluster.wait_for_deployment(show_output=True)


from azureml.core.webservice import AksWebservice

classifier_deploy_config = AksWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)


#final deployment

from azureml.core.model import Model

model = ws.models['classi_model']

service = Model.deploy(ws,
                       name = "",
                       models = [model],
                       inference_config=classifier_inference_config,
                       deployment_config=classifier_deploy_config,
                       deployment_target=production_cluster)

service.wait_for_deployment()


#there may be possible that troubleshooting is there So for that we can deploy service in local docker instance


from azureml.core.webservice import LocalWebservice

deployment_config = LocalWebservice.deploy_configuration(port = 8484)
service = Model.deploy(ws, 'test-svc', [model], inference_config, deployment_config)


service.run(input_data = json_data)



#deploy batch inference piplines

#in real time we will not predict of a single row, we predict in terms of batchs.

# scoring script

import os
import numpy as np
from azureml.core import Model
import joblib

def init():
    # Runs when the pipeline step is initialized
    global model

    # load the model
    model_path = Model.get_model_path('classification_model')
    model = joblib.load(model_path)

def run(mini_batch):
    # This runs for each batch
    resultList = []

    # process each file in the batch
    for f in mini_batch:
        # Read comma-delimited data into an array
        data = np.genfromtxt(f, delimiter=',')
        # Reshape into a 2-dimensional array for model input
        prediction = model.predict(data.reshape(1, -1))
        # Append prediction to results
        resultList.append("{}: {}".format(os.path.basename(f), prediction[0]))
    return resultList


#parallel Run step 


from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import Pipeline

# Get the batch dataset for input
batch_data_set = ws.datasets['batch-data']

# Set the output location
default_ds = ws.get_default_datastore()
output_dir = PipelineData(name='inferences',
                          datastore=default_ds,
                          output_path_on_compute='results')

# Define the parallel run step step configuration
parallel_run_config = ParallelRunConfig(
    source_directory='batch_scripts',
    entry_script="batch_scoring_script.py",
    mini_batch_size="5",
    error_threshold=10,
    output_action="append_row",
    environment=batch_env,
    compute_target=aml_cluster,
    node_count=4)

# Create the parallel run step
parallelrun_step = ParallelRunStep(
    name = 'batch-score',
    parallel_run_config = parallel_run_config,
    inputs = [batch_data_set.as_named_input('batch_data')],
    output = output_dir,
    arguments = [],
    allow_reuse = True
)

# Create the pipeline
pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])


# same deployment like above pipeline

#parameter tuning

# for this we have to define the search space

#discrete hyperparams

# -qnormal
# -quniform
# -lognormal
# -loguniform


#continuous hyperparams

#normal
#uniform
#lognormal
#loguniform


from azureml.train.hyperdrive import choice, normal

param_sapce = {
    '--batch_size': choice(10,12,15),
    '--learning_rate': normal(10,3)
}


#configuring sample - grid,random, baysian

from azureml.train.hyperdrive import GridParameterSampling, RandomParameterSampling, BayesianParameterSampling


param_sampling = GridParameterSampling(param_sapce)


#configuring the early stopping 

- bandit policy            -> BanditPolicy(slack_amount = 0.2, evaluation_interval = 1, delay_evaluation = 5)
- median stopping policy   -> MedianStoppingPolicy(evaluation_interval = 1, delay_evaluation = 5)
- truncation policy        -> TruncationSelectionPolicy(truncation_percentage=10,
                                                     evaluation_interval=1,
                                                     delay_evaluation=5)



#for hyperparameter tuning we need a trianing script

# Now we already have training script above 

#Configuring

from azureml.core import Experiment
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal


hyper_drive = HyperDriveConfig(run_config=script_config,
                                hyperparameter_sampling = param_sampling,
                                policy=None,
                                primary_metric_goal = PrimaryMetricGoal.MAXIMIZE,
                                primary_metric_name = 'Accuracy',
                                max_total_runs = 6,
                                max_concurrent_runs=4
                                )


#we can also monitor and review the hyperdrive runs

for child_run in run.get_children():
    print(child_run.id, child_run.get_metrics())



#we have done alot but you know there is simple way as well for doing it like we can use automl

#data scientist have an ethical resposibilities to protect sensitive data.

#for that what can we do

# - by adding statistical noise to the analysis process so that we cna say about inviduals row features and we will add different noise in different analysis process
#- Epsilon factor added to the features 
# 0 to 1, 0 tells full privacy and 1 says added complete noise




#Now how will you explain the model understanding

# Model explainers use statistical techniques to calculate feature importance.
# This enables you to quantify the relative influence each feature in the training dataset has on label prediction. Explainers work by evaluating a test data set of feature cases and the labels the model predicts for them.

# -global feature importance  -> overall which features are very important.
# -local feature importance -> for seeing some rows or a particular row which feature is more importance.

# pip install azureml-interpret

# explainer :- 

# -MimicExplainer -  An explainer that creates a global surrogate model that approximates your trained model and can be used to generate explanations. This explainable model must have the same kind of architecture as your trained model (for example, linear or tree-based).
# -TabuExplainer -  An explainer that acts as a wrapper around various SHAP explainer algorithms, automatically choosing the one that is most appropriate for your model architecture.
# -PFIExplainer -  a Permutation Feature Importance explainer that analyzes feature importance by shuffling feature values and measuring the impact on prediction performance.


# MimicExplainer
from interpret.ext.blackbox import MimicExplainer
from interpret.ext.glassbox import DecisionTreeExplainableModel

mim_explainer = MimicExplainer(model=loan_model,
                             initialization_examples=X_test,
                             explainable_model = DecisionTreeExplainableModel,
                             features=['loan_amount','income','age','marital_status'], 
                             classes=['reject', 'approve'])
                             

# TabularExplainer
from interpret.ext.blackbox import TabularExplainer

tab_explainer = TabularExplainer(model=loan_model,
                             initialization_examples=X_test,
                             features=['loan_amount','income','age','marital_status'],
                             classes=['reject', 'approve'])


# PFIExplainer
from interpret.ext.blackbox import PFIExplainer

pfi_explainer = PFIExplainer(model = loan_model,
                             features=['loan_amount','income','age','marital_status'],
                             classes=['reject', 'approve'])



# MimicExplainer
global_mim_explanation = mim_explainer.explain_global(X_train)
global_mim_feature_importance = global_mim_explanation.get_feature_importance_dict()


# TabularExplainer
global_tab_explanation = tab_explainer.explain_global(X_train)
global_tab_feature_importance = global_tab_explanation.get_feature_importance_dict()


# PFIExplainer
global_pfi_explanation = pfi_explainer.explain_global(X_train, y_train)
global_pfi_feature_importance = global_pfi_explanation.get_feature_importance_dict()


# MimicExplainer
local_mim_explanation = mim_explainer.explain_local(X_test[0:5])
local_mim_features = local_mim_explanation.get_ranked_local_names()
local_mim_importance = local_mim_explanation.get_ranked_local_values()


# TabularExplainer
local_tab_explanation = tab_explainer.explain_local(X_test[0:5])
local_tab_features = local_tab_explanation.get_ranked_local_names()
local_tab_importance = local_tab_explanation.get_ranked_local_values()

#for showing it on client interface

# Import Azure ML run library
from azureml.core.run import Run
from azureml.contrib.interpret.explanation.explanation_client import ExplanationClient
from interpret.ext.blackbox import TabularExplainer
# other imports as required

# Get the experiment run context
run = Run.get_context()

# code to train model goes here

# Get explanation
explainer = TabularExplainer(model, X_train, features=features, classes=labels)
explanation = explainer.explain_global(X_test)

# Get an Explanation Client and upload the explanation
explain_client = ExplanationClient.from_run(run)
explain_client.upload_model_explanation(explanation, comment='Tabular Explanation')

# Complete the run
run.complete()



























