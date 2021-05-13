from azureml.core import Workspace, Experiment, ScriptRunConfig, Dataset
ws = Workspace.from_config()


#creating the Env
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

my_env = Environment("MyEnvironment2")

conda_dep = CondaDependencies.create(conda_packages=['scikit-learn'])
my_env.python.conda_dependencies = conda_dep

my_env.register()


#creating the compute cluster
from azureml.core.compute import AmlCompute

compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D11_V2", max_nodes=2)

cluster = AmlCompute.create(ws, "My_cluster_01", compute_config)

cluster.wait_for_completion()


#defining the pipeline Steps
from azureml.pipeline.steps import PythonScriptStep

# for input dataset
input_ds = Dataset(ws, "Loan Application")

#for datafolder reference
from azureml.pipeline.core import PipelineData
data_folder = PipelineData("data_folder", ws.get_default_datastore())


#for run configuration
from azureml.core.runconfig import RunConfiguration
run_config = RunConfiguration()
run_config.target = cluster
run_config.environment = my_env

data_prep_step = PythonScriptStep(name = "Data Preprocessing",
                                  source_directory= ".",
                                  script_name="04 data_prep.py", 
                                  inputs=[input_ds.as_named_input("raw_data")],
                                  outputs=[data_folder],
                                  runconfig=run_config,
                                  arguments=["--datafolder", datafolder])


training_step = PythonScriptStep(name = "Training Model",
                                  source_directory= ".",
                                  script_name="04 training.py", 
                                  inputs=[data_folder],
                                  runconfig=run_config,
                                  arguments=["--datafolder", datafolder])


#configuration of entire pipeline

from azureml.pipeline.core import Pipeline

steps = [data_prep_step, training_step]
new_pipeline = Pipeline(ws, steps = steps)

from azureml.core import Experiment

exp = Experiment(ws, "pipeline_exp-01")

new_run = exp.submit(new_pipeline)

new_run.wait_for_completion(show_output=True)


