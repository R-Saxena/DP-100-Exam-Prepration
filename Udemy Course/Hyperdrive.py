from azureml.core import Workspace
ws = Workspace.from_config()

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

#created an env
my_env = Environment("My_new_env")
conda_dep = CondaDependencies.create(conda_packages=['scikit-learn'])
my_env.python.conda_dependencies = conda_dep

my_env.register()

#creating the cluster
from azureml.core.compute import AmlCompute

cluster_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D11_V2"),
                                                       max_nodes = 2)

cluster = AmlCompute.create(ws, "My_cluster", cluster_config)

cluster.wait_for_completion()

#fetching the data
input_ds = ws.datasets.get("Loan Application")

#for ScriptRunning

from azureml.core import ScriptRunConfig, Experiment

script_run = ScriptRunConfig(source_directory = ".",
                             script = "hyperdrive_script.py",
                             arguments = ["--input_data", input_ds.as_named_input("raw_data")],
                             environment = my_env,
                             compute_taret = cluster
                            ) 

#creating hyper parmas

from azureml.train.hyperdrive import GridParameterSampling, choice

hyper_params = GridParameterSampling({
    '--n_estimators': choice(10,20,30,100),
    '--min_samples_leaf': choice(1,2,5)
})


#configuring hyperdrive class
from azureml.train.hyperdrive import HyperDriveConfig,PrimaryMetricGoal

hyper_config = HyperDriveConfig(run_config=script_run,
hyperparameter_sampling = hyper_params,
policy= None,
primary_metric_name = 'accuray',
primary_metric_goal = PrimaryMetricGoal.MAXIMIZE,
max_total_runs = 20,
max_concurrent_runs=2)


exp = Experiment(ws, "My_hyperdrive_exp")

new_run = exp.submit(script_run)

new_run.wait_for_completion(show_output = True)
