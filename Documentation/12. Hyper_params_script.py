from azureml.core import Workspace,Experiment
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal

ws = Workspace.from_config()


hyperdrive = HyperDriveConfig(run_config=script_config,
                              hyperparameter_sampling = param_sampling,
                              policy = None,
                              primary_metric_goal = PrimaryMetricGoal.MAXIMIZE,
                              primary_metric_name = Accuracy,
                              max_total_runs = 6,
                              max_concurrent_runs=4)


experiment = Experiment(ws, name = "hyper_training")
hyperdrive_run = experiment.submit(config = hyperdrive)


# monitoring the childs 

for child_run in run.get_children():
    print(child_run.id, child_run.get_metrics())


for child_run in hyperdrive_run.get_children_sorted_by_primary_metric():
    print(child_run)

best_run = hyperdrive_run.get_best_run_by_primary_metric()

