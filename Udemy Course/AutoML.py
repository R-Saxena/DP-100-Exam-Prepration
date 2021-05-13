from azureml.core import Workspace

# Access the workspace from the config.json 
print("Accessing the workspace...")
ws = Workspace.from_config(path="./config")

# Get the input data from the workspace
print("Accessing the dataset...")
input_ds = ws.datasets.get('Loan Application')


# Create the compute Cluster 
# --------------------------------------------------------------------
# Specify the cluster name
cluster_name = "My-Cluster-001"

# Provisioning configuration using AmlCompute
from azureml.core.compute import AmlCompute

print("Accessing the compute cluster...")

if cluster_name not in ws.compute_targets:
    print("Creating the compute cluster with name: ", cluster_name)
    compute_config = AmlCompute.provisioning_configuration(
                                     vm_size="STANDARD_D11_V2",
                                     max_nodes=2)

    cluster = AmlCompute.create(ws, cluster_name, compute_config)
    cluster.wait_for_completion()
else:
    cluster = ws.compute_targets[cluster_name]
    print(cluster_name, ", compute cluster found. Using it...")



#automl configuration
from azureml.train.automl import AutoMLConfig

automl_config = AutoMLConfig(task = "classification",
                             compute_target = cluster,
                             training_data = input_ds,
                             validation_size = 0.3,
                             label_column_name = "Loan_Status",
                             primary_metric = "norm_macro_recall",
                             iterations = 10,
                             max_concurrent_iterations = 2,
                             experiment_timeout_hours = 0.25,
                             featurization = 'auto'
                             )

from azureml.core.experiment import Experiment

new_exp = Experiment(ws, "new_automl")

new_run = new_exp.submit(automl_config)

new_run.wait_for_completion(show_output=True)


#for best model
best_child_run = new_run.get_best_child(metric = "accuracy")

