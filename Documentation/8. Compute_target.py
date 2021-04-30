# Compute targets are physical or virtual computers used to run the experiments.

#Types of compute - local, compute clusters, Attached compute

#local - low cost, used for code development
#cluster - for experiments workload with high scalibility requirements.
#Atached - can attach existing azure compute like databricks for running the workloads
#inference - used for the deployment of trained models

#One of the core benefit of cloud computing is ability to manage costs by running different peice of code via different targets.

#for creating the workspace
from azureml.core import Workspace
ws = Workspace.from_config()

# Creating the compute
from azureml.core.compute import ComputeTarget, AmlCompute

compute_name = "aml-cluster"

compute_config = AmlCompute.provisioning_configuration( vm_priority="dedicated",
                                                        vm_size='STANDARD_DS11_V2',
                                                        min_nodes = 0,
                                                        max_nodes = 4)

aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)

aml_cluster.wait_for_completion(show_output=True)


#An unmanaged compute or attached compute is defined and managed outside of the workspace like Azure VM or Azure Databricks cluster.

#we can also attch with the help of access token in the confirguration


#checking for the Existing Compute Target i not found any then create

from azureml.core.compute_target import ComputeTargetException

compute_name = "aml-compute"

try:
    aml_cluster = ComputeTarget(ws, compute_name)
    print("exists")
except ComputeTargetException:

    compute_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_DS11_V2", max_nodes=4, min_nodes=0)

    aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)

aml_cluster.wait_for_completion(show_output=True)

