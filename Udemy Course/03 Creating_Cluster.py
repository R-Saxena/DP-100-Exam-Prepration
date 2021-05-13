from azureml.core import Workspace, Dataset, Datastore
ws = Workspace.from_config()

cluster_name = "My-Cluster-001"

#provisioning the cluster
from azureml.core.compute import AmlCompute

# for configuring the cluster
compute_config = AmlCompute.provisioning_configuration(
    vm_size="STANDARD_D11_V2",
    max_nodes=2
)

cluster = AmlCompute.create(ws, cluster_name, compute_config)

