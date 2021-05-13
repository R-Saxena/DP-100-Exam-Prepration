from azureml.core import Experiment, Workspace, Dataset, Datastore

from azureml.core import ScriptRunConfig

ws = Workspace.from_config()
ds = Datastore.get(ws, "mydatastore")
df = Dataset.get_by_name(ws, "Loan Application")

experiment = Experiment(ws, "exp-01")

script_config = ScriptRunConfig(source_directory = ".",
                                script = "01-exp-script.py")

new_run = experiment.submit(config = script_config)

