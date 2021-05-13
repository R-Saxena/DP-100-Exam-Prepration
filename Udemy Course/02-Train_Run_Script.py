from azureml.core import Workspace, Experiment, ScriptRunConfig

ws = Workspace.from_config()

experiment = Experiment(ws, "training-exp-02")

from azureml.core import Environment
from azureml.core.environment import CondaDependencies

#creating Custom Environment
my_env = Environment(name = "MyEnvironment2")

#creating the Dependencies
my_env_dep = CondaDependencies.create(conda_packages=['scikit-learn'])

my_env.python.conda_dependencies = my_env_dep

#registering the Environment
my_env.register(ws)

script_config = ScriptRunConfig(source_directory = ".",
                                script = "02-training_script.py",
                                environment = my_env)


new_run = experiment.submit(config = script_config)

new_run.wait_for_completion()
