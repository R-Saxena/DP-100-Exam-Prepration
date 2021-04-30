from azureml.core import Experiment, ScriptRunConfig, Workspace, Environment
from azureml.core.conda_dependencies import CondaDependencies

#connecting to workspace 
ws = Workspace.from_config()

sklearn_env = Environment('sklearn-env')

#Ensures the required packages are installed
packages = CondaDependencies.create(conda_packages=['scikit-learn', 'pip'], 
                                    pip_packages=['azureml-defaults'])

sklearn_env.python.conda_dependencies = packages

#creating a config file
script = ScriptRunConfig(source_directory = "experiments_directory",
                         script = "training_experiment.py",
                         arguments = ['--reg-rate', 0.1],
                         environment = sklearn_env)

#submit the experiment 
exp = Experiment(workspace = ws, name = "Training_model_Experiment")
run = exp.submit(config=script)
run.wait_for_completion(show_output=True)