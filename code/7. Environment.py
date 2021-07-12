#Like In the local system, we can also work with the env in the azure. there we can create env, manage them very easily.


import numpy as np
import pandas as pd

#Connecting the Workspace
from azureml.core import Workspace
ws = Workspace.from_config()


# we can create env via several ways 

# 1. via conda file
from azureml.core import Environment
env = Environment.from_conda_specification(name = "training_env", file_path = "./conda.yml")


# 2. via pip requiremnt file
env.from azureml.core import Environment
env = Environment.from_pip_requirements(
    name='env_name',
    file_path='requirements.txt',
)

# 3. via existing conda env
env = Environment.from_existing_conda_environment(name = "training_env", conda_environment_name = 'py_env')


# 4. via specifying packages
from azureml.core.conda_dependencies import CondaDependencies
env = Environment("training_env")
deps = CondaDependencies.create(conda_packages=['scikit-learn', 'pandas', 'numpy'],
                                pip_packages=['azureml-defaults'])

env.python.conda_dependencies = deps


# Registering the environment to the workspace

env.register(workspace = ws)


# get list of all the environments in the workspace
for env_name in Environment.list(workspace = ws):
    print('Name:', env_name)


#for using in the script 

from azureml.core import ScriptRunConfig, Environment

training_env = Environment.get(workspace = ws, name = "training_env")

script_config = ScriptRunConfig(source_directory = "experiments_directory",
                                script = "script.py",
                                environment = training_env)

