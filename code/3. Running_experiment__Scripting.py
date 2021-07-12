import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from azureml.core import Workspace
ws = Workspace.from_config()

#importing the scripting config library
from azureml.core import Experiment, ScriptRunConfig

#need to give directory name in which the script is present 
script_config = ScriptRunConfig(source_directory = "experiments_directory", script = "experiment.py")

#creating an experiment
exp = Experiment(workspace = ws, name = "script_exp")

#running the script in this experiment
run = exp.submit(config=script_config)

#completion of script
run.wait_for_completion(show_output=True)
