import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from azureml.core import Workspace

#creating object of workspace
ws = Workspace.from_config()

#importing the standard libraries for creating experiment and logging.
from azureml.core import Experiment

#creating the Experiment for my workspace 
exp = Experiment(workspace = ws, name= "My_first_Experiment")

#In an experiment we use run context for logging all the things in the experiment.

#start logging the data 
run = exp.start_logging()

data = pd.read_csv("data/diabetes.csv")
row_count = (len(data))

print(row_count)

#logging the row_count value
run.log("number", row_count)

#Complete the experiment
run.complete()


#some times we get some snapshot memory error. it is all because of memory limitation. we can set it to 2000mb instead of 300 and don't run the code in which virtual env is present because it copies the complete current directory.


#now lets check the log of the metrics with runDetails package 

# it also uses 3rd party website permission.
from azureml.widgets import RunDetails
RunDetails(run).show()

#second method to see log metrics

import json

# Get logged metrics
metrics = run.get_metrics()
print(json.dumps(metrics, indent=2))


#we can also upload local output file to the experiment output 
run.upload_file(name='outputs/sample.csv', path_or_stream='./sample.csv')



# Get logged output files
f = run.get_file_names()
print(json.dumps(f, indent=2))


#generally we don't write code of scripting like this

# we make some scripts then we do that
