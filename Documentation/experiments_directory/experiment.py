import numpy as np 
import pandas as pd
# import matplotlib.pyplot as plt
import os

from azureml.core import Run

run = Run.get_context()
data = pd.read_csv("diabetes.csv")

#logging in the experiment
row_count = data.shape[0]
run.log("number", row_count)

#storing the sample of data as the output
os.makedirs('outputs', exist_ok=True)
data.sample(100).to_csv("outputs/sample.csv", index=False, header=True)

#complete the run
run.complete()
