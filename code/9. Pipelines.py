#Pipelines are the workflow of ML tasks in which each task is implemented in steps and can be processed on different compute target and run in a sinle experiment

#we can also publish a pipeline as a rest endpoint, enabling client applications to initiate a pipeline run.

#common types of steps in azure pipeline:

# 1. PythonScriptStep : Runs a specified python script.
# 2. DataTransferStep : Uses Azure Data Factory to copy data between data stores.
# 3. DatabricksStep: Runs a notebook, script or compiles JAR on a databricks cluster.
# 4. AdlaStep: Runs a U-SQL job in Azure Data Lake Analytics.
# 5. ParallelRunStep: Runs a python script as a distributed task on multiple compute nodes.


#we must have to define each step and then we can include all the steps into a pipeline

#example

from azureml.core import Workspace
ws = Workspace.from_config()

from azureml.pipeline.steps import PythonScriptStep

#step to run the python script

step1 = PythonScriptStep(name = "prepare data",
                        source_directory= "scripts",
                        script_name = 'data_prep.py',
                        compute_target='aml-cluster')


step2 = PythonScriptStep(name = "train model",
                        source_directory="scripts",
                        script_name = "train_model.py",
                        compute_target="aml-cluster")


#creating the pipeline

from azureml.core import Experiment,Dataset
from azureml.pipeline.core import Pipeline

train_pipeline = Pipeline(workspace = ws, steps = [step1, step2])

#creating the experiment
experiment = Experiment(workspace = ws, name = 'training-pipeline')
pipeline_run = experiment.submit(train_pipeline)


#Now Passing the data between the Pipelines Steps

# PipelineData Object is used as an intermediary store for data that must 
# be passed from a step to a subsequent step

from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep

# get a dataset for a intial data
raw_ds = Dataset.get_by_name(ws, "new_data")

#Defining a pipeline data object to pass data between the steps

ds = ws.get_default_datastore()
prepped_data = PipelineData('prepped', datastore=ds)

#steps to run a python script 

step1 = PythonScriptStep(name = 'prepare_data',
                        source_directory='scripts',
                        script_name = 'data_prep.py',
                        compute_target= 'aml-cluster',
                        arguments=['--raw-ds', raw_ds.as_named_input('raw_data'),
                                   '--out_folder', prepped_data],
                        
                        outputs=[prepped_data])


steps2 = PythonScriptStep(name = 'train model',
                         source_directory= 'scripts',
                         script_name= 'data_prep.py',
                         compute_target='aml-cluster',
                         arguments=['--in_folder', prepped_data],
                         inputs=[prepped_data])




#to reduce the time we can also manage which step can be reuse by passing some parameter.
#allow_reuse = False in PythonScriptStep.


# We can also use regenerate_outputs = True in experiment.submit() for forcefully disable the reusability of all the steps

#we can also publish via several method.

# 1. directly pipeline publish via pipeline.publish
# 2. can also publish after successful run of the pipeline.

#after publishing we can also provide the end point.

#we can also increase the flexibily of pipelines by passing the parameters

from azureml.pipeline.core import PipelineParameter

reg_param = PipelineParameter(name="reg_rate", default_value = 0.01)


# in Step

step = PythonScriptStep(name = 'train_model',
                        source_directory = 'scripts',
                        script_name = 'data_prep.py',
                        compute_target = 'aml-cluster',
                        arguments = ['--in_folder', prepped_data,
                                    '--reg', reg_param],
                        inputs = [prepped_data]
                        )


#we can run the pipeline on ONdemand like using rest endpoint or we can make a schedule like once in a day it will run with the help of Schedule recurrence library


from azureml.pipeline.core import ScheduleRecurrence, Schedule

daily = ScheduleRecurrence(frequency = 'Day', interval = 1)

pipeline_schedule = Schedule.create(ws, name = 'Daily training',
                                    description='trains model every day',
                                    pipeline_id = published_pipeline.id)
                                    experiment_name = 'Training_pipeline',
                                    recurrence = daily)


# we can also create it like, whenevr data will change pipeline will be run

from azureml.core import Datastore
from azureml.pipeline.core import Schedule

training_ds = Datastore(workspace = ws, name = "blob_data")

pipeline_schedule = Schedule.create(ws, name = 'Reactive training',
                                    description='trains model on data change',
                                    pipeline_id = published_pipeline_id,
                                    experiment_name = 'training_pipeline',
                                    datastore=training_ds,
                                    path_on_datastore='data/training')









