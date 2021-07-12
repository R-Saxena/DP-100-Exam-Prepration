from azureml.train.automl import AutoMLConfig

autu_run_config = RunConfiguration(framework = 'python')

automl_config = AutoMLConfig(name = "Automated ML Experiment",
                             task = 'classification',
                             primary_metric = 'AUC_weighted',
                             compute_target = aml_compute,
                             training_data = train_dataset,
                             validation_data = test_dataset,
                             label_column_name = 'Label',
                             featurization = 'auto',
                             iterations = 12,
                             max_concurrent_iterations = 4)





