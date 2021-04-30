# IN AzureML, we can acheive hyper parameter tuning through an experiment that consists of a hyperdrive run, which initiates a child run
# for each hyper parameter combination to be tested. Each child run uses a training script with parameterized hyperparams values to train a model 
# and logs the target performance metric achieved by the trained model.

#for hyperparameter tuning we have to define the search space

from azureml.train.hyperdrive import choice, normal

param_space = {

    '--batch_size' : choice(16,32,64),
    '--learning_rate': normal(10,3)
}


from azureml.train.hyperdrive import GridParameterSampling
#all list would be discrete for grid search 
param_space = {
    "--batch_size" : choice(16,32,64),
    '--learning_rate' : choice(0.01, 0.02, 0.03)
}

param_sampling = GridParameterSampling(param_space)

#distributions can be these type

#for discrete:
# qnormal,quniform,qlognormal, qloguniform

# for continuous:

#normal, uniform, lognormal, loguniform



#for different type of sampling we need parameter as per there nature

# 1. GridSearch - parametres range should be discrete values
# 2. RandomSearch - params can be a mix of discrete and continuous values
# 3. Baysian Search - parms can be choice, uniform, quniform and can be combined with early stopping policy


# Early Stopping Policies

# 1. Bandit policy: stop the run if the target metric performance underperforms the best run so far by a specified margin
# for example - if the metric is 0.2 or more worse than the best then

from azureml.train.hyperdrive import BanditPolicy
early_termination_policy = BanditPolicy(slack_amount=0.2,
                                        evaluation_interval=1,
                                        delay_evaluation=5)

# 2. Median Stopping policy: abandons runs where the target performance metric is worse than the median of th erunning averages for all the runs

from azureml.train.hyperdrive import MedianStoppingPolicy
early_termination_policy = MedianStoppingPolicy(evaluation_interval=1,
                                        delay_evaluation=5)


# 3. truncation selection policy: it cancels the lower performing X% of the runs at each evaluatin interval based on the trunc_perc value you specify for X.

