
import argparse
import joblib
import pandas as pd
import numpy as np
from azureml.core import Workspace,Run
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#setting parameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg', type = float, dest = 'reg_rate', default = 0.01)
args = parser.parse_args()
reg = args.reg_rate

#get Run context
run = Run.get_context()

data = run.input_datasets['training_data'].to_pandas_dataframe()

X = data.iloc[:,:4].values
Y = data.iloc[:,-1].values

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2)

model= LogisticRegresssion(C = 1/reg, solver = 'liblinear').fit(X_train, Y_train)

y_hat = model.predict(X_test)

acc = np.average(y_hat == y_test)
run.log("Accuracy", np.float(acc))

os.makedir('outputs', exist_ok = True)
joblib.dump(model, 'outputs/model.pkl')


run.complete()





