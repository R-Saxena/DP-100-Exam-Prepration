# -----------------------------------------------------------------
# Decision Tree Classifier
# Predict the income of an adult based on the census data
# -----------------------------------------------------------------

# Import libraries
import pandas as pd


# Read dataset
df = pd.read_csv('./data/adultincomedata.csv')


# Create Dummy variables
data_prep = pd.get_dummies(df, drop_first=True)


# Create X and Y Variables
X = data_prep.iloc[:, :-1]
Y = data_prep.iloc[:, -1]


# Split the X and Y dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = 0.3, random_state = 1234, stratify=Y)


# Import and train Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)


trained_model = rfc.fit(X_train, Y_train)


# Test the RFC model
Y_predict = rfc.predict(X_test)

# Evaluate the RFC model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)
score = rfc.score(X_test, Y_test)


#explainability

from interpret.ext.blackbox import TabularExplainer 


classes = ['Not Greater than 50k', "Greater than 50k"]
features = list(X.columns)

tab_explainer = TabularExplainer(trained_model,
                                 X_train, 
                                 features = features,
                                 classes = classes)

#global 
global_explaination = tab_explainer.explain_global(X_train)

global_fi = global_explaination.get_feature_importance_dict()

#local

X_explain = X_test[:5]

local_explaination = tab_explainer.explain_local(X_explain)

local_f = local_explaination.get_ranked_local_names()
local_importance = local_explaination.get_ranked_local_values()


