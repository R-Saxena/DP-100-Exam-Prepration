from azureml.core import Workspace, Run

ws = Workspace.from_config()

new_run = Run.get_context()

#doing my stuff
import pandas as pd

df = pd.read_csv("./data/loan.csv")

#selecting only few columns
LoanPrep = df[["Married",
               "Education",
               "Self_Employed",
               "ApplicantIncome",
               "LoanAmount",
               "Loan_Amount_Term",
               "Credit_History",
               "Loan_Status"]]

LoanPrep = LoanPrep.dropna()

LoanPrep = pd.get_dummies(LoanPrep, drop_first=True)

Y = LoanPrep[['Loan_Status_Y']]
X = LoanPrep.drop(["Loan_Status_Y"], axis = 1)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=1234, stratify=Y)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


lr.fit(X_train, Y_train)

Y_predict = lr.predict(X_test)

Y_prob = lr.predict_proba(X_test)[:,1]

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_predict)

score = lr.score(X_test, Y_test)


#logging the metrics

new_run.log("TotalObservations", df.shape[0])
new_run.log("Score", score)

cm_dict = {
       "schema_type": "confusion_matrix",
       "schema_version": "1.0.0",
       "data": {
           "class_labels": ["N", "Y"],
           "matrix": cm.tolist()
       }
   }

new_run.log_confusion_matrix("ConfusionMatrix", cm_dict)

X_test = X_test.reset_index(drop = True)
Y_test = Y_test.reset_index(drop = True)

Y_pred_df = pd.DataFrame(Y_prob, columns = ["Scored Probs"])

Y_predict_df = pd.DataFrame(Y_predict, columns = ["Scored label"])

scored_df = pd.concat([X_test, Y_test, Y_pred_df, Y_predict_df], axis = 1)

scored_df.to_csv("./outputs/loan_scored.csv", index = False)

new_run.complete()