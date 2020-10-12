import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



def cv_report(X_train, Y_train, X_test, Y_test, model, model_name, cv_no):
    accuracies = cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=cv_no)
    print(model_name)
    print(accuracies)
    acc_mean = accuracies.mean()
    acc_std = accuracies.std()
    model.fit(X_train, Y_train)
    Y_test_pred = model.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    print(accte, "\n")
    return [model_name, acc_mean, acc_std, accte]


def cv_gridsearch(X_train, Y_train, estimator, param_grid, cv):
    print(str(estimator))
    gs_model = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv)
    return [str(estimator), gs_model.fit(X_train, Y_train).best_params_]

def cv_report_list(X_train, Y_train, X_test, Y_test, model_list, cv):
    report = pd.DataFrame(columns=['Model', 'Mean Acc. Training', 'Standard Deviation', 'Acc. Test'])
    for i in model_list:
        report.loc[len(report)] = cv_report(X_train, Y_train, X_test, Y_test, i[0], i[1], cv)
    return report


def cv_gridsearch_list(X_train, Y_train, model_param_grid, cv):
    report = pd.DataFrame(columns=["Model", "Best_Params"])
    for i in model_param_grid:
        report.loc[len(report)] = cv_gridsearch(X_train, Y_train, i[0], i[1], cv)
    return report

def get_example_cv_gridsearch_list():
    model_param_grid = [
        [LogisticRegression(), {
            "max_iter":[150, 200, 250]
        }],
        [DecisionTreeClassifier(),{
            "criterion":["entropy","gini"],
            "max_depth": [3, 5, 7, 10, 15, 20],
            "random_state": [0]
        }],
        [RandomForestClassifier(),{
            "max_depth": [3, 4, 5, 6, 7, 8, 12, 20],
            "n_estimators": [10, 50, 100, 150, 200],
            "random_state": [0]
        }]
    ]
    return model_param_grid


def get_example_cv_report_list():
    model_list = [
        [LogisticRegression(max_iter=200), 'Logistic Regression'],
        [GaussianNB(), 'Naive Bayes'],
        [DecisionTreeClassifier(criterion='entropy', random_state=0), 'Decision Tree C5.0']
    ]
    return model_list


get_example_cv_gridsearch_list()
get_example_cv_report_list()