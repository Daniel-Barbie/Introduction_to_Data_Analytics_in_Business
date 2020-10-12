import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV


def cv_report(X_train, Y_train, X_test, Y_test, model, model_name, cv_no):
    report = pd.DataFrame(columns=['Model', 'Mean Acc. Training', 'Standard Deviation', 'Acc. Test'])
    accuracies = cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=cv_no)
    print(model_name)
    print(accuracies)
    acc_mean = accuracies.mean()
    acc_std = accuracies.std()
    model.fit(X_train, Y_train)
    Y_test_pred = model.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    print(accte, "\n")
    report.loc[len(report)] = [model_name, acc_mean, acc_std, accte]
    return report


def cv_gridsearch(model_param_grid, cv_n, X_train, Y_train):
    report = pd.DataFrame(columns=["Model", "Best_Params"])
    for i in model_param_grid:
        print(str(i[0]))
        gs_model = GridSearchCV(estimator=i[0], param_grid=i[1], cv=cv_n)
        report.loc[len(report)] = [str(i[0]), gs_model.fit(X_train, Y_train).best_params_]
    return report
