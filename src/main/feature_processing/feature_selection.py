import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


def feature_selection_mutualinfo_regress(train_x, train_y, verbose=False):
    mi = pd.DataFrame(mutual_info_regression(train_x, train_y, random_state=100), index=train_x.columns)

    if verbose:
        print("Mutual Information in regress:")
        print(mi)

    return mi


def feature_selection_mutualinfo_classif(train_x, train_y, verbose=False):
    mi = pd.DataFrame(mutual_info_classif(train_x, train_y, random_state=100), index=train_x.columns)

    if verbose:
        print("Mutual Information in classif:")
        print(mi)

    return mi


def select_features(train_x, train_y, test_x, test_y, type, num_features=20, verbose=False):
    # This function orders feature from Mutual Info, selects top 60 of the
    # feature and the performs RFE on the selected features.

    if type == "classification":

        # Feature Selection through Mutual Information.
        mi = feature_selection_mutualinfo_classif(train_x, train_y, verbose=False)
        mi.reset_index(drop=True, inplace=True)
        selected = mi.nlargest(60, columns=0).index
        train_x = train_x.iloc[:, selected]
        test_y = test_y.iloc[:, selected]

        estimator = LogisticRegression(C=0.05, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=5,
          multi_class='ovr', n_jobs=-1, penalty='l1', random_state=100,
          solver='liblinear', tol=0.1, verbose=0, warm_start=False)

        selector = RFE(estimator, n_features_to_select=num_features, step=1)
        selector.get_params()
        selector.fit(train_x, train_y)
        train_x = train_x.iloc[:, selector.support_]
        test_x = test_x.iloc[:, selector.support_]

    else:
        # If not Classification then do feature selection using Regression.
        # Feature Selection through Mutual Information.
        mi = feature_selection_mutualinfo_regress(train_x, train_y, verbose=False)
        mi.reset_index(drop=True, inplace=True)
        selected = mi.nlargest(60, columns=0).index
        train_x = train_x.iloc[:, selected]
        test_y = test_y.iloc[:, selected]

        estimator = LinearRegression(n_jobs=-1, fit_intercept=True)

        selector = RFE(estimator, n_features_to_select=num_features, step=1)
        selector.get_params()
        selector.fit(train_x, train_y)
        train_x = train_x.iloc[:, selector.support_]
        test_x = test_x.iloc[:, selector.support_]

    return train_x, train_y, test_x, test_y


