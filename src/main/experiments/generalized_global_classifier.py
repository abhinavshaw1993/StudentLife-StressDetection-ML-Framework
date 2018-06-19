from main.experiments.experiment_base import ExperimentBase
from main.feature_processing.generalized_global_data_loader import GenralizedGlobalDataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import os
import sys
import yaml
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class GeneralizedGlobal(ExperimentBase):
    """
    This Class is for generalized global experiments.
    This class will generate all the output files if required.
    """

    def run_experiment(self):
        exp = GenralizedGlobalDataLoader(agg_window=self.agg_window, splitter=self.splitter,
                                         transformer_type=self.transformer)
        train_x, train_y, test_x, test_y = exp.get_data(stress_agg=self.stress_agg, verbose=False)
        splitter = exp.get_val_splitter()

        # A dict for every model and its parameters.
        # param_grid = {
        #
        #     'classifier__C': [0.1, 0.2, 0.25, 0.5]
        # }
        print("train_x shape: ", train_x.shape)
        param_grid = {

            'classifier__n_estimators': [5, 10, 20, 30, 50],
            'classifier__max_features': [5, 10, 15, 20 , 30]
        }
        steps = [

            ('feature_selector', SelectKBest(k=50)),
            ('classifier', RandomForestClassifier())

        ]

        exp_pipeline = Pipeline(steps)

        # searching over params.
        clf = GridSearchCV(exp_pipeline, cv=splitter, param_grid=param_grid, n_jobs=-1, verbose=True)
        clf.fit(train_x, train_y)
        best_estimator = clf.best_estimator_
        pred_y = best_estimator.predict(test_x)


        # Printing results
        print("best params", clf.best_params_)
        print("best score", clf.best_score_)
        print("accuracy: ", accuracy_score(test_y, pred_y))
        print("f1: ", f1_score(test_y, pred_y, average=None))
        # print(pd.DataFrame(clf.cv_results_))

    def read_configs(self):
        root = os.path.dirname(sys.modules['__main__'].__file__)
        file_name = root + "/resources/generalized_global_classifier.yml"
        # Reading from YML file.
        with open(file_name, "r") as ymlfile:
            self.exp_config = yaml.load(ymlfile)

    def set_configs(self):
        self.agg_window = self.exp_config['agg_window']
        self.splitter = self.exp_config['splitter']
        self.transformer = self.exp_config['transformer_type']
        self.stress_agg = self.exp_config['stress_agg']
        self.ml_model = self.exp_config['ml_model']
