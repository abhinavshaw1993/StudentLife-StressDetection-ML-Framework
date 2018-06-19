from main.experiments.experiment_base import ExperimentBase
from main.feature_processing.generalized_global_data_loader import GenralizedGlobalDataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class GeneralizedGlobal(ExperimentBase):
    """
    This Class is for generalized global experiments.
    This class will generate all the output files if required.
    """

    def run_experiment(self, verbose=False):

        # initializing some things
        classifiers = []
        param_grid = []
        model_configs = ExperimentBase.get_model_configurations()

        # Getting the model configs and preparing param grid

        try:
            classifiers = self.get_ml_models()
            print("working here")
        except Exception as e:
            print("Class GeneralizedAggregates: ", e)

            # Preparing the Param Dictionary, where models will be included.
        for idx, model in enumerate(self.ml_models):
            try:
                model_config = model_configs[model]
            except KeyError:
                model_config = {}

            model_config['classifier'] = [classifiers[idx]]
            param_grid.append(model_config)

        # Initialize Steps and Pipeline.
        steps = [

            ('feature_seletor', SelectKBest(k=50)),
            ('classifier', classifiers[0])

        ]

        exp_pipeline = Pipeline(steps)

        if verbose:
            print(exp_pipeline.get_params().keys())

        # Fetching the required data and splitter.
        exp = GenralizedGlobalDataLoader(agg_window=self.agg_window, splitter=self.splitter,
                                         transformer_type=self.transformer)
        train_x, train_y, test_x, test_y = exp.get_data(stress_agg=self.stress_agg, verbose=verbose)
        splitter = exp.get_val_splitter()

        # GridSearch Initialization.
        clf = GridSearchCV(exp_pipeline, param_grid=param_grid, cv=splitter)
        clf.fit(train_x, train_y)
        best_estimator = clf.best_estimator_
        pred_y = best_estimator.predict(test_x)

        # Printing results
        if verbose:
            print("best params", clf.best_params_)
            print("best score", clf.best_score_)
            print("accuracy: ", accuracy_score(test_y, pred_y))
            print("f1: ", f1_score(test_y, pred_y, average=None))
            print(pd.DataFrame(clf.cv_results_))

