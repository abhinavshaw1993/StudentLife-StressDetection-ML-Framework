from main.experiments.experiment_base import ExperimentBase
from main.feature_processing.generalized_global_data_loader import GenralizedGlobalDataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from pathlib import Path, PurePosixPath
import pandas as pd
import sys
import warnings
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class GeneralizedGlobal(ExperimentBase):
    """
    This Class is for generalized global experiments.
    This class will generate all the output files if required.
    """

    def run_experiment(self, write=True, verbose=False):

        # initializing some things
        classifiers = []
        param_grid = []
        model_configs = ExperimentBase.get_model_configurations()

        # Getting the model configs and preparing param grid

        try:
            classifiers = self.get_ml_models()
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
        train_x, train_y, test_x, test_y, train_lebel_dist, test_label_dist = exp.get_data(stress_agg=self.stress_agg, verbose=False)
        splitter = exp.get_val_splitter()

        # GridSearch Initialization.
        clf = GridSearchCV(exp_pipeline, param_grid=param_grid, cv=splitter, n_jobs=-1)
        clf.fit(train_x, train_y)
        best_estimator = clf.best_estimator_
        pred_y = best_estimator.predict(test_x)

        # Getting the relevant results
        best_param = clf.best_params_
        best_score = clf.best_score_
        accuracy = accuracy_score(test_y, pred_y)
        f1 = f1_score(test_y, pred_y, average=None)
        result = pd.DataFrame(clf.cv_results_)

        # Generating Base line with the Given Data.
        most_freq_accuracy, most_freq_label = ExperimentBase.generate_baseline(test_y)

        if write:

            s = "Best Params: {} \n\n  Best Accuracy: {} \n\n Best Score: {} \n\n Most Freq BaseLine: {} Most Freq Label: {}".format(
                best_param,
                accuracy,
                best_score,
                most_freq_accuracy,
                most_freq_label)

            self.write_output(result, s)

        # Printing results
        if verbose:
            print("best params", best_param)
            print("best score", best_score)
            print("accuracy: ", accuracy)
            print("f1: ", f1)
            print("Most Freq Baseline:{}, Most Freq Label: {} ".format(most_freq_accuracy, most_freq_label))
            print("Train Label Distribution:\n {} \n Test Label Distribution: \n{}".format(train_lebel_dist, test_label_dist))

    def write_output(self, result_df, string_to_write=None):

        # Write the whole data frame in a CSV.
        root = os.path.dirname(sys.modules['__main__'].__file__)
        output_path = root + "/outputs/generalizedExperiment"
        grid_path = output_path + "/GeneralizedExperimentResultGrid.csv"
        file_path = output_path + "/GeneralizedExperiment.txt"
        result_df.to_csv(grid_path, index=False, header=True)

        if string_to_write:
            f = open(file_path, "w+")
            f.write(string_to_write)
            f.close()