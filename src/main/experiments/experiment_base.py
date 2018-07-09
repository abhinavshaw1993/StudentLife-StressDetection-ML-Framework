from abc import ABCMeta, abstractmethod
from main.definition import ROOT_DIR
import yaml
import sklearn.linear_model as linear_model
import sklearn.ensemble as ensemble
import sklearn.svm as svm
import numpy as np
import time
import datetime


class ExperimentBase:
    __metaclass__ = ABCMeta

    exp_config = {}
    agg_window = str()
    splitter = str()
    transformer = str()
    stress_agg = str()
    ml_models = str()
    previous_stress = True
    feature_selection = True
    loss = []

    def __init__(self, config_file):
        """
        Initialize the different properties for the experiment.
        """

        self.read_configs(config_file)
        self.set_configs()

    @abstractmethod
    def run_experiment(self, train=True, write=True, verbose=False):
        """
        To run te experiments.
        """
        pass

    @staticmethod
    def get_model_configurations():
        """
        Returns Dictionary of hyperparameters.
        """
        file_name = ROOT_DIR + "/resources/model_configs.yml"
        # Reading from YML file.
        with open(file_name, "r") as ymlfile:
            model_configs = yaml.load(ymlfile)

        return model_configs

    @staticmethod
    def generate_baseline(true_y):
        """
        Generate Baseline accuracy using most label etc.
        """

        # Most Freq. Accuracy.
        counts = np.bincount(true_y.astype(int))
        most_freq = np.max(counts)
        most_freq_label = np.argmax(counts)
        most_freq_accuracy = most_freq / true_y.shape[0] * 1.0

        return most_freq_accuracy, most_freq_label

    # Non Abstract Methods

    def get_ml_models(self):
        model_list = []

        # Classifiers
        if "LogisticRegression" in self.ml_models:
            model_list.append(linear_model.LogisticRegression())
        if "RandomForestClassifier" in self.ml_models:
            model_list.append(ensemble.RandomForestClassifier())
        if "SVM" in self.ml_models:
            model_list.append(svm.SVC())
        if "AdaBoostClassifier" in self.ml_models:
            model_list.append(ensemble.AdaBoostClassifier())

        # Regression
        if "LinearRegression" in self.ml_models:
            model_list.append(linear_model.LinearRegression())
        if "RandomForestRegressor" in self.ml_models:
            model_list.append(ensemble.RandomForestRegressor())
        if "SVR" in self.ml_models:
            model_list.append(svm.SVR())

        # Raise Error if no model Selected.
        if len(model_list) == 0:
            raise Exception("Model Config Not Found!! Check Model config for Experiment.")

        return model_list

    def read_configs(self, config_file):

        file_name = ROOT_DIR + "/resources/" + config_file
        # Reading from YML file.
        with open(file_name, "r") as ymlfile:
            self.exp_config = yaml.load(ymlfile)

    def set_configs(self):
        self.agg_window = self.exp_config['agg_window']
        self.splitter = self.exp_config['splitter']
        self.transformer = self.exp_config['transformer_type']
        self.stress_agg = self.exp_config['stress_agg']
        self.ml_models = self.exp_config['ml_models']
        self.previous_stress = self.exp_config['previous_stress']
        self.feature_selection = self.exp_config['feature_selection']
        self.loss = self.exp_config['loss']

    @staticmethod
    def write_output(exp_name, result_df, string_to_write=None):
        # Write the whole data frame in a CSV.

        output_path = ROOT_DIR + "/outputs/" + exp_name
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%m_%d_%H_%M')

        grid_path = output_path + "/" + exp_name + "ResultGrid_" + st + ".csv"
        file_path = output_path + "/" + exp_name + ".txt"
        result_df.to_csv(grid_path, index=False, header=True)

        if string_to_write:
            f = open(file_path, "w+")
            f.write(string_to_write)
            f.close()
