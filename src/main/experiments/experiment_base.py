from abc import ABCMeta, abstractmethod
import sys
import os
import yaml
import sklearn.linear_model as linear_model
import sklearn.ensemble as ensemble
import sklearn.svm as svm
import numpy as np


class ExperimentBase:
    __metaclass__ = ABCMeta

    exp_config = {}
    agg_window = str()
    splitter = str()
    transformer = str()
    stress_agg = str()
    ml_models = str()
    previous_stress = True


    def __init__(self):
        """
        Initialize the different properties for the experiment.
        """
        self.read_configs()
        self.set_configs()

    @abstractmethod
    def run_experiment(self, verbose=False):
        """
        To run te experiments.
        """
        pass

    @staticmethod
    def get_model_configurations():
        """
        Returns Dictionary of hyperparameters.
        """
        root = os.path.dirname(sys.modules['__main__'].__file__)
        file_name = root + "/resources/model_configs.yml"
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

        if "LogisticRegression" in self.ml_models:
            model_list.append(linear_model.LogisticRegression())
        if "RandomForestClassifier" in self.ml_models:
            model_list.append(ensemble.RandomForestClassifier())
        if "SVM" in self.ml_models:
            model_list.append(svm.SVC())
        if "AdaBoostClassifier" in self.ml_models:
            model_list.append(ensemble.AdaBoostClassifier())

        if len(model_list) == 0:
            raise Exception("Model Config Not Found!! Check Model config for Experiment.")

        return model_list

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
        self.ml_models = self.exp_config['ml_models']
        self.previous_stress = self.exp_config['previous_stress']
