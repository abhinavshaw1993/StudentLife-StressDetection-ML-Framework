from abc import ABCMeta, abstractmethod
from definition import ROOT_DIR
import os
import sys


class DataLoaderBase:
    __metaclass__ = ABCMeta

    def __init__(self, agg_window='d', splitter='predefined', transformer_type='minmax'):
        """
        Initialize the different properties for the data loader.
        :parameter ml_models: a dictionary of different models to be tried out for the pipeline.
        :parameter transformer: string, for which transformer to be used eg: minmax, normalizer etc. Default - standard.
        :parameter train_test_split: Splitting technique, leave k, standart train test split our etc.
        """
        self.aggregation_window = agg_window
        self.splitter = splitter
        self.transformer_type = transformer_type

    @staticmethod
    def get_file_list(agg_window):
        cwd = ROOT_DIR + "/data/aggregated_data"
        student_list = os.listdir(cwd)
        file_list = []

        if agg_window == 'd':
            file_list = [cwd + '/' + student + '/one_day_aggregate.csv' for student in student_list]

        return file_list

    @abstractmethod
    def get_data(self):
        """
        To run te experiments.
        """
        pass

    @abstractmethod
    def get_val_splitter(self):
        """
        TO get an iterable for splits of the data.
        """
        pass

    @staticmethod
    def adjust_stress_values(stress_level):
        mapping = {
            1: 2,
            2: 3,
            3: 4,
            4: 1,
            5: 0
        }
        return mapping[stress_level]

    @staticmethod
    def segregate_y_labels(stress_level):
        mapping = {
            0: 0,
            1: 1,
            2: 1,
            3: 1,
            4: 2
        }

        return mapping[stress_level]
