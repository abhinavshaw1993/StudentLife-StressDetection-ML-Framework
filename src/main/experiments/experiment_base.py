from abc import ABCMeta, abstractmethod


class ExperimentBase:
    __metaclass__ = ABCMeta

    exp_config = {}
    agg_window = str()
    splitter = str()
    transformer = str()
    stress_agg = str()
    ml_model = str()

    def __init__(self):
        """
        Initialize the different properties for the experiment.
        :parameter ml_models: a dictionary of different models to be tried out for the pipeline.
        :parameter transformer: string, for which transformer to be used eg: minmax, normalizer etc. Default - standard.
        :parameter train_test_split: Splitting technique, leave k, standart train test split our etc.
        """
        self.read_configs()
        self.set_configs()


    @abstractmethod
    def run_experiment(self):
        """
        To run te experiments.
        """
        pass

    @abstractmethod
    def read_configs(self):
        """
        To read the configs of the experiment.
        """
        pass

    @abstractmethod
    def set_configs(self):
        """
        To Set the configs of the experiment.
        """
        pass