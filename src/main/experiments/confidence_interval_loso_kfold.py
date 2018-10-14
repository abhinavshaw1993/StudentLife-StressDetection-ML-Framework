from main.experiments.experiment_base import ExperimentBase
from main.feature_processing.generalized_global_data_loader import GeneralizedGlobalDataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix as cm
from definition import ROOT_DIR
import pandas as pd
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class ConfidenceIntervalsResult(ExperimentBase):
    """
    This Class is for generalized global experiments.
    This class will generate all the output files if required.
    """

    def run_experiment(self, train=True, write=True, verbose=False):
        results = pd.DataFrame()
        exp_count = 10

        feature_tuple = (
            ["sum", "sum", "kurtosis", "mcr", "var", "sum", "previous_stress_level", "poly_b", "poly_c", "poly_b",
             "sum",
             "median", "linear_m", "sum", "median", "mean", "mcr", "sum", "linear_m", "median"],
            ["hours_slept", "activity_inference", "activity_inference", "audio_activity_inference", "conv_duration_min",
             "dark_duration_min", "Unnamed: 103_level_1", "phonelock_duration_min", "conv_duration_min",
             "dark_duration_min", "phonelock_duration_min", "phonecharge_duration_min", "dark_duration_min",
             "conv_duration_min", "dark_duration_min", "phonelock_duration_min", "activity_inference", "call_recorded",
             "phonecharge_duration_min", "phonelock_duration_min"])

        print("#########################################################################################")

        print(
            "Previous Stress Levels:{} Feature Selection:{} Transformer:{} Splitter(s):{}".format(self.previous_stress,
                                                                                                  self.feature_selection,
                                                                                                  self.transformer,
                                                                                                  self.splitter))

        print("########################################################################################")
        print("################################### Truncate Sq : {} ###################################".format(
            self.exp_config["truncate_sq"]))

        for i in range(exp_count):

            for segragate_by_median in [True]:

                for splitter in self.splitter:
                    # Initiaziling Metrics lists/

                    accuracy = []
                    micro_f1 = []
                    macro_f1 = []

                    # recall
                    micro_recall = []
                    macro_recall = []

                    # Precision.
                    micro_precision = []
                    macro_precision = []

                    confusion_matrix = []

                    most_freq_accuracy = []
                    most_freq_label = []
                    print("############################## {} split running #############################".format(splitter))

                    exp = GeneralizedGlobalDataLoader(agg_window=self.agg_window, splitter=splitter,
                                                      transformer_type=self.transformer)

                    train_x, train_y, test_x, test_y, train_label_dist, test_label_dist = exp.get_data(
                        stress_agg=self.stress_agg, previous_stress=True,
                        feature_selection=False, verbose=verbose,
                        segragate_by_median=segragate_by_median, truncate_sq=self.exp_config["truncate_sq"])

                    # Selecting Features.
                    train_x, test_x = train_x.loc[:, feature_tuple], test_x.loc[:, feature_tuple]

                    if not train:
                        return

                    for classifier in self.get_ml_models():
                        classifier_name = classifier.__class__.__name__
                        param = self.exp_config[classifier_name]
                        classifier.set_params(**param)

                        print("############################## {} classifier running #############################".format(
                            classifier_name))

                        for train_idx, test_idx in exp.get_val_splitter():
                            # Preparing splits.
                            split_train_x, split_train_y = train_x.iloc[train_idx], train_y.iloc[train_idx]
                            split_test_x, split_test_y = train_x.iloc[test_idx], train_y.iloc[test_idx]

                            # Fitting Classifier
                            classifier.fit(split_train_x, split_train_y)
                            split_pred_y = classifier.predict(split_test_x)

                            ############################### Metrics ###############################
                            accuracy.append(accuracy_score(split_test_y, split_pred_y))

                            # f1 scores.
                            micro_f1.append(f1_score(split_test_y, split_pred_y, average="micro"))
                            macro_f1.append(f1_score(split_test_y, split_pred_y, average="macro"))

                            # recall
                            micro_recall.append(recall_score(split_test_y, split_pred_y, average="micro"))
                            macro_recall.append(recall_score(split_test_y, split_pred_y, average="macro"))

                            # Precision.
                            micro_precision.append(precision_score(split_test_y, split_pred_y, average='micro'))
                            macro_precision.append(precision_score(split_test_y, split_pred_y, average='macro'))
                            confusion_matrix.append(cm(y_true=split_test_y, y_pred=split_pred_y))

                            # Generating Base line with the Given Data.
                            mfa, mfl = ExperimentBase.generate_baseline(test_y)
                            most_freq_accuracy.append(mfa)
                            most_freq_label.append(mfl)

                        metrics = pd.DataFrame({
                            "segragate_by_median": segragate_by_median,
                            "previous_stress": self.previous_stress,
                            "splitter": splitter,
                            "classifier": classifier_name,
                            "avg_accuracy": sum(accuracy) / len(accuracy),
                            "avg_micro_f1": sum(micro_f1) / len(micro_f1),
                            "avg_macro_f1": sum(macro_f1) / len(macro_f1),
                            "avg_micro_recall": sum(micro_recall) / len(micro_recall),
                            "avg_macro_recall": sum(macro_recall) / len(macro_recall),
                            "avg_micro_precision": sum(micro_precision) / len(micro_precision),
                            "avg_macro_precision": sum(macro_precision) / len(macro_precision)},
                            index=[0])

                        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                            print(metrics)

                        results = results.append(metrics, ignore_index=True)

            ################################ Writing to csv ################################
            results.to_csv(path_or_buf=ROOT_DIR + "/outputs/ConfidenceInterval/results.csv")
