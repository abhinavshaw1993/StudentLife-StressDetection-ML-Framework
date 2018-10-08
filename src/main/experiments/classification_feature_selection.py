from main.experiments.experiment_base import ExperimentBase
from main.feature_processing.generalized_global_data_loader import GeneralizedGlobalDataLoader
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
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


class GeneralizedFeatureSelection(ExperimentBase):
    """
    This Class is for generalized global experiments.
    This class will generate all the output files if required.
    """

    def run_experiment(self, train=True, write=True, verbose=False):

        # initializing some things
        accuracy = []
        micro_f1 = []
        macro_f1 = []
        weighted_f1 = []

        # recall
        micro_recall = []
        macro_recall = []
        weighted_recall = []

        # Precision.
        micro_precision = []
        macro_precision = []
        weighted_precision = []

        feature_selection_rankings = []
        feature_selection_masks = []
        confusion_matrix = []

        most_freq_accuracy = []
        most_freq_label = []

        # Selecting Experiment Type
        experiment_type = {
            "predefined": "Global Generalized with Val and Test",
            "loso": "Global Generalized with Loso",
            "kfold": "Global Generalized with kfold CV with Randomization"
        }

        min_features_to_be_selected = 20

        ## This is feature selection experiments, model will be hardcoded to one of the random forests.

        print("#########################################################################################")

        print(
            "Previous Stress Levels:{} Feature Selection:{} Transformer:{} Splitter(s):{}".format(self.previous_stress,
                                                                                                  self.feature_selection,
                                                                                                  self.transformer,
                                                                                                  self.splitter))

        print("########################################################################################")

        for splitter in self.splitter:

            print("############################## {} split running #############################".format(splitter))

            exp = GeneralizedGlobalDataLoader(agg_window=self.agg_window, splitter=splitter,
                                              transformer_type=self.transformer)
            train_x, train_y, test_x, test_y, train_label_dist, test_label_dist = exp.get_data(
                stress_agg=self.stress_agg, previous_stress=self.previous_stress,
                feature_selection=self.feature_selection, feature_selection_type='classification', verbose=verbose)

            if not train:
                return

            n_estimators = self.exp_config["RandomForestClassifier"].get("n_estimators", 50)
            max_features = self.exp_config["RandomForestClassifier"].get("max_features", 20)
            criterion = self.exp_config["RandomForestClassifier"].get("criterion", "gini")
            max_depth = self.exp_config["RandomForestClassifier"].get("max_depth", 100)
            random_state = self.exp_config["RandomForestClassifier"].get("random_state", 100)
            min_samples_split = self.exp_config["RandomForestClassifier"].get("min_samples_split", 5)
            class_weight = self.exp_config["RandomForestClassifier"].get("class_weight", "balanced")
            n_jobs = -1

            val_generator = exp.get_val_splitter()

            for train_idx, test_idx in val_generator:

                random_forest_classifier = RandomForestClassifier(

                    n_estimators=n_estimators,
                    max_features=max_features,
                    criterion=criterion,
                    max_depth=max_depth,
                    random_state=random_state,
                    min_samples_split=min_samples_split,
                    class_weight=class_weight,
                    n_jobs=n_jobs

                )

                selector = RFE(estimator=random_forest_classifier, step=1, n_features_to_select=20)
                split_train_x, split_train_y = train_x.iloc[train_idx], train_y.iloc[train_idx]
                split_test_x, split_test_y = train_x.iloc[test_idx], train_y.iloc[test_idx]

                # Fitting RFE.
                selector.fit(split_train_x, split_train_y)
                feature_selection_rankings.append(list(selector.ranking_))
                feature_selection_masks.append(list(selector.support_))

                # training the model with selected features.
                split_train_x, split_test_x = split_train_x.iloc[:, selector.support_], split_test_x.iloc[:,
                                                                                        selector.support_]

                random_forest_classifier.fit(split_train_x, split_train_y)
                split_pred_y = random_forest_classifier.predict(split_test_x)

                ############################### Metrics ###############################
                accuracy.append(accuracy_score(split_test_y, split_pred_y))

                # f1 scores.
                micro_f1.append(f1_score(split_test_y, split_pred_y, average="micro"))
                macro_f1.append(f1_score(split_test_y, split_pred_y, average="macro"))
                weighted_f1.append(f1_score(split_test_y, split_pred_y, average="weighted"))

                # recall
                micro_recall.append(recall_score(split_test_y, split_pred_y, average="micro"))
                macro_recall.append(recall_score(split_test_y, split_pred_y, average="macro"))
                weighted_recall.append(recall_score(split_test_y, split_pred_y, average="weighted"))

                # Precision.
                micro_precision.append(precision_score(split_test_y, split_pred_y, average='micro'))
                macro_precision.append(precision_score(split_test_y, split_pred_y, average='macro'))
                weighted_precision.append(precision_score(split_test_y, split_pred_y, average='weighted'))
                confusion_matrix.append(cm(y_true=split_test_y, y_pred=split_pred_y))

                # Generating Base line with the Given Data.
                mfa, mfl= ExperimentBase.generate_baseline(test_y)
                most_freq_accuracy.append(mfa)
                most_freq_label.append(mfl)

            metrics = pd.DataFrame({"accuracy": accuracy,
                                    "micro_f1": micro_f1,
                                    "macro_f1": macro_f1,
                                    "weighted_f1": weighted_f1,
                                    "micro_recall": micro_recall,
                                    "macro_recall": macro_recall,
                                    "weighted_recall": weighted_recall,
                                    "micro_precision": micro_precision,
                                    "macro_precision": macro_precision,
                                    "weighted_precision": weighted_precision,
                                    "confustion_matrix": confusion_matrix}
                                   )

            metrics.append(metrics.mean(), ignore_index=True)

            feature_selection_rankings_pd = pd.DataFrame(feature_selection_rankings, columns=train_x.columns)
            feature_selection_masks_pd = pd.DataFrame(feature_selection_masks, columns=train_x.columns)
            feature_selection_rankings = feature_selection_rankings.append(feature_selection_rankings.mean())
            feature_selection_masks = feature_selection_masks.append(feature_selection_masks.mean())

            ########################## writing to csv ##########################
            metrics.to_csv(path_or_buf=ROOT_DIR+"/outputs/FeatureSelection/metrics.csv")
            feature_selection_rankings_pd.to_csv(path_or_buf=ROOT_DIR+"/outputs/FeatureSelection/rankings.csv")
            feature_selection_masks_pd.to_csv(path_or_buf=ROOT_DIR + "/outputs/FeatureSelection/masks.csv")

            print("AVG accuracy: ", sum(accuracy)/len(accuracy))
            print("AVG macro F1: ", sum(macro_f1)/ len(macro_f1))
            print("AVG microF1: ", sum(micro_f1) / len(micro_f1))
            print("AVG weighted F1: ", sum(weighted_f1) / len(weighted_f1))

            print("AVG macro precision: ", sum(macro_precision) / len(macro_precision))
            print("AVG micro precision: ", sum(micro_precision) / len(micro_precision))
            print("AVG weighted precision: ", sum(weighted_precision) / len(weighted_precision))

            print("AVG macro recall: ", sum(macro_recall) / len(macro_recall))
            print("AVG micro recall: ", sum(micro_recall) / len(micro_recall))
            print("AVG weighted recall: ", sum(weighted_recall) / len(weighted_recall))

