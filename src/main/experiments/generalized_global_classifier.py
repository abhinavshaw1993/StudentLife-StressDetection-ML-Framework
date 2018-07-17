from main.experiments.experiment_base import ExperimentBase
from main.feature_processing.generalized_global_data_loader import GeneralizedGlobalDataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class GeneralizedGlobalClassifier(ExperimentBase):
    """
    This Class is for generalized global experiments.
    This class will generate all the output files if required.
    """

    def run_experiment(self, train=True, write=True, verbose=False):

        # initializing some things
        classifiers = []
        temp = []

        # Selecting Experiment Type
        experiment_type = {
            "predefined": "Global Generalized with Val and Test",
            "loso": "Global Generalized with Loso",
            "kfold": "Global Generalized with kfold CV with Randomization"
        }

        model_configs = ExperimentBase.get_model_configurations()

        # Getting the model configs and preparing param grid
        try:
            classifiers = self.get_ml_models()
        except Exception as e:
            print("No ML models Specified:", e)

        # Printing Details of Experiment

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

            # Iterating over all the models to test.
            for idx, model in enumerate(self.ml_models):
                try:
                    model_config = model_configs[model]
                except KeyError:
                    model_config = {}

                classifier = classifiers[idx]

                print("######################### Classifier {} running #######################".format(model))

                clf = GridSearchCV(classifier, param_grid=model_config, cv=exp.get_val_splitter(), n_jobs=-1,
                                   scoring="accuracy")
                clf.fit(train_x, train_y)
                best_estimator = clf.best_estimator_
                pred_y = best_estimator.predict(test_x)

                # Getting the relevant results
                best_param = clf.best_params_
                best_score = clf.best_score_
                accuracy = accuracy_score(test_y, pred_y)

                # f1 scores.
                micro_f1 = f1_score(test_y, pred_y, average="micro")
                macro_f1 = f1_score(test_y, pred_y, average="macro")
                weighted_f1 = f1_score(test_y, pred_y, average="weighted")

                # recall
                micro_recall = recall_score(test_y, pred_y, average="micro")
                macro_recall = recall_score(test_y, pred_y, average="macro")
                weighted_recall = recall_score(test_y, pred_y, average="weighted")

                # Precision.
                micro_precision = precision_score(test_y, pred_y, average='micro')
                macro_precision = precision_score(test_y, pred_y, average='macro')
                weighted_precision = precision_score(test_y, pred_y, average='weighted')

                confusion = confusion_matrix(test_y, pred_y)

                temp.append(
                    [model, best_param, best_score, splitter, accuracy, micro_f1, macro_f1, weighted_f1,
                     micro_recall, macro_recall, weighted_recall, micro_precision, macro_precision,
                     weighted_precision, confusion, experiment_type[splitter]]
                )

                ######## STD prints ##########
                print("Best Params: ", best_param)
                print()
                print("best score", best_score)
                print("accuracy: ", accuracy)
                print("")
                print("micro_f1: {}   macro_f1: {}  weigthed_f1: {}".format(micro_f1,
                                                                            macro_f1,
                                                                            weighted_f1))
                print("")
                print("micro_recall: {}   macro_recall: {}  weigthed_recall: {}".format(micro_recall,
                                                                                        macro_recall,
                                                                                        weighted_recall))
                print("")
                print("micro_precision: {}   macro_precision: {}  weigthed_precision: {}".format(micro_precision,
                                                                                                 macro_precision,
                                                                                                 weighted_precision))
                print("")
                print("############################### Confusion Matrix ############################")
                print(confusion)
                print("")
                if verbose:
                    print("best params", best_param)

                print("#################################################################################")

        result = pd.DataFrame(temp, columns=["Model", "Model_Config", "Best_CrossVal_Score", "Splitter",
                                             "Test_Accuracy", "Micro_f1", "Macro_f1", "Weighted_f1", "Micro_recall",
                                             "Macro_recall", "Weighted_recall", "Micro_precision",
                                             "Macro_precision", "Weighted_precision", "Confusion",
                                             "Experiment_Type"])

        # Generating Base line with the Given Data.
        most_freq_accuracy, most_freq_label = ExperimentBase.generate_baseline(test_y)
        result["Most_freq_accuracy"] = most_freq_accuracy
        result["Most_Freq_Label"] = most_freq_label

        if write:
            ExperimentBase.write_output("GeneralizedGlobalClassifier", result, None)

        # Printing results
        if verbose:
            print("Most Freq Baseline:{}, Most Freq Label: {} ".format(most_freq_accuracy, most_freq_label))
