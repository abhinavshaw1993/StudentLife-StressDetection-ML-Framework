from main.experiments.experiment_base import ExperimentBase
from main.feature_processing.generalized_global_data_loader import GeneralizedGlobalDataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
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

    def run_experiment(self, write=True, verbose=False):

        # initializing some things
        classifiers = []
        param_grid = []
        result = pd.DataFrame()
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

        for splitter in self.splitter:

            print("########## {} split running #########".format(splitter))

            exp = GeneralizedGlobalDataLoader(agg_window=self.agg_window, splitter=splitter,
                                              transformer_type=self.transformer)
            train_x, train_y, test_x, test_y, train_label_dist, test_label_dist = exp.get_data(
                stress_agg=self.stress_agg, previous_stress=self.previous_stress, verbose=False)
            # cv = exp.get_val_splitter()

            # Iterating over all the models to test.
            for idx, model in enumerate(self.ml_models):
                try:
                    model_config = model_configs[model]
                except KeyError:
                    model_config = {}

                classifier = classifiers[idx]

                print("######## Classifier {} running #######".format(model))

                clf = GridSearchCV(classifier, param_grid=model_config, cv=exp.get_val_splitter(), n_jobs=-1,
                                   scoring="accuracy")
                clf.fit(train_x, train_y)
                best_estimator = clf.best_estimator_
                pred_y = best_estimator.predict(test_x)

                # Getting the relevant results
                best_param = clf.best_params_
                best_score = clf.best_score_
                accuracy = accuracy_score(test_y, pred_y)
                f1 = f1_score(test_y, pred_y, average=None)

                temp.append(
                    [model, best_param, best_score, splitter, accuracy, f1, experiment_type[splitter]]
                )

                ######## STD prints ##########
                print("best score", best_score)
                print("accuracy: ", accuracy)
                print("f1: ", f1)

                if verbose:
                    print("best params", best_param)

                print("#########################################################################")

        result = pd.DataFrame(temp, columns=["Model", "Model_Config", "Best_CrossVal_Score", "Splitter",
                                             "Test_Accuracy", "f1_Scores", "Experiment_type"])
        # Generating Base line with the Given Data.
        most_freq_accuracy, most_freq_label = ExperimentBase.generate_baseline(test_y)
        result["Most_freq_accuracy"] = most_freq_accuracy
        result["Most_Freq_Label"] = most_freq_label

        if write:
            ExperimentBase.write_output("GeneralizedGlobalClassifier", result, None)

        # Printing results
        if verbose:
            print("Most Freq Baseline:{}, Most Freq Label: {} ".format(most_freq_accuracy, most_freq_label))

