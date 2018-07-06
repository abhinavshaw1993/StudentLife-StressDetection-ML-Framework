from main.experiments.generalized_global_classifier import GeneralizedGlobalClassifier
# from main.experiments.generalized_global_regressor import GeneralizedGlobalRegressor

if __name__ == "__main__":

    # Classifier
    gen_classif_exp = GeneralizedGlobalClassifier(config_file="generalized_global_classifier.yml")
    gen_classif_exp.run_experiment(write=True, verbose=False)

    # # Regressor
    # gen_regress_exp = GeneralizedGlobalRegressor(config_file="generalized_global_regressor.yml")
    # gen_regress_exp.run_experiment(write=True, verbose=False)


