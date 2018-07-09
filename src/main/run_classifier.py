from main.experiments.generalized_global_classifier import GeneralizedGlobalClassifier

if __name__ == "__main__":

    # Classifier
    gen_classif_exp = GeneralizedGlobalClassifier(config_file="generalized_global_classifier.yml")
    gen_classif_exp.run_experiment(train=False, write=True, verbose=False)



