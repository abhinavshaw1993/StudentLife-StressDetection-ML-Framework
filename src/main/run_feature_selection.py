from main.experiments.classification_feature_selection import GeneralizedFeatureSelection

if __name__ == "__main__":

    # Classifier
    feature_selection_exp = GeneralizedFeatureSelection(config_file="generalized_gloabal_feature_selection.yml")
    feature_selection_exp.run_experiment(train=True, write=True, verbose=False)



