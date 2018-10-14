from main.experiments.confidence_interval_loso_kfold import ConfidenceIntervalsResult

if __name__ == "__main__":

    # Classifier
    confidence_interval_exp = ConfidenceIntervalsResult(config_file="confidence_interval_loso_kfold.yml")
    confidence_interval_exp.run_experiment(train=True, write=True, verbose=False)



