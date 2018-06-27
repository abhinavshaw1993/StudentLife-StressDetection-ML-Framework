from main.experiments.generalized_global_classifier import GeneralizedGlobal

if __name__ == "__main__":
    gen_exp = GeneralizedGlobal()
    gen_exp.run_experiment(write=True, verbose=True)