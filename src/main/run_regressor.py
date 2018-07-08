from main.experiments.generalized_global_regressor import GeneralizedGlobalRegressor

if __name__ == "__main__":

    # Regressor
    gen_regress_exp = GeneralizedGlobalRegressor(config_file="generalized_global_regressor.yml")
    gen_regress_exp.run_experiment(write=True, verbose=False)


