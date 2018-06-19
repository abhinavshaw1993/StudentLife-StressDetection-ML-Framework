import pandas as pd
import warnings
import yaml
import os
from main.data_processor.agrregation_functions import extend_complex_features

warnings.filterwarnings("ignore")

owd = os.getcwd()
print("owd: ", owd)

fwd = "F:\Projects\DataProcessor"
print("working directory of flat files: ", fwd)
os.chdir(fwd)

# Reading from YML file.
with open("./feature_details.yml", "r") as ymlfile:
    feature_cfg = yaml.load(ymlfile)

os.chdir(fwd)
data_dir = fwd + "\StudentLife Data"
os.chdir(data_dir)
dir_list = [x for x in os.listdir('.') if "student" in x]

# Window Length
freq = 'd'

# Skipping Student 0, because of bad data.
print("dir list: ", dir_list)

dir_list = dir_list[30:]

print("DirList that will be precessed: ", dir_list)
converter_dict = {'time': pd.to_datetime}


for folder in dir_list:

    path = data_dir + "\\" + folder
    os.chdir(path)
    feature_files = feature_cfg['feats_to_use_from_file']
    feature_train_list = []

    print("Working on: ", folder)

    for file in feature_files:
        raw_feature_data = pd.read_csv(file + "_train_x.csv",
                                       skip_blank_lines=False,
                                       index_col=0,
                                       converters=converter_dict
                                       )

        # Resetting Index.
        raw_feature_data = raw_feature_data.reset_index(drop=True)
        raw_feature_data = raw_feature_data.set_index(keys='time')

        # Feature Columns required, avoiding Nusance columns.
        feature_cols = [col for col in raw_feature_data.columns.values if col not in ['student_id', 'stress_level']]

        # Exctracting information and columns.
        student_id = int(folder.split(" ")[1])
        raw_stress_data = raw_feature_data['stress_level']
        raw_feature_data = raw_feature_data[feature_cols]

        # resampling at fixed window length.
        resampled_raw_features = raw_feature_data.resample(rule=freq)

        # Simple aggregates being prepared.
        simple_low_lvl_features = {f: f for f in feature_cfg[file]["simple"]}
        simple_aggregates = resampled_raw_features.agg(simple_low_lvl_features)
        simple_aggregates.fillna(value=0, inplace=True)

        complex_aggregates = pd.DataFrame()

        # Complex aggregations functions
        if "linear" in feature_cfg[file].keys() and feature_cfg[file]['linear']:
            # Linear Slope and intercept.
            linear_slope = extend_complex_features('linear', resampled_raw_features, feature_cols)
            complex_aggregates = pd.concat([complex_aggregates, linear_slope], axis=1)

        if "poly" in feature_cfg[file].keys() and feature_cfg[file]['poly']:
            # Poly Coeffs.
            poly_coeff = extend_complex_features('poly', resampled_raw_features, feature_cols)
            complex_aggregates = pd.concat([complex_aggregates, poly_coeff], axis=1)

        if "iqr" in feature_cfg[file].keys() and feature_cfg[file]['iqr']:
            # Inter Quartile Range.
            iqr_df = extend_complex_features('iqr', resampled_raw_features, feature_cols)
            complex_aggregates = pd.concat([complex_aggregates, iqr_df], axis=1)

        if "kurtosis" in feature_cfg[file].keys() and feature_cfg[file]['kurtosis']:
            # Kurtosis.
            kurt_df = extend_complex_features('kurtosis', resampled_raw_features, feature_cols)
            complex_aggregates = pd.concat([complex_aggregates, kurt_df], axis=1)

        if "mcr" in feature_cfg[file].keys() and feature_cfg[file]['mcr']:
            # mean crossign rate.
            mcr_df = extend_complex_features('mcr', resampled_raw_features, feature_cols)
            complex_aggregates = pd.concat([complex_aggregates, mcr_df], axis=1)

        if "fft" in feature_cfg[file].keys() and feature_cfg[file]['fft']:
            # Fourier Transform magnatude spectra.
            fft_df = extend_complex_features('fft', resampled_raw_features, feature_cols)
            complex_aggregates = pd.concat([complex_aggregates, fft_df], axis=1)

        final_aggregates = pd.concat([simple_aggregates, complex_aggregates], axis=1)

        feature_train_list.append(final_aggregates)

    # resampling stress levels.
    resampled_raw_stress_levels = raw_stress_data.resample(rule=freq)
    aggregated_stress_levels = resampled_raw_stress_levels.agg({'stress_level': ['min', 'max', 'mean']})
    aggregated_stress_levels.fillna(method='pad', inplace=True)

    if feature_cfg['dow']:
        # Extracting Day of the week.
        aggregated_stress_levels.insert(loc=0, column='dow', value=simple_aggregates.index.dayofweek)

    # preparing final dataset.
    train_data = pd.concat(feature_train_list + [aggregated_stress_levels], axis=1)
    train_data.insert(loc=0, column='student_id', value=student_id)

    os.chdir(owd)
    os.chdir('../data/aggregated_data')

    if not os.path.isdir("student "+str(student_id)):
        os.mkdir("student " + str(student_id))

    os.chdir("./"+"student " + str(student_id))

    if freq == 'd':
        train_data.to_csv("one_day_aggregate.csv", index=True)