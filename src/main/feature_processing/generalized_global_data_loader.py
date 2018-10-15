import pandas as pd
import numpy as np
from main.feature_processing.data_loader_base import DataLoaderBase
from main.feature_processing.transformer import get_transformer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import StratifiedKFold
from main.feature_processing.feature_selection import select_features


class GeneralizedGlobalDataLoader(DataLoaderBase):
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()
    train_indices = pd.DataFrame()
    val_indices = pd.DataFrame()
    student_count = 0
    students = pd.DataFrame()
    train_x = pd.DataFrame()
    train_y = pd.DataFrame()

    def get_data(self, stress_agg='min', previous_stress=True, feature_selection=True,
                 feature_selection_type='classification', verbose=False, segragate_by_median=False, truncate_sq=False):

        file_list = DataLoaderBase.get_file_list(self.aggregation_window)

        if truncate_sq:
            file_list = file_list[:3]

        self.student_count = len(file_list)

        for idx, file in enumerate(file_list):

            if self.aggregation_window == 'd':
                temp_data = pd.read_csv(file,
                                        index_col=0,
                                        header=[0, 1])

            if previous_stress:
                # Modelling previous stress level to the data frame
                previous_stress_levels = temp_data.loc[:, ("stress_level", stress_agg)]
                previous_stress_levels_len = len(previous_stress_levels)
                previous_stress_levels = [2] + list(previous_stress_levels)
                previous_stress_levels = previous_stress_levels[:previous_stress_levels_len]
                index = len(temp_data.columns) - 3

                temp_data.insert(index, "previous_stress_level", previous_stress_levels)

            if self.splitter == "predefined":
                self.train_data = self.train_data.append(temp_data[:'2013-05-01'])
                self.val_data = self.val_data.append(temp_data['2013-05-01':'2013-05-14'])
                self.test_data = self.test_data.append(temp_data['2013-05-15':])
            elif self.splitter == "loso" or self.splitter == "kfold":
                if idx == 1 or ('student 10' in file) or (
                        ('student 20' in file) or ('student 20' in file) or ('student 41' in file) and idx != 1):
                    self.test_data = self.test_data.append(temp_data)
                else:
                    self.train_data = self.train_data.append(temp_data)

        # Now that the data has been read let us produce test train split, since it is global generalized we

        if self.splitter == "predefined":
            self.train_data["set"] = "training_set"
            self.val_data["set"] = "val_set"
            self.train_data = self.train_data.append(self.val_data, ignore_index=True)
            self.train_data.reset_index(drop=True)
            self.train_indices = self.train_data[self.train_data['set'] == "training_set"].index.values
            self.val_indices = self.train_data[self.train_data['set'] == "val_set"].index.values
            self.train_data.drop(columns="set", inplace=True)

        # Fixing Inf Values, NaN values in df.
        self.train_data.replace(np.inf, 10000000, inplace=True)
        self.train_data.replace(-np.inf, -10000000, inplace=True)
        self.train_data.fillna(method='pad', inplace=True)
        self.train_data.fillna(value=0, inplace=True)

        self.test_data.replace(np.inf, 10000000, inplace=True)
        self.test_data.replace(-np.inf, -10000000, inplace=True)
        self.test_data.fillna(method='pad', inplace=True)
        self.test_data.fillna(value=0, inplace=True)

        # Slicing our t
        train_x, train_y = self.train_data.iloc[:, :-3], self.train_data.loc[:, ("stress_level", stress_agg)]
        test_x, test_y = self.test_data.iloc[:, :-3], self.test_data.loc[:, ("stress_level", stress_agg)]

        # Calculating Label Distribution for train and test.
        train_label_dist = train_y.value_counts()
        test_label_dist = test_y.value_counts()

        # Changing mapping to adjusted stress values.
        train_y = train_y.apply(DataLoaderBase.adjust_stress_values)
        test_y = test_y.apply(DataLoaderBase.adjust_stress_values)

        if segragate_by_median:
            # Segragate by media stress.
            train_y = train_y.apply(DataLoaderBase.segregate_y_labels_as_median)
            test_y = test_y.apply(DataLoaderBase.segregate_y_labels_as_median)

        # Selecting Features
        if feature_selection:
            train_x, train_y, test_x, test_y = select_features(train_x, train_y, test_x, test_y,
                                                               selection_type=feature_selection_type, num_features=20,
                                                               verbose=verbose)

        self.students = train_x.iloc[:, 0]

        # Transforming Data by getting custom transformer.
        transformer = get_transformer(self.transformer_type)
        train_x = pd.DataFrame(transformer.fit_transform(train_x), columns=train_x.columns)
        test_x = pd.DataFrame(transformer.fit_transform(test_x), columns=test_x.columns)

        self.train_x = train_x
        self.train_y = train_y

        if verbose:
            print("train_data_len: {}, val_data_len: {}, test_data_len: {}".format(len(self.train_data),
                                                                                   len(self.val_data),
                                                                                   len(self.test_data)))
            print()
            print("Is NaN", self.train_data.isnull().any(axis=1).any())
            print()

        return self.train_x, self.train_y, test_x, test_y, train_label_dist, test_label_dist

    def get_val_splitter(self):
        if self.splitter == "predefined":
            return self.__get_predefined_splitter()
        elif self.splitter == "loso":
            return LeaveOneGroupOut().split(self.train_x, groups=self.students)
        elif self.splitter == 'kfold':
            return StratifiedKFold(shuffle=True, n_splits=5).split(X=self.train_x, y=self.train_y)
        else:
            return self.__get_predefined_splitter()

    def __get_predefined_splitter(self):

        yield (self.train_indices, self.val_indices)