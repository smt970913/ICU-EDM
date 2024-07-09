import pandas as pd
from collections import Counter

import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

import math
import numpy as np
from scipy import stats

from multiprocessing import Pool

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPrepare:
    def __init__(self, data):
        self.data = data.copy()
        self.feature_list = ['anchor_age', 'patientweight',
                             'Heart Rate', 'Arterial O2 pressure', 'Hemoglobin',
                             'Arterial CO2 Pressure', 'PH (Venous)', 'Hematocrit (serum)', 'WBC',
                             'Chloride (serum)', 'Creatinine (serum)', 'Glucose (serum)',
                             'Magnesium', 'Sodium (serum)', 'PH (Arterial)', 'Inspired O2 Fraction',
                             'Tidal Volume (set)', 'Tidal Volume (observed)',
                             'Tidal Volume (spontaneous)',
                             'Respiratory Rate', 'Respiratory Rate (Set)',
                             'Respiratory Rate (spontaneous)', 'Respiratory Rate (Total)',
                             'Arterial Base Excess', 'BUN', 'Ionized Calcium', 'Total Bilirubin',
                             'Venous CO2 Pressure', 'Venous O2 Pressure', 'Sodium (whole blood)',
                             'Chloride (whole blood)', 'Glucose (whole blood)',
                             'Hematocrit (whole blood - calc)', 'Potassium (serum)', 'HCO3 (serum)',
                             'Albumin', 'Platelet Count', 'Potassium (whole blood)',
                             'Prothrombin time', 'PTT', 'INR', 'M',
                             'Blood Pressure Systolic', 'Blood Pressure Diastolic',
                             'Blood Pressure Mean', 'Temperature C', 'SaO2', 'GCS score',
                             'PEEP Level']
        self.drop_list = []
        self.middle_list = []
        self.knn_list = []
        self.names_var = []

    def identify_missing_values(self):
        for i in self.feature_list:
            if (self.data[i].isnull().sum() / len(self.data)) > 0.60:
                self.drop_list.append(i)
            elif ((self.data[i].isnull().sum() / len(self.data)) <= 0.60) & ((self.data[i].isnull().sum() / len(self.data)) >= 0.10):
                self.middle_list.append(i)
            elif (self.data[i].isnull().sum() / len(self.data)) < 0.10:
                self.knn_list.append(i)

    def preprocess_data(self):
        self.data = self.data.drop(columns=self.drop_list)
        self.names_var = self.middle_list + self.knn_list
        self.abv_data = self.data[self.names_var]
        self.inspect_col = ['Arterial O2 pressure', 'Hemoglobin', 'Arterial CO2 Pressure', 'Hematocrit (serum)', 'WBC', 'Chloride (serum)',
                            'Creatinine (serum)', 'Glucose (serum)', 'Magnesium', 'Sodium (serum)', 'PH (Arterial)', 'Tidal Volume (observed)',
                            'Tidal Volume (spontaneous)', 'Respiratory Rate (Set)', 'Respiratory Rate (spontaneous)', 'Respiratory Rate (Total)',
                            'Arterial Base Excess', 'BUN', 'Potassium (serum)', 'HCO3 (serum)', 'Platelet Count', 'Temperature C',
                            'patientweight', 'Heart Rate', 'Respiratory Rate', 'Inspired O2 Fraction', 'Blood Pressure Systolic',
                            'Blood Pressure Diastolic', 'Blood Pressure Mean', 'SaO2']
        self.ab_data_sub = self.abv_data[self.inspect_col]

    def handle_outliers(self):
        q1 = self.ab_data_sub.quantile(0.25)
        q3 = self.ab_data_sub.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        for column in self.ab_data_sub.columns:
            self.data.loc[self.data[column] > upper_bound[column], column] = np.nan
            self.data.loc[self.data[column] < lower_bound[column], column] = np.nan

        self.abv_data = self.data[self.names_var]

    def process(self):
        self.identify_missing_values()
        self.preprocess_data()
        self.handle_outliers()
        return self.data


class PatientDataImputation:
    def __init__(self, data, middle_list, knn_list):
        self.data = data.copy()
        self.middle_list = middle_list
        self.knn_list = knn_list

    def forward_fill_missing_values(self):
        for i in range(len(self.middle_list)):
            self.data[self.middle_list[i]] = self.data.groupby(by=['stay_id', 'ext_time'])[self.middle_list[i]].ffill()

    def reorder_columns(self):
        names_reorder = ['subject_id', 'stay_id', 'mv_id', 'ext_id', 'ext_time', 'epoch', 'EXT', 'dod',
                         'first_careunit', 'last_careunit', 'intime', 'outtime', 'starttime', 'endtime', 'time',
                         'ext_fail', 'ext_death_bfe', 'ext_death_afe', 'itemid', 'value', 'valueuom', 'storetime',
                         'ordercategoryname', 'los', 'Ventilation_duration', 'ICU_LOS_hr', 'RLOS_MV_start',
                         'RLOS_MV_end', 'RLOS_MV_epoch', 'race', 'anchor_age', 'M', 'patientweight', 'Heart Rate',
                         'Arterial O2 pressure', 'Hemoglobin', 'Arterial CO2 Pressure', 'Hematocrit (serum)', 'WBC',
                         'Chloride (serum)', 'Creatinine (serum)', 'Glucose (serum)', 'Magnesium', 'Sodium (serum)',
                         'PH (Arterial)', 'Inspired O2 Fraction', 'Tidal Volume (set)', 'Tidal Volume (observed)',
                         'Tidal Volume (spontaneous)', 'Respiratory Rate', 'Respiratory Rate (Set)',
                         'Respiratory Rate (spontaneous)', 'Respiratory Rate (Total)', 'Arterial Base Excess', 'BUN',
                         'Potassium (serum)', 'HCO3 (serum)', 'Platelet Count', 'Blood Pressure Systolic',
                         'Blood Pressure Diastolic', 'Blood Pressure Mean', 'Temperature C', 'SaO2', 'GCS score',
                         'PEEP Level']

        self.data = self.data[names_reorder]

    def linear_impute_missing_values(self):
        weight_missing_value = self.data[self.data['patientweight'].isna()]
        self.data = self.data[~self.data['ext_id'].isin(pd.unique(weight_missing_value['ext_id']))]

        feature_list_1 = self.middle_list + self.knn_list

        for i in range(len(feature_list_1)):
            self.data[feature_list_1[i]] = self.data.groupby(by=['stay_id', 'ext_time'])[feature_list_1[i]].apply(
                lambda x: x.interpolate(method='linear'))
        drop_feature_list = ['Tidal Volume (set)', 'Respiratory Rate (Set)']

        self.data = self.data.drop(columns=drop_feature_list)

    def knn_impute_missing_values(self, num_neigh=5, scaler=MinMaxScaler()):

        self.data_ini = self.data[self.data['epoch'] == 1]

        imputer = KNNImputer(n_neighbors=num_neigh)

        columns_with_missing_values = self.data.columns[self.data.isnull().any()].tolist()
        columns_with_missing_values.remove('dod')

        data_pre = self.data[columns_with_missing_values]

        data_pre[columns_with_missing_values] = scaler.fit_transform(data_pre[columns_with_missing_values])
        data_pre[columns_with_missing_values] = imputer.fit_transform(data_pre[columns_with_missing_values])
        data_pre[columns_with_missing_values] = scaler.inverse_transform(data_pre[columns_with_missing_values])

        self.data[columns_with_missing_values] = data_pre[columns_with_missing_values]