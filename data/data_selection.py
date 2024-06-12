import pandas as pd
from collections import Counter

import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

import numpy as np

from multiprocessing import Pool

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

import math

from tqdm import tqdm
import time

class ICUDataInput:
    def __init__(self):
        self.d_items_data = None
        self.input_events_data = None
        self.pro_events_data = None
        self.output_event_data = None
        self.ICU_patient_data = None
        self.d_labitems_data = None
        self.ICU_patient_data_1 = None
        self.icu_ad_list = None
        self.patient_list = None

    def load_data(self):
        self.d_items_data = pd.read_csv('d_items.csv.gz', compression = 'gzip')
        self.input_events_data = pd.read_csv('inputevents.csv.gz', compression = 'gzip')
        self.pro_events_data = pd.read_csv('procedureevents.csv.gz', compression = 'gzip')
        self.output_event_data = pd.read_csv('outputevents.csv.gz', compression = 'gzip')
        self.ICU_patient_data = pd.read_csv('icustays.csv.gz', compression = 'gzip')
        self.d_labitems_data = pd.read_csv('d_labitems.csv.gz', compression = 'gzip')

    def process_data(self):
        icu_patient_names_select = ['subject_id', 'stay_id', 'first_careunit', 'last_careunit', 'intime', 'outtime', 'los']
        self.ICU_patient_data_1 = self.ICU_patient_data[icu_patient_names_select]

        items_names_select = ['itemid', 'label', 'linksto', 'category', 'unitname', 'param_type']
        self.d_items_data = self.d_items_data[items_names_select]

        self.icu_ad_list = pd.unique(self.ICU_patient_data_1["stay_id"])
        self.patient_list = pd.unique(self.ICU_patient_data_1["subject_id"])

    def get_icu_patient_data(self):
        return self.ICU_patient_data_1

    def get_icu_ad_list(self):
        return self.icu_ad_list

    def get_patient_list(self):
        return self.patient_list

class VariableSelect:
    def __init__(self, d_items_path, pro_events_path):
        self.d_items_path = d_items_path
        self.pro_events_path = pro_events_path
        self.d_items_data = None
        self.pro_events_data = None
        self.d_items_data_1 = None
        self.d_items_data_2 = None
        self.pro_events_data_1 = None
        self.item_id_list = None

        self.VitalSigns_id = [220045, 220048, 220179, 220050, 220180, 220051, 220052, 220181, 225312, 220210, 224690, 223761, 223762, 220277]
        self.GCS_score_id = [223901, 223900, 220739]
        self.Vent_para_id = [220339, 224700, 224685, 224684, 224686, 223835, 223848, 223849]
        self.Labs_id = [225624, 226536, 220602, 227464, 226534, 226537, 229761, 220653, 220546, 227466, 227467, 227457, 220274, 223830,
                        220228, 220235, 220224, 226062, 226063, 227456, 226540, 224828, 220635, 220545, 220615, 220621, 220645]
        self.General_id = [224639, 226260, 226512, 226531, 226892, 227428]
        self.ADT_id = [220003, 226228, 226545]
        self.add_id_1 = [224719, 226862, 228878, 225624, 220615, 229761, 227465, 227442, 227443,
                         225651, 225690, 226566, 227489, 226627, 220994, 227519, 227488, 225667, 228699, 228709, 228713,
                         228703, 228704, 228705, 225309, 225310, 220227, 223830, 224688, 224689]
        self.Ventilation_id = [224385, 225468, 225477, 225792, 225794, 227194]

    def load_data(self):
        self.d_items_data = pd.read_csv(self.d_items_path, compression='gzip')
        self.pro_events_data = pd.read_csv(self.pro_events_path, compression='gzip')

    def select_variables(self):
        variable_list = self.VitalSigns_id + self.GCS_score_id + self.Vent_para_id + self.Labs_id + self.General_id + self.ADT_id + self.add_id_1
        self.d_items_data_1 = self.d_items_data[self.d_items_data['itemid'].isin(variable_list)]
        self.d_items_data_2 = self.d_items_data[self.d_items_data['itemid'].isin(self.Ventilation_id)]

        names_select = ['subject_id', 'stay_id', 'starttime', 'endtime', 'itemid', 'value', 'valueuom',
                        'storetime', 'ordercategoryname', 'patientweight']
        self.pro_events_data_1 = self.pro_events_data[names_select]
        self.pro_events_data_1 = self.pro_events_data_1[self.pro_events_data_1['itemid'].isin(self.d_items_data_2['itemid'])]

        self.item_id_list = list(self.d_items_data_2['itemid']) + list(self.d_items_data_1['itemid'])

    def get_selected_data(self):
        return {
            'd_items_data_1': self.d_items_data_1,
            'd_items_data_2': self.d_items_data_2,
            'pro_events_data_1': self.pro_events_data_1,
            'item_id_list': self.item_id_list
        }

class ChartProcess:
    def __init__(self, chart_events_path, pro_events_data_1):
        self.chart_events_path = chart_events_path
        self.pro_events_data_1 = pro_events_data_1
        self.chart_events = None
        self.chart_events_data = None

    def load_and_filter_data(self):
        self.chart_events = dd.read_csv(
            self.chart_events_path,
            dtype={'cgid': 'float64', 'stay_id': 'float64', 'error': 'float64',
                   'resultstatus': 'object', 'stopped': 'object', 'value': 'object',
                   'valuenum': 'float64', 'warning': 'float64', 'valueuom': 'object',
                   'caregiver_id': 'float64'},
            low_memory=False, blocksize=None, compression='gzip'
        )
        self.chart_events = self.chart_events[self.chart_events.stay_id.isin(self.pro_events_data_1['stay_id'].unique())]
        self.chart_events_data = self.chart_events.compute(assume_missing=True)

    def select_columns(self):
        names_select = ['subject_id', 'stay_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom']
        self.chart_events_data = self.chart_events_data[names_select]

    def get_chart_events_data(self):
        return self.chart_events_data


class DataForwardProcess:
    def __init__(self, ICU_patient_data, pro_events_data_1, chart_events_data, d_items_data_1):
        self.ICU_patient_data = ICU_patient_data
        self.pro_events_data_1 = pro_events_data_1
        self.chart_events_data = chart_events_data
        self.d_items_data_1 = d_items_data_1

        self.ICU_unit_exp = [
            'Medical Intensive Care Unit (MICU)',
            'Surgical Intensive Care Unit (SICU)',
            'Medical/Surgical Intensive Care Unit (MICU/SICU)',
            'Cardiac Vascular Intensive Care Unit (CVICU)',
            'Trauma SICU (TSICU)'
        ]

        self.ICU_patient_data_test = None
        self.pro_events_data_2 = None
        self.pro_events_data_3 = None
        self.comb_data = None
        self.chart_events_data_1 = None
        self.d_items_data_chart = None
        self.d_items_data_output = None
        self.d_items_data_datetime = None
        self.d_items_data_ingred = None

    def filter_ICU_patients(self):
        self.ICU_patient_data_test = self.ICU_patient_data[
            self.ICU_patient_data['first_careunit'].isin(self.ICU_unit_exp)]

    def filter_procedure_events(self):
        self.pro_events_data_2 = self.pro_events_data_1[
            self.pro_events_data_1['stay_id'].isin(self.ICU_patient_data_test['stay_id'])]
        self.pro_events_data_3 = self.pro_events_data_2[self.pro_events_data_2['stay_id'].isin(
            self.pro_events_data_2[self.pro_events_data_2['ordercategoryname'] == 'Ventilation']['stay_id'])]

    def merge_data(self):
        self.comb_data = pd.merge(self.ICU_patient_data_test, self.pro_events_data_3, how='inner', on="stay_id")
        self.comb_data = self.comb_data.drop(columns=['subject_id_y'])

    def filter_chart_events(self):
        self.chart_events_data_1 = self.chart_events_data[
            self.chart_events_data['stay_id'].isin(self.comb_data['stay_id'])]

    def classify_d_items_data(self):
        self.d_items_data_chart = self.d_items_data_1[self.d_items_data_1['linksto'] == 'chartevents']
        self.d_items_data_output = self.d_items_data_1[self.d_items_data_1['linksto'] == 'outputevents']
        self.d_items_data_datetime = self.d_items_data_1[self.d_items_data_1['linksto'] == 'datetimeevents']
        self.d_items_data_ingred = self.d_items_data_1[self.d_items_data_1['linksto'] == 'ingredientevents']

    def process_all(self):
        self.filter_ICU_patients()
        self.filter_procedure_events()
        self.merge_data()
        self.filter_chart_events()
        self.classify_d_items_data()

    def get_processed_data(self):
        return {
            'ICU_patient_data_test': self.ICU_patient_data_test,
            'pro_events_data_2': self.pro_events_data_2,
            'pro_events_data_3': self.pro_events_data_3,
            'comb_data': self.comb_data,
            'chart_events_data_1': self.chart_events_data_1,
            'd_items_data_chart': self.d_items_data_chart,
            'd_items_data_output': self.d_items_data_output,
            'd_items_data_datetime': self.d_items_data_datetime,
            'd_items_data_ingred': self.d_items_data_ingred
        }

class DataSelect:
    def __init__(self, comb_data, d_items_data_1):
        self.comb_data = comb_data.rename(columns={'subject_id_x': 'subject_id'})
        self.d_items_data_1 = d_items_data_1
        self.comb_data_NIV = None
        self.comb_data_1 = None
        self.comb_data_2 = None
        self.d_items_data_3 = None
        self.d_items_data_4 = None

    def filter_comb_data(self):
        self.comb_data = self.comb_data[self.comb_data['los'] <= 30.00]

    def find_NIV_list(self):
        NIV_list = []
        for i in range(len(self.comb_data)):
            if self.comb_data['itemid'].iloc[i] == 225794:
                NIV_list.append(self.comb_data['stay_id'].iloc[i])
        self.comb_data_NIV = self.comb_data[self.comb_data['itemid'] == 225794]

    def classify_d_items_data(self):
        self.d_items_data_3 = self.d_items_data_1[self.d_items_data_1['param_type'] == 'Date and time']
        self.d_items_data_4 = self.d_items_data_1[self.d_items_data_1['param_type'] == 'Text']

    def filter_comb_data_1(self):
        self.comb_data_up = self.comb_data[self.comb_data['itemid'].isin([225468, 225477])]
        up_list_adm = pd.unique(self.comb_data_up['stay_id'])
        self.comb_data_1 = self.comb_data[self.comb_data['itemid'].isin([224385, 225792, 225794, 227194])]
        self.comb_data_1['intime'] = pd.to_datetime(self.comb_data_1['intime'])
        self.comb_data_1['outtime'] = pd.to_datetime(self.comb_data_1['outtime'])
        self.comb_data_1['starttime'] = pd.to_datetime(self.comb_data_1['starttime'])
        self.comb_data_1['endtime'] = pd.to_datetime(self.comb_data_1['endtime'])
        self.comb_data_1['TD_ICU'] = self.comb_data_1['outtime'] - self.comb_data_1['intime']
        self.comb_data_1['TD_MV'] = self.comb_data_1['endtime'] - self.comb_data_1['starttime']
        self.comb_data_1 = self.comb_data_1[~self.comb_data_1['stay_id'].isin(up_list_adm)]
        self.comb_data_1 = self.comb_data_1[self.comb_data_1['TD_MV'] <= pd.Timedelta('7 days 00:00:00')]
        self.comb_data_1['RLOS'] = self.comb_data_1['outtime'] - self.comb_data_1['endtime']
        self.comb_data_1['LOS_ini'] = self.comb_data_1['outtime'] - self.comb_data_1['starttime']
        self.comb_data_1 = self.comb_data_1[~self.comb_data_1['stay_id'].isin(self.find_up_list_adm())]
        self.comb_data_1 = self.comb_data_1[self.comb_data_1['itemid'].isin([225792, 225794])]
        self.comb_data_1 = self.comb_data_1.sort_values(by=['subject_id', 'stay_id', 'starttime', 'endtime'])
        self.comb_data_1['mv_id'] = 1
        for i in range(1, len(self.comb_data_1)):
            if self.comb_data_1['stay_id'].iloc[i] == self.comb_data_1['stay_id'].iloc[i-1]:
                self.comb_data_1['mv_id'].iloc[i] = self.comb_data_1['mv_id'].iloc[i-1] + 1

    def find_up_list_adm(self):
        comb_data_up = self.comb_data[self.comb_data['itemid'].isin([225468, 225477])]
        return pd.unique(comb_data_up['stay_id'])

    def mark_ext_fail(self):
        admission_list = pd.unique(self.comb_data_1['stay_id'])
        ext_fail_list = []
        mv_id_record = []
        for i in tqdm(range(len(admission_list))):
            sub_table = self.comb_data_1[self.comb_data_1['stay_id'] == admission_list[i]]
            if len(sub_table) >= 2:
                for j in range(len(sub_table) - 1):
                    if sub_table['itemid'].iloc[j] == 225792:  # this is an invasive MV event
                        for k in range(j + 1, len(sub_table)):
                            if sub_table['starttime'].iloc[k] - sub_table['endtime'].iloc[j] <= pd.Timedelta('7 days 00:00:00'):
                                ext_fail_list.append(admission_list[i])
                                mv_id_record.append(sub_table['mv_id'].iloc[j])
                    else:
                        continue
            else:
                continue
        self.comb_data_1['ext_fail'] = 0
        for i in range(len(self.comb_data_1)):
            for j in range(len(ext_fail_list)):
                if self.comb_data_1['stay_id'].iloc[i] == ext_fail_list[j]:
                    self.comb_data_1.loc[
                        ((self.comb_data_1['stay_id'] == ext_fail_list[j]) &
                         (self.comb_data_1['mv_id'] == mv_id_record[j])), 'ext_fail'] = 1

    def filter_and_sort(self):
        admission_list = pd.unique(self.comb_data_1['stay_id'])
        drop_list_1 = []
        drop_list_2 = []
        drop_list_mv = []
        for i in tqdm(range(len(admission_list))):
            sub_table = self.comb_data_1[self.comb_data_1['stay_id'] == admission_list[i]]
            if len(pd.unique(sub_table['itemid'])) == 1:
                if pd.unique(sub_table['itemid']) == 225794:
                    drop_list_1.append(admission_list[i])
                else:
                    continue
            else:
                for j in range(len(sub_table) - 1):
                    if sub_table['itemid'].iloc[j] == 225794:
                        for k in range(j + 1, len(sub_table)):
                            if sub_table['itemid'].iloc[k] == 225792:
                                if sub_table['starttime'].iloc[k] - sub_table['endtime'].iloc[j] < pd.Timedelta('0 days 00:00:00'):
                                    drop_list_2.append(admission_list[i])
                                    drop_list_mv.append(sub_table['mv_id'].iloc[k])
                    else:
                        for k in range(j + 1, len(sub_table)):
                            if sub_table['itemid'].iloc[k] == 225794:
                                if sub_table['starttime'].iloc[k] - sub_table['endtime'].iloc[j] < pd.Timedelta('0 days 00:00:00'):
                                    drop_list_2.append(admission_list[i])
                                    drop_list_mv.append(sub_table['mv_id'].iloc[j])
        self.comb_data_1 = self.comb_data_1[~self.comb_data_1['stay_id'].isin(drop_list_1)]
        for i in range(len(drop_list_2)):
            condition = (self.comb_data_1['stay_id'] == drop_list_2[i]) & (self.comb_data_1['mv_id'] == drop_list_mv[i])
            self.comb_data_1 = self.comb_data_1[~condition]
        self.comb_data_2 = self.comb_data_1[self.comb_data_1['itemid'] == 225792]

    def process_all(self):
        self.filter_comb_data()
        self.find_NIV_list()
        self.classify_d_items_data()
        self.filter_comb_data_1()
        self.mark_ext_fail()
        self.filter_and_sort()

    def get_processed_data(self):
        return {
            'comb_data_NIV': self.comb_data_NIV,
            'comb_data_1': self.comb_data_1,
            'comb_data_2': self.comb_data_2,
            'd_items_data_3': self.d_items_data_3,
            'd_items_data_4': self.d_items_data_4
        }

class DataMarker:
    def __init__(self, comb_data_1_path, admissions_path, patients_path, chart_events_data_1):
        self.comb_data_1 = pd.read_csv(comb_data_1_path, compression='gzip')
        self.admission_data = pd.read_csv(admissions_path, compression='gzip')
        self.patients_data = pd.read_csv(patients_path, compression='gzip')
        self.chart_events_data_1 = chart_events_data_1
        self.comb_data_2 = None
        self.comb_data_3 = None
        self.comb_data_4 = None
        self.comb_data_5 = None
        self.comb_data_6 = None
        self.comb_data_6_sl = None
        self.data_TimeInt_chart_df = None

    def preprocess_data(self):
        self.patients_data = self.patients_data.drop(columns=['anchor_year', 'anchor_year_group'])
        self.admission_data = self.admission_data[['subject_id', 'race']]

        self.comb_data_2 = self.comb_data_1[self.comb_data_1['itemid'] == 225792]
        self.comb_data_3 = self.comb_data_2.copy()
        self.comb_data_3['ext_time'] = 1

        for i in range(1, len(self.comb_data_3)):
            if self.comb_data_3['stay_id'].iloc[i] == self.comb_data_3['stay_id'].iloc[i - 1]:
                self.comb_data_3['ext_time'].iloc[i] = self.comb_data_3['ext_time'].iloc[i - 1] + 1

        self.comb_data_4 = pd.merge(self.patients_data, self.comb_data_3, how="right", on="subject_id")

        admission_data_1 = self.admission_data[self.admission_data['subject_id'].isin(self.comb_data_4['subject_id'])]
        admission_data_1 = admission_data_1.drop_duplicates(subset=['subject_id'], keep='first')

        self.comb_data_5 = pd.merge(admission_data_1, self.comb_data_4, how="right", on="subject_id")

        self.comb_data_6 = self.comb_data_5.copy()
        names_select_cb = ['stay_id', 'race', 'gender', 'anchor_age', 'intime', 'outtime', 'dod', 'los',
                           'patientweight', 'mv_id', 'ext_fail', 'ext_time']
        self.comb_data_6_sl = self.comb_data_6[names_select_cb]
        self.comb_data_6_sl = self.comb_data_6_sl.drop_duplicates(subset=['stay_id'], keep='first')

        self.chart_events_data_1 = self.chart_events_data_1[self.chart_events_data_1['stay_id'].isin(self.comb_data_6['stay_id'])]
        self.comb_data_6 = self.comb_data_6[self.comb_data_6['stay_id'].isin(self.chart_events_data_1['stay_id'])]

    def build_time_intervals(self):
        self.chart_events_data_1['charttime'] = pd.to_datetime(self.chart_events_data_1['charttime'])
        self.chart_events_data_1 = self.chart_events_data_1.sort_values(by=['subject_id', "stay_id", 'itemid', "charttime"])

        data_TimeInt_chart = {'stay_id': [], 'start': [], 'end': [], 'dod': []}
        ad_list_chart = pd.unique(self.chart_events_data_1["stay_id"])

        for stay_id in ad_list_chart:
            sub_table = self.comb_data_6[self.comb_data_6['stay_id'] == stay_id]
            if len(sub_table) == 1:
                data_TimeInt_chart['stay_id'].append(stay_id)
                data_TimeInt_chart['start'].append(sub_table['starttime'].iloc[0])
                data_TimeInt_chart['end'].append(sub_table['endtime'].iloc[0])
                data_TimeInt_chart['dod'].append(sub_table['dod'].iloc[0])
            else:
                for j in range(len(sub_table)):
                    data_TimeInt_chart['stay_id'].append(sub_table['stay_id'].iloc[j])
                    data_TimeInt_chart['start'].append(sub_table['starttime'].iloc[j])
                    data_TimeInt_chart['end'].append(sub_table['endtime'].iloc[j])
                    data_TimeInt_chart['dod'].append(sub_table['dod'].iloc[j])

        self.data_TimeInt_chart_df = pd.DataFrame.from_dict(data_TimeInt_chart)

    def mark_deaths(self):
        data_TimeInt_chart_dod = self.comb_data_6[['stay_id', 'starttime', 'endtime', 'dod', 'outtime', 'mv_id', 'ext_fail', 'ext_time']]
        data_TimeInt_chart_dod.rename(columns={'starttime': 'start', 'endtime': 'end'}, inplace=True)
        data_TimeInt_chart_dod['dod'] = pd.to_datetime(data_TimeInt_chart_dod['dod'])
        data_TimeInt_chart_dod['TV_death_ext'] = data_TimeInt_chart_dod['dod'] - data_TimeInt_chart_dod['end']

        death_list_1 = list(data_TimeInt_chart_dod[data_TimeInt_chart_dod['TV_death_ext'] <= pd.Timedelta('0 days 00:00:00')]['stay_id'])
        death_list_2 = list(data_TimeInt_chart_dod[(data_TimeInt_chart_dod['TV_death_ext'] <= pd.Timedelta('30 days 00:00:00')) &
                                                   (data_TimeInt_chart_dod['TV_death_ext'] > pd.Timedelta('0 days 00:00:00'))]['stay_id'])

        data_TimeInt_chart_dod['ext_death_bf'] = 0
        data_TimeInt_chart_dod['ext_death_afe'] = 0

        for i in range(len(data_TimeInt_chart_dod)):
            if data_TimeInt_chart_dod['TV_death_ext'].iloc[i] <= pd.Timedelta('0 days 00:00:00'):
                data_TimeInt_chart_dod.at[i, 'ext_death_bf'] = 1
            elif (data_TimeInt_chart_dod['TV_death_ext'].iloc[i] <= pd.Timedelta('30 days 00:00:00')) and \
                    (data_TimeInt_chart_dod['TV_death_ext'].iloc[i] > pd.Timedelta('0 days 00:00:00')):
                data_TimeInt_chart_dod.at[i, 'ext_death_afe'] = 1

        self.data_TimeInt_chart_df = data_TimeInt_chart_dod.drop(['dod', 'TV_death_ext'], axis='columns').copy()

    def process_all(self):
        self.preprocess_data()
        self.build_time_intervals()
        self.mark_deaths()

    def get_processed_data(self):
        return {
            'comb_data_2': self.comb_data_2,
            'comb_data_3': self.comb_data_3,
            'comb_data_4': self.comb_data_4,
            'comb_data_5': self.comb_data_5,
            'comb_data_6': self.comb_data_6,
            'comb_data_6_sl': self.comb_data_6_sl,
            'data_TimeInt_chart_df': self.data_TimeInt_chart_df
        }
