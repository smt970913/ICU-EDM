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
    def __init__(self, d_items_table, pro_events_table):
        self.d_items_data = d_items_table
        self.pro_events_data = pro_events_table
        self.d_items_data_1 = None
        self.d_items_data_2 = None
        self.pro_events_data_1 = None
        self.item_id_list = None

        self.VitalSigns_id = [220045, 220048, 220179, 220050, 220180, 220051, 220052, 220181, 225312, 220210, 224690,
                              223761, 223762, 220277]
        self.GCS_score_id = [223901, 223900, 220739]
        self.Vent_para_id = [220339, 224700, 224685, 224684, 224686, 223835, 223848, 223849]
        self.Labs_id = [225624, 226536, 220602, 227464, 226534, 226537, 229761, 220653, 220546, 227466, 227467, 227457,
                        220274, 223830,
                        220228, 220235, 220224, 226062, 226063, 227456, 226540, 224828, 220635, 220545, 220615, 220621,
                        220645]
        self.General_id = [224639, 226260, 226512, 226531, 226892, 227428]
        self.ADT_id = [220003, 226228, 226545]
        self.add_id_1 = [224719, 226862, 228878, 225624, 220615, 229761, 227465, 227442, 227443,
                         225651, 225690, 226566, 227489, 226627, 220994, 227519, 227488, 225667, 228699, 228709, 228713,
                         228703, 228704, 228705, 225309, 225310, 220227, 223830, 224688, 224689]
        self.Ventilation_id = [224385, 225468, 225477, 225792, 225794, 227194]

    def select_variables(self):
        variable_list = self.VitalSigns_id + self.GCS_score_id + self.Vent_para_id + self.Labs_id + self.General_id + self.ADT_id + self.add_id_1
        self.d_items_data_1 = self.d_items_data[self.d_items_data['itemid'].isin(variable_list)]
        self.d_items_data_2 = self.d_items_data[self.d_items_data['itemid'].isin(self.Ventilation_id)]

        names_select = ['subject_id', 'stay_id', 'starttime', 'endtime', 'itemid', 'value', 'valueuom',
                        'storetime', 'ordercategoryname', 'patientweight']

        self.pro_events_data_1 = self.pro_events_data[names_select]
        self.pro_events_data_1 = self.pro_events_data_1[
            self.pro_events_data_1['itemid'].isin(self.d_items_data_2['itemid'])]

        self.item_id_list = list(self.d_items_data_2['itemid']) + list(self.d_items_data_1['itemid'])


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
            low_memory = False, blocksize = None, compression = 'gzip'
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
        self.comb_data = pd.merge(self.ICU_patient_data_test, self.pro_events_data_3, how = 'inner', on = "stay_id")
        self.comb_data = self.comb_data.drop(columns = ['subject_id_y'])

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

        # Only consider those patients without unplanned extubation (both cases)
        self.comb_data_1 = self.comb_data[self.comb_data['itemid'].isin([224385, 225792, 225794, 227194])]
        self.comb_data_1['intime'] = pd.to_datetime(self.comb_data_1['intime'])
        self.comb_data_1['outtime'] = pd.to_datetime(self.comb_data_1['outtime'])
        self.comb_data_1['starttime'] = pd.to_datetime(self.comb_data_1['starttime'])
        self.comb_data_1['endtime'] = pd.to_datetime(self.comb_data_1['endtime'])
        self.comb_data_1['TD_ICU'] = self.comb_data_1['outtime'] - self.comb_data_1['intime']
        self.comb_data_1['TD_MV'] = self.comb_data_1['endtime'] - self.comb_data_1['starttime']
        self.comb_data_1 = self.comb_data_1[~self.comb_data_1['stay_id'].isin(up_list_adm)]

        # Consider those patients with a ventilation duration less than one week (7 days)
        self.comb_data_1 = self.comb_data_1[self.comb_data_1['TD_MV'] <= pd.Timedelta('7 days 00:00:00')]
        self.comb_data_1['RLOS'] = self.comb_data_1['outtime'] - self.comb_data_1['endtime']
        self.comb_data_1['LOS_ini'] = self.comb_data_1['outtime'] - self.comb_data_1['starttime']

        # Exclude readmitted cases within 30 days after dischaging from ICU, and we also need to keep their first admission records
        self.comb_data_1 = self.comb_data_1[self.comb_data_1['itemid'].isin([225792, 225794])]
        pa_list = pd.unique(self.comb_data_1['subject_id'])
        ad_list = pd.unique(self.comb_data_1['stay_id'])

        rd_list = []  # build the readmission list

        for i in range(len(pa_list)):
            sub_data = self.comb_data_1[self.comb_data_1['subject_id'] == pa_list[i]]
            if len(pd.unique(sub_data['stay_id'])) > 1:
                rd_list.append(pa_list[i])

        self.comb_data_read = self.comb_data_1[self.comb_data_1['subject_id'].isin(rd_list)].copy()

        pa_list_2 = []
        drop_list_2 = []

        for i in range(len(rd_list)):
            sub_data = self.comb_data_read[self.comb_data_read['subject_id'] == rd_list[i]]

            for j in range(1, len(sub_data)):
                if len(pd.unique(sub_data['stay_id'])) == 2:
                    if sub_data['stay_id'].iloc[j] != sub_data['stay_id'].iloc[j - 1]:
                        if sub_data['intime'].iloc[j] - sub_data['outtime'].iloc[j - 1] <= pd.Timedelta(
                                '30 days 00:00:00'):
                            pa_list_2.append(rd_list[i])
                            drop_list_2.append(sub_data['stay_id'].iloc[j])
                            break

                else:
                    if sub_data['stay_id'].iloc[j] != sub_data['stay_id'].iloc[j - 1]:
                        if sub_data['intime'].iloc[j] - sub_data['outtime'].iloc[j - 1] <= pd.Timedelta(
                                '30 days 00:00:00'):
                            pa_list_2.append(rd_list[i])
                            drop_list_2.extend(pd.unique(sub_data['stay_id'])[1:])
                            break

        self.comb_data_1 = self.comb_data_1[~self.comb_data_1['stay_id'].isin(drop_list_2)].copy()

        self.mark_ext_fail()

        # Exclude the patients with an NIV record during their MV because the exact length of mechanical ventilation cannot be determined.
        admission_list = pd.unique(self.comb_data_1['stay_id'])
        drop_list_1 = []
        drop_list_2 = []
        drop_list_mv = []

        for i in range(len(admission_list)):
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
                                if sub_table['starttime'].iloc[k] - sub_table['endtime'].iloc[j] < pd.Timedelta(
                                        '0 days 00:00:00'):
                                    drop_list_2.append(admission_list[i])
                                    drop_list_mv.append(sub_table['mv_id'].iloc[k])

                    else:
                        for k in range(j + 1, len(sub_table)):
                            if sub_table['itemid'].iloc[k] == 225794:
                                if sub_table['starttime'].iloc[k] - sub_table['endtime'].iloc[j] < pd.Timedelta(
                                        '0 days 00:00:00'):
                                    drop_list_2.append(admission_list[i])
                                    drop_list_mv.append(sub_table['mv_id'].iloc[j])

        self.comb_data_1 = self.comb_data_1[~self.comb_data_1['stay_id'].isin(drop_list_1)].copy()

        for i in range(len(drop_list_2)):
            condition = (self.comb_data_1['stay_id'] == drop_list_2[i]) & (self.comb_data_1['mv_id'] == drop_list_mv[i])
            self.comb_data_1 = self.comb_data_1[~condition]

        # Only consider the invasive MV cases and delete the NIV cases from the table
        self.comb_data_2 = self.comb_data_1[self.comb_data_1['itemid'] == 225792].copy()

    def mark_ext_fail(self):

        self.comb_data_1 = self.comb_data_1.sort_values(by=['subject_id', 'stay_id', 'starttime', 'endtime'])
        self.comb_data_1['mv_id'] = 1
        for i in range(1, len(self.comb_data_1)):
            if self.comb_data_1['stay_id'].iloc[i] == self.comb_data_1['stay_id'].iloc[i - 1]:
                self.comb_data_1['mv_id'].iloc[i] = self.comb_data_1['mv_id'].iloc[i - 1] + 1

        admission_list = pd.unique(self.comb_data_1['stay_id'])
        ext_fail_list = []
        mv_id_record = []
        for i in range(len(admission_list)):
            sub_table = self.comb_data_1[self.comb_data_1['stay_id'] == admission_list[i]]
            if len(sub_table) >= 2:
                for j in range(len(sub_table) - 1):
                    if sub_table['itemid'].iloc[j] == 225792:  # this is an invasive MV event
                        for k in range(j + 1, len(sub_table)):
                            if sub_table['starttime'].iloc[k] - sub_table['endtime'].iloc[j] <= pd.Timedelta(
                                    '7 days 00:00:00'):
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

        self.comb_data_imv = self.comb_data_1[self.comb_data_1['itemid'] == 225792].copy()

    def process_all(self):
        self.filter_comb_data()
        self.find_NIV_list()
        self.classify_d_items_data()
        self.filter_comb_data_1()


class DataMarker:
    def __init__(self, comb_data_2, admissions_path, patients_path, chart_events_data_1):
        self.comb_data_2 = comb_data_2
        self.admission_data = pd.read_csv(admissions_path, compression='gzip')
        self.patients_data = pd.read_csv(patients_path, compression='gzip')
        self.chart_events_data_1 = chart_events_data_1
        self.comb_data_3 = None
        self.comb_data_4 = None
        self.comb_data_5 = None
        self.comb_data_6 = None
        self.comb_data_6_sl = None
        self.data_TimeInt_chart_df = None

    def preprocess_data(self):
        self.patients_data = self.patients_data.drop(columns=['anchor_year', 'anchor_year_group'])
        self.admission_data = self.admission_data[['subject_id', 'race']]

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

        self.chart_events_data_1 = self.chart_events_data_1[
            self.chart_events_data_1['stay_id'].isin(self.comb_data_6['stay_id'])]
        self.comb_data_6 = self.comb_data_6[self.comb_data_6['stay_id'].isin(self.chart_events_data_1['stay_id'])]

    def build_time_intervals(self):
        self.chart_events_data_1['charttime'] = pd.to_datetime(self.chart_events_data_1['charttime'])
        self.chart_events_data_1 = self.chart_events_data_1.sort_values(
            by=['subject_id', "stay_id", 'itemid', "charttime"])

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
        data_TimeInt_chart_dod = self.comb_data_6[
            ['stay_id', 'starttime', 'endtime', 'dod', 'outtime', 'mv_id', 'ext_fail', 'ext_time']]
        data_TimeInt_chart_dod.rename(columns={'starttime': 'start', 'endtime': 'end'}, inplace=True)
        data_TimeInt_chart_dod['dod'] = pd.to_datetime(data_TimeInt_chart_dod['dod'])
        data_TimeInt_chart_dod['TV_death_ext'] = data_TimeInt_chart_dod['dod'] - data_TimeInt_chart_dod['end']

        death_list_1 = list(
            data_TimeInt_chart_dod[data_TimeInt_chart_dod['TV_death_ext'] <= pd.Timedelta('0 days 00:00:00')][
                'stay_id'])
        death_list_2 = list(
            data_TimeInt_chart_dod[(data_TimeInt_chart_dod['TV_death_ext'] <= pd.Timedelta('30 days 00:00:00')) &
                                   (data_TimeInt_chart_dod['TV_death_ext'] > pd.Timedelta('0 days 00:00:00'))][
                'stay_id'])

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


class StateTableGen:
    def __init__(self, chart_events_data_1, d_items_data_chart, d_items_data_4, data_TimeInt_chart_df):
        self.chart_events_data_1 = chart_events_data_1
        self.d_items_data_chart = d_items_data_chart
        self.d_items_data_4 = d_items_data_4
        self.data_TimeInt_chart_df = data_TimeInt_chart_df
        self.state_1 = {'stay_id': [],
                        'time': [],
                        'mv_id': [],
                        'ext_time': [],
                        'ext_fail': [],
                        'ext_death_bfe': [],
                        'ext_death_afe': []}
        self.state_2_df = None

    def process_data(self):
        self.chart_events_data_2 = self.chart_events_data_1.copy()
        self.d_items_data_chart_text = self.d_items_data_chart[self.d_items_data_chart['param_type'] == 'Text']
        self.d_items_data_chart_num = self.d_items_data_chart[self.d_items_data_chart['param_type'] == 'Numeric']
        self.d_items_data_chart_ckbx = self.d_items_data_chart[self.d_items_data_chart['param_type'] == 'Checkbox']
        self.d_items_data_chart_numwg = self.d_items_data_chart[
            self.d_items_data_chart['param_type'] == 'Numeric with tag']
        self.d_items_data_4 = self.d_items_data_4[self.d_items_data_4['itemid'].isin([220739, 223900, 223901])]
        self.d_items_data_chart_select = self.d_items_data_chart[
            ~self.d_items_data_chart['label'].isin(['Ventilator Type', 'Ventilator Mode',
                                                    'SaO2 < 90% > 2 min', 'Gender',
                                                    'Race', 'Cardiovascular', 'Musculoskeletal',
                                                    'Neurological', 'Nutrition', 'Respiratory',
                                                    'Vascular', 'Mechanically Ventilated',
                                                    'Re-admit < 48 hours'])]
        self.chart_events_data_3 = self.chart_events_data_2[
            self.chart_events_data_2['itemid'].isin(self.d_items_data_chart_select['itemid'])]
        self.chart_events_data_4 = self.chart_events_data_3[self.chart_events_data_3['itemid'] != 220048]
        self.d_items_data_chart_select = self.d_items_data_chart_select[
            self.d_items_data_chart_select['itemid'] != 220048]

        for i in range(len(self.d_items_data_chart_select)):
            self.state_1.update({self.d_items_data_chart_select['label'].iloc[i]: []})

        self.ad_list_chart = list(self.data_TimeInt_chart_df['stay_id'])
        self.chart_events_data_4 = self.chart_events_data_4[
            self.chart_events_data_4['stay_id'].isin(self.data_TimeInt_chart_df['stay_id'])]

    def data_select(self, data, i_1, i_2, i_3):
        sub_data = data.loc[(data['charttime'] <= i_2) &
                            (data['charttime'] >= i_1) &
                            (data["itemid"] == i_3)]
        return sub_data

    def create_state_df(self):
        for i in range(len(self.ad_list_chart)):
            print("The number of processed admission records: ", i)

            # the index for catching
            index = self.data_TimeInt_chart_df["start"].iloc[i]

            s_table_id = self.chart_events_data_4[self.chart_events_data_4['stay_id'] == self.ad_list_chart[i]]

            while index <= self.data_TimeInt_chart_df["end"].iloc[i]:
                self.state_1['stay_id'].append(self.ad_list_chart[i])
                self.state_1['mv_id'].append(self.data_TimeInt_chart_df['mv_id'].iloc[i])
                self.state_1['ext_time'].append(self.data_TimeInt_chart_df['ext_time'].iloc[i])
                self.state_1['ext_fail'].append(self.data_TimeInt_chart_df['ext_fail'].iloc[i])
                self.state_1['ext_death_bfe'].append(self.data_TimeInt_chart_df['ext_death_bf'].iloc[i])
                self.state_1['ext_death_afe'].append(self.data_TimeInt_chart_df['ext_death_afe'].iloc[i])

                # we define our time interval as 6 hours, this value can be changed
                index_1 = index + pd.Timedelta('0 days 06:00:00')

                if index_1 <= self.data_TimeInt_chart_df["end"].iloc[i]:
                    self.state_1['time'].append(index_1)
                else:
                    index_1 = self.data_TimeInt_chart_df["end"].iloc[i]
                    self.state_1['time'].append(index_1)

                for j in range(len(self.d_items_data_chart_select)):
                    s_table = self.data_select(s_table_id,
                                               index,
                                               index_1,
                                               self.d_items_data_chart_select["itemid"].iloc[j])

                    n = len(s_table)

                    if n >= 1:
                        self.state_1[self.d_items_data_chart_select['label'].iloc[j]].append(s_table['valuenum'].mean())

                    else:
                        self.state_1[self.d_items_data_chart_select['label'].iloc[j]].append(np.nan)

                index = index + pd.Timedelta('0 days 06:00:00')

        self.state_1_df = pd.DataFrame.from_dict(self.state_1)
        self.state_2_df = self.state_1_df.copy()


class DataOutput:
    def __init__(self, state_df, comb_data_6, data_TimeInt_chart_df):
        self.state_df = state_df
        self.comb_data_6 = comb_data_6
        self.data_TimeInt_chart_df = data_TimeInt_chart_df

    def scale_tidal_volume(self):
        self.state_df['Tidal Volume (set)'] /= 1000
        self.state_df['Tidal Volume (observed)'] /= 1000
        self.state_df['Tidal Volume (spontaneous)'] /= 1000

    def merge_data(self):
        comb_data_7 = self.comb_data_6.drop_duplicates(subset=['stay_id'], keep='first')
        comb_data_7_stv = comb_data_7.drop(columns=['mv_id', 'ext_time', 'ext_fail', 'dod'])
        self.state_df_1 = pd.merge(self.comb_data_6, self.state_df, how="right", on=["stay_id", 'mv_id'])
        self.state_df_2 = self.state_df_1.copy()

    def calculate_RLOS(self):
        self.state_df_2['intime'] = pd.to_datetime(self.state_df_2['intime'])
        self.state_df_2['outtime'] = pd.to_datetime(self.state_df_2['outtime'])
        self.state_df_2['time'] = pd.to_datetime(self.state_df_2['time'])
        self.state_df_2['RLOS_icu'] = self.state_df_2['outtime'] - self.state_df_2['time']
        self.state_df_2['RLOS_hr'] = self.state_df_2['RLOS_icu'] / pd.Timedelta('1 hour')
        self.state_df_3 = self.state_df_2.copy()

    def process_gender(self):
        gender_dummies = pd.get_dummies(self.state_df_3.gender)
        self.state_df_3 = pd.concat([self.state_df_3, gender_dummies], axis='columns')
        self.state_df_3 = self.state_df_3.drop(['gender', 'F'], axis='columns')

    def rename_and_drop_columns(self):
        self.state_df_4 = self.state_df_3.copy()
        self.state_df_4 = self.state_df_4.drop(['ext_time_y', 'ext_fail_y'], axis='columns')
        self.state_df_4 = self.state_df_4.rename(columns={'ext_time_x': 'ext_time', 'ext_fail_x': 'ext_fail'})

    def add_extubation_flag(self):
        ad_list_chart_2 = list(self.data_TimeInt_chart_df['stay_id'])
        ad_list_ext = list(self.data_TimeInt_chart_df['ext_time'])
        self.state_df_4['EXT'] = 0

        for i in range(len(ad_list_chart_2)):
            ti = self.state_df_4[(self.state_df_4['stay_id'] == ad_list_chart_2[i]) &
                                 (self.state_df_4['ext_time'] == ad_list_ext[i])]['time'].iloc[-1]
            self.state_df_4.loc[(self.state_df_4['stay_id'] == ad_list_chart_2[i]) &
                                (self.state_df_4['ext_time'] == ad_list_ext[i]) &
                                (self.state_df_4['time'] == ti), 'EXT'] = 1

    def remove_duplicates(self):
        duplicates_subset_columns = self.state_df_4[self.state_df_4.duplicated(subset=['stay_id', 'time', 'ext_time'],
                                                                               keep=False)]
        self.state_df_4 = self.state_df_4.drop_duplicates(subset=['stay_id', 'time', 'ext_time'], keep='first').copy()
        self.state_df_4 = self.state_df_4.drop(columns=['Daily Weight', "Direct Bilirubin", 'Admission Weight (Kg)',
                                                        'Admission Weight (lbs.)', 'PA %O2 Saturation (PA Line)',
                                                        'SOFA Score', 'Urine output_ApacheIV', 'PeCO2',
                                                        'Creatinine (whole blood)'])

    def save_to_csv(self, file_name):
        self.state_df_4.to_csv(file_name, index=False)

    def process_data(self):
        self.scale_tidal_volume()
        self.merge_data()
        self.calculate_RLOS()
        self.process_gender()
        self.rename_and_drop_columns()
        self.add_extubation_flag()
        self.remove_duplicates()
        self.save_to_csv('state_df_4_0709.csv')


class StateTableProcessor:
    def __init__(self, state_df):
        self.state_table = state_df

    def assign_blood_pressure(self, row):
        if pd.isna(row['Arterial Blood Pressure systolic']) and not pd.isna(
                row['Non Invasive Blood Pressure systolic']):
            return row['Non Invasive Blood Pressure systolic']
        elif not pd.isna(row['Arterial Blood Pressure systolic']):
            return row['Arterial Blood Pressure systolic']
        elif not pd.isna(row['ART BP Systolic']):
            return row['ART BP Systolic']
        else:
            return np.nan

    def assign_blood_pressure_diastolic(self, row):
        if pd.isna(row['Arterial Blood Pressure diastolic']) and not pd.isna(
                row['Non Invasive Blood Pressure diastolic']):
            return row['Non Invasive Blood Pressure diastolic']
        elif not pd.isna(row['Arterial Blood Pressure diastolic']):
            return row['Arterial Blood Pressure diastolic']
        elif not pd.isna(row['ART BP Diastolic']):
            return row['ART BP Diastolic']
        else:
            return np.nan

    def assign_blood_pressure_mean(self, row):
        if pd.isna(row['Arterial Blood Pressure mean']) and not pd.isna(row['Non Invasive Blood Pressure mean']):
            return row['Non Invasive Blood Pressure mean']
        elif not pd.isna(row['Arterial Blood Pressure mean']):
            return row['Arterial Blood Pressure mean']
        elif not pd.isna(row['ART BP Mean']):
            return row['ART BP Mean']
        else:
            return np.nan

    def assign_temperature(self, row):
        if pd.isna(row['Temperature Celsius']) and not pd.isna(row['Temperature Fahrenheit']):
            return (row['Temperature Fahrenheit'] - 32) * 5.0 / 9.0
        elif not pd.isna(row['Temperature Celsius']):
            return row['Temperature Celsius']
        else:
            return np.nan

    def assign_SaO2(self, row):
        if pd.isna(row['Arterial O2 Saturation']) and not pd.isna(row['O2 saturation pulseoxymetry']):
            return row['O2 saturation pulseoxymetry']
        elif not pd.isna(row['Arterial O2 Saturation']):
            return row['Arterial O2 Saturation']
        else:
            return np.nan

    def assign_gcs_score(self, row):
        return row['GCS - Eye Opening'] + row['GCS - Verbal Response'] + row['GCS - Motor Response']

    def assign_peep_level(self, row):
        if pd.isna(row['PEEP set']) and not pd.isna(row['Total PEEP Level']):
            return row['Total PEEP Level']
        elif not pd.isna(row['PEEP set']):
            return row['PEEP set']
        else:
            return np.nan

    def feature_eng(self):
        self.state_table['Blood Pressure Systolic'] = self.state_table.apply(self.assign_blood_pressure, axis=1)
        self.state_table['Blood Pressure Diastolic'] = self.state_table.apply(self.assign_blood_pressure_diastolic,
                                                                              axis=1)
        self.state_table['Blood Pressure Mean'] = self.state_table.apply(self.assign_blood_pressure_mean, axis=1)
        self.state_table['Temperature C'] = self.state_table.apply(self.assign_temperature, axis=1)
        self.state_table['SaO2'] = self.state_table.apply(self.assign_SaO2, axis=1)
        self.state_table['GCS score'] = self.state_table.apply(self.assign_gcs_score, axis=1)
        self.state_table['PEEP Level'] = self.state_table.apply(self.assign_peep_level, axis=1)
        self.state_table = self.state_table.drop(columns=[
            'Arterial Blood Pressure systolic', 'Non Invasive Blood Pressure systolic', 'ART BP Systolic',
            'Arterial Blood Pressure diastolic', 'Non Invasive Blood Pressure diastolic', 'ART BP Diastolic',
            'Arterial Blood Pressure mean', 'Non Invasive Blood Pressure mean', 'ART BP Mean',
            'Temperature Celsius', 'Temperature Fahrenheit', 'Arterial O2 Saturation', 'O2 saturation pulseoxymetry',
            'GCS - Eye Opening', 'GCS - Verbal Response', 'GCS - Motor Response', 'PEEP set', 'Total PEEP Level'
        ])
        self.state_table['epoch'] = self.state_table.groupby(['stay_id', 'ext_time']).cumcount() + 1
        self.state_table['intime'] = pd.to_datetime(self.state_table['intime'])
        self.state_table['outtime'] = pd.to_datetime(self.state_table['outtime'])
        self.state_table['starttime'] = pd.to_datetime(self.state_table['starttime'])
        self.state_table['endtime'] = pd.to_datetime(self.state_table['endtime'])
        self.state_table['time'] = pd.to_datetime(self.state_table['time'])
        self.state_table['Ventilation_duration'] = (self.state_table['endtime'] - self.state_table[
            'starttime']) / pd.Timedelta('1 hour')
        self.state_table['ICU_LOS_hr'] = (self.state_table['outtime'] - self.state_table['intime']) / pd.Timedelta(
            '1 hour')
        self.state_table['RLOS_MV_start'] = (self.state_table['outtime'] - self.state_table[
            'starttime']) / pd.Timedelta('1 hour')
        self.state_table['RLOS_MV_end'] = (self.state_table['outtime'] - self.state_table['endtime']) / pd.Timedelta(
            '1 hour')
        self.state_table['RLOS_MV_epoch'] = (self.state_table['outtime'] - self.state_table['time']) / pd.Timedelta(
            '1 hour')
        self.state_table = self.state_table.drop(columns=['TD_ICU', 'TD_MV', 'RLOS', 'LOS_ini', 'RLOS_icu', 'RLOS_hr'])
        self.state_table_ini = self.state_table[self.state_table['epoch'] == 1]

    def process_state_table(self):
        # Initialize 'ext_id' column
        self.state_table['ext_id'] = 0

        # Determine where 'stay_id' changes or 'ext_time' changes for the same 'stay_id'
        stay_id_changes = self.state_table['stay_id'] != self.state_table['stay_id'].shift(fill_value = self.state_table['stay_id'][0])
        ext_time_changes = (self.state_table['ext_time'] != self.state_table['ext_time'].shift()) & (self.state_table['stay_id'] == self.state_table['stay_id'].shift())

        # Apply conditions to assign incremental IDs
        # Increment 'ext_id' where 'stay_id' changes or 'ext_time' changes for the same 'stay_id'
        self.state_table['ext_id'] = (stay_id_changes | ext_time_changes).astype(int).cumsum()
        self.state_table['ext_id'] += 1

        # Filter the initial state_table_1 where epoch equals 1
        self.state_table_ini = self.state_table[self.state_table['epoch'] == 1].copy()
        self.state_table_ini['dod'] = pd.to_datetime(self.state_table_ini['dod'])
        self.state_table_ini['ext_death_7_day'] = 0

        # Update 'ext_death_7_day' column
        for i in range(len(self.state_table_ini)):
            if not pd.isna(self.state_table_ini['dod'].iloc[i]):
                self.state_table_ini['ext_death_7_day'].iloc[i] = 1 if (self.state_table_ini['dod'].iloc[i] - self.state_table_ini['endtime'].iloc[i]).days <= 7 else 0

        self.state_table_ini['ext_discharge_6_hr'] = 0

        # Update 'ext_discharge_6_hr' column
        for i in range(len(self.state_table_ini)):
            self.state_table_ini['ext_discharge_6_hr'].iloc[i] = 1 if (self.state_table_ini['outtime'].iloc[i] - self.state_table_ini['endtime'].iloc[i]) <= pd.Timedelta('0 days 06:00:00') else 0

        # Get unique 'stay_id' values to be dropped
        drop_stay_id = self.state_table_ini[self.state_table_ini['ext_death_7_day'] == 1]['stay_id'].unique()
        drop_stay_id_1 = self.state_table_ini[self.state_table_ini['ext_discharge_6_hr'] == 1]['stay_id'].unique()
        drop_ext_id_1 = self.state_table_ini[self.state_table_ini['RLOS_MV_epoch'] <= 0.0]['ext_id'].unique()

        # Drop the specified rows from state_table_1
        self.state_table = self.state_table[~self.state_table['stay_id'].isin(drop_stay_id)]
        self.state_table = self.state_table[~self.state_table['stay_id'].isin(drop_stay_id_1)]
        self.state_table = self.state_table[~self.state_table['ext_id'].isin(drop_ext_id_1)]

    def get_processed_data(self):
        return self.state_table, self.state_table_ini
