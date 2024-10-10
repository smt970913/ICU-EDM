import sys
import os

import numpy as np
import pandas as pd

import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

import OCRL_model_fqi

import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, cfg, state_id_path, rl_state_path, test_size = 0.20, random_state = 1024, scaler = StandardScaler()):
        self.cfg = cfg
        self.state_id_path = state_id_path
        self.rl_state_path = rl_state_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = scaler

        # Load datasets
        self.state_df_id = pd.read_csv(self.state_id_path)
        self.rl_state_var = pd.read_csv(self.rl_state_path)

        # Initial processing
        self.process_costs()
        self.rl_state_var = self.drop_unwanted_columns(self.rl_state_var)
        self.rl_state_var_sc = self.scale_data(self.rl_state_var)

        # Splitting the data
        self.state_df_id_exp = self.state_df_id.copy()
        self.split_data()

    def process_costs(self):

        self.state_df_id['con_costs'] = self.state_df_id['obj_costs'].copy()

        # Initialize 'con_costs' to 0 by default
        self.state_df_id['obj_costs'] = 0

        # Processing conditions
        condition_1 = (self.state_df_id['EXT'] == 1) & (self.state_df_id['ext_fail'] == 1)
        self.state_df_id.loc[condition_1, 'obj_costs'] = 100

        condition_2 = (self.state_df_id['EXT'] == 1) & (self.state_df_id['ext_fail'] != 1)
        self.state_df_id.loc[condition_2, 'obj_costs'] = 0

    def drop_unwanted_columns(self, df):
        columns_to_drop = ['Respiratory Rate', 'Respiratory Rate (spontaneous)', 'Respiratory Rate (Total)',
                           'Tidal Volume (observed)', 'Tidal Volume (spontaneous)']
        return df.drop(columns=columns_to_drop)

    def scale_data(self, df):
        df_1 = df.copy()
        feature_list = list(df.columns)

        df_1[feature_list] = self.scaler.fit_transform(df[feature_list])

        df_1['M'] = df['M'].copy()
        df_1['M'] =  df_1['M'].astype(float)

        return df_1

    def split_data(self):
        self.train_ext_id, self.test_ext_id = train_test_split(pd.unique(self.state_df_id_exp['ext_id']), test_size = self.test_size, random_state = self.random_state)
        self.prepare_dataframes()
        self.data_buffer_train()
        self.data_buffer_test()

    def prepare_dataframes(self):
        self.state_df_id_cont = self.state_df_id_exp[self.state_df_id_exp['EXT'] == 0].copy()
        self.state_df_id_ext = self.state_df_id_exp[self.state_df_id_exp['EXT'] == 1].copy()

        self.train_state_table = self.state_df_id_exp[self.state_df_id_exp['ext_id'].isin(self.train_ext_id.tolist())].copy()
        self.test_state_table = self.state_df_id_exp[self.state_df_id_exp['ext_id'].isin(self.test_ext_id.tolist())].copy()
        self.train_index_list = self.train_state_table.index.tolist()
        self.test_index_list = self.test_state_table.index.tolist()
        self.rl_state_var_sc_train = self.rl_state_var_sc.loc[self.train_index_list].copy()
        self.rl_state_var_sc_test = self.rl_state_var_sc.loc[self.test_index_list].copy()
        self.rl_state_var_train = self.rl_state_var.loc[self.train_index_list].copy()
        self.rl_state_var_test = self.rl_state_var.loc[self.test_index_list].copy()

        self.train_state_table_cont = self.state_df_id_cont[self.state_df_id_cont['ext_id'].isin(self.train_ext_id.tolist())].copy()
        self.test_state_table_cont = self.state_df_id_cont[self.state_df_id_cont['ext_id'].isin(self.test_ext_id.tolist())].copy()

        self.train_state_table_ext = self.state_df_id_ext[self.state_df_id_ext['ext_id'].isin(self.train_ext_id.tolist())].copy()
        self.test_state_table_ext = self.state_df_id_ext[self.state_df_id_ext['ext_id'].isin(self.test_ext_id.tolist())].copy()

        self.train_index_list_cont = self.train_state_table_cont.index.tolist()
        self.train_index_list_ext = self.train_state_table_ext.index.tolist()

        self.test_index_list_cont = self.test_state_table_cont.index.tolist()
        self.test_index_list_ext = self.test_state_table_ext.index.tolist()

        self.rl_state_var_sc_train_cont = self.rl_state_var_sc.loc[self.train_index_list_cont].copy()
        self.rl_state_var_sc_test_cont = self.rl_state_var_sc.loc[self.test_index_list_cont].copy()

        self.rl_state_var_sc_train_ext = self.rl_state_var_sc.loc[self.train_index_list_ext].copy()
        self.rl_state_var_sc_test_ext = self.rl_state_var_sc.loc[self.test_index_list_ext].copy()

        self.rl_state_var_train_cont = self.rl_state_var.loc[self.train_index_list_cont].copy()
        self.rl_state_var_test_cont = self.rl_state_var.loc[self.test_index_list_cont].copy()

        self.rl_state_var_train_ext = self.rl_state_var.loc[self.train_index_list_ext].copy()
        self.rl_state_var_test_ext = self.rl_state_var.loc[self.test_index_list_ext].copy()

        self.terminal_state = np.zeros(self.rl_state_var_sc.shape[1])

    def data_buffer_train(self):
        self.train_memory_cont = OCRL_model_fqi.ReplayBuffer(self.cfg.memory_capacity)
        self.train_memory_ext = OCRL_model_fqi.ReplayBuffer(self.cfg.memory_capacity)

        for i in range(len(self.train_state_table_cont)):
            state = self.rl_state_var_sc_train_cont.values[i]
            action = self.train_state_table_cont['EXT'].values[i]
            obj_cost = self.train_state_table_cont['obj_costs'].values[i]
            con_cost = self.train_state_table_cont['con_costs'].values[i]
            done = self.train_state_table_cont['EXT'].values[i]

            idx = self.train_index_list_cont[i]
            next_state = self.rl_state_var_sc.loc[idx + 1].values

            self.train_memory_cont.push(state, action, obj_cost, con_cost, next_state, done)

        for i in range(len(self.train_state_table_ext)):
            state = self.rl_state_var_sc_train_ext.values[i]

            action = self.train_state_table_ext['EXT'].values[i]
            obj_cost = self.train_state_table_ext['obj_costs'].values[i]
            con_cost = self.train_state_table_ext['con_costs'].values[i]
            done = self.train_state_table_ext['EXT'].values[i]
            next_state = self.terminal_state.copy()

            self.train_memory_ext.push(state, action, obj_cost, con_cost, next_state, done)

    def data_buffer_test(self):
        self.test_memory_cont = OCRL_model_fqi.ReplayBuffer(self.cfg.memory_capacity)
        self.test_memory_ext = OCRL_model_fqi.ReplayBuffer(self.cfg.memory_capacity)

        for i in range(len(self.test_state_table_cont)):
            state = self.rl_state_var_sc_test_cont.values[i]
            action = self.test_state_table_cont['EXT'].values[i]
            obj_cost = self.test_state_table_cont['obj_costs'].values[i]
            con_cost = self.test_state_table_cont['con_costs'].values[i]
            done = self.test_state_table_cont['EXT'].values[i]

            idx = self.test_index_list_cont[i]
            next_state = self.rl_state_var_sc.loc[idx + 1].values

            self.test_memory_cont.push(state, action, obj_cost, con_cost, next_state, done)

        for i in range(len(self.test_state_table_ext)):
            state = self.rl_state_var_sc_test_ext.values[i]

            action = self.test_state_table_ext['EXT'].values[i]
            obj_cost = self.test_state_table_ext['obj_costs'].values[i]
            con_cost = self.test_state_table_ext['con_costs'].values[i]
            done = self.test_state_table_ext['EXT'].values[i]
            next_state = self.terminal_state.copy()

            self.test_memory_ext.push(state, action, obj_cost, con_cost, next_state, done)

    def data_torch_loader_train(self):
        state_batch_1, action_batch_1, obj_cost_batch_1, con_cost_batch_1, next_state_batch_1, done_batch_1 = self.train_memory_cont.sample(self.cfg.batch_size_cont)
        state_batch_2, action_batch_2, obj_cost_batch_2, con_cost_batch_2, next_state_batch_2, done_batch_2 = self.train_memory_ext.sample(self.cfg.batch_size_ext)

        constraint_num = len(con_cost_batch_1)

        state_batch = state_batch_1 + state_batch_2
        action_batch = action_batch_1 + action_batch_2
        obj_cost_batch = obj_cost_batch_1 + obj_cost_batch_2

        con_cost_batch_dict = {i: [] for i in range(constraint_num)}
        for m in range(constraint_num):
            con_cost_batch_dict[m] = tuple(con_cost_batch_1[m]) + tuple(con_cost_batch_2[m])

        next_state_batch = next_state_batch_1 + next_state_batch_2
        done_batch = done_batch_1 + done_batch_2

        state_batch = torch.tensor(np.array(state_batch), device = self.cfg.device, dtype = torch.float)
        action_batch = torch.tensor(np.array(action_batch), device = self.cfg.device, dtype = torch.long).unsqueeze(1)
        obj_cost_batch = torch.tensor(np.array(obj_cost_batch), device = self.cfg.device, dtype = torch.float)

        for m in range(constraint_num):
            con_cost_batch_dict[m] = torch.tensor(np.array(con_cost_batch_dict[m]), device = self.cfg.device, dtype = torch.float)

        next_state_batch = torch.tensor(np.array(next_state_batch), device = self.cfg.device, dtype = torch.float)
        done_batch = torch.tensor(np.array(done_batch), device = self.cfg.device)

        return state_batch, action_batch, obj_cost_batch, con_cost_batch_dict, next_state_batch, done_batch

    def data_torch_loader_test(self):
        state_batch_1, action_batch_1, obj_cost_batch_1, con_cost_batch_1, next_state_batch_1, done_batch_1 = self.test_memory_cont.sample(self.cfg.batch_size_cont)
        state_batch_2, action_batch_2, obj_cost_batch_2, con_cost_batch_2, next_state_batch_2, done_batch_2 = self.test_memory_ext.sample(self.cfg.batch_size_ext)

        constraint_num = len(con_cost_batch_1)

        state_batch = state_batch_1 + state_batch_2
        action_batch = action_batch_1 + action_batch_2
        obj_cost_batch = obj_cost_batch_1 + obj_cost_batch_2

        con_cost_batch_dict = {i: [] for i in range(constraint_num)}
        for m in range(constraint_num):
            con_cost_batch_dict[m] = tuple(con_cost_batch_1[m]) + tuple(con_cost_batch_2[m])

        next_state_batch = next_state_batch_1 + next_state_batch_2
        done_batch = done_batch_1 + done_batch_2

        state_batch = torch.tensor(np.array(state_batch), device = self.cfg.device, dtype = torch.float)
        action_batch = torch.tensor(np.array(action_batch), device = self.cfg.device, dtype = torch.long).unsqueeze(1)
        obj_cost_batch = torch.tensor(np.array(obj_cost_batch), device = self.cfg.device, dtype = torch.float)

        for m in range(constraint_num):
            con_cost_batch_dict[m] = torch.tensor(np.array(con_cost_batch_dict[m]), device = self.cfg.device, dtype = torch.float)

        next_state_batch = torch.tensor(np.array(next_state_batch), device = self.cfg.device, dtype = torch.float)
        done_batch = torch.tensor(np.array(done_batch), device = self.cfg.device)

        return state_batch, action_batch, obj_cost_batch, con_cost_batch_dict, next_state_batch, done_batch