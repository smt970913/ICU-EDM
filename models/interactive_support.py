import pandas as pd
from collections import Counter

import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random
import math
import numpy as np
import scipy.stats

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import time
import datetime
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import shap

import warnings
warnings.filterwarnings('ignore')

def save_results(costs, ma_costs, tag = 'train', path = './results'):
    np.save(path + '{}_costs.npy'.format(tag), costs)
    np.save(path + '{}_ma_costs.npy'.format(tag), ma_costs)
    print('Results Successfully stored')

def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents = True, exist_ok = True)

def del_empty_dir(*paths):
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))


class FCN_fqe(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(FCN_fqe, self).__init__()
        self.fc1 = nn.Linear(state_dim, action_dim)

        # Initialize weights and biases to zero
        # torch.nn.init.zeros_(self.fc1.weight)
        # torch.nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.softplus(x) # positive output

        return x


class FQI_LR(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(FQI_LR, self).__init__()
        self.fc1 = nn.Linear(state_dim, action_dim)

        # Initialize weights and biases to zero
        # torch.nn.init.zeros_(self.fc1.weight)
        # torch.nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.softplus(x) # positive output

        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, obj_cost, con_cost, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, obj_cost, con_cost, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, obj_cost, con_cost, next_state, done = zip(*batch)

        return state, action, obj_cost, con_cost, next_state, done

    def extract(self):
        batch = self.buffer
        state, action, obj_cost, con_cost, next_state, done = zip(*batch)

        return state, action, obj_cost, con_cost, next_state, done

    def clear(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)


class FQE:
    def __init__(self, state_dim, action_dim, cfg, eval_agent):

        self.action_dim = action_dim
        self.device = cfg.device

        self.gamma = cfg.gamma
        self.lr = 1e-3

        # Policy Q-network
        self.policy_net = FCN_fqe(state_dim, action_dim).to(self.device)

        # Target Q-network
        self.target_net = FCN_fqe(state_dim, action_dim).to(self.device)

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr, weight_decay = 1e-4)
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr)
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=self.lr)

        self.eval_agent = eval_agent

    def update(self, state_batch, action_batch, cost_batch, next_state_batch, done_batch):

        # We need to evaluate the parameterized policy
        policy_action_batch = self.eval_agent.rl_policy(next_state_batch)

        # predicted Q-value using policy Q-network
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)

        # target Q-value calculated by target Q-network
        next_q_values = self.target_net(next_state_batch).gather(dim=1, index=policy_action_batch).squeeze(1).detach()
        expected_q_values = cost_batch + self.gamma * next_q_values * (1 - done_batch)

        # now we define our loss function here, we can try different loss function here
        ### MSE Loss Function
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        ### Huber Loss Function
        # loss = nn.HuberLoss()(q_values, expected_q_values.unsqueeze(1))

        # Update reward Q-network by minimizing the above loss function
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def avg_Q_value_est(self, state_batch):

        # extract the parameterized policy
        policy_action_batch = self.eval_agent.rl_policy(state_batch)

        q_value = self.policy_net(state_batch).gather(dim=1, index=policy_action_batch)

        return q_value.mean().item()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'pd_fqe_checkpoint.pth')

class FQI:
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.device = cfg.device

        self.gamma = cfg.gamma
        self.lr = 5e-3

        self.policy_net = FQI_LR(state_dim, action_dim).to(self.device)
        self.target_net = FQI_LR(state_dim, action_dim).to(self.device)

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr, weight_decay = 5e-4)
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr)
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr = self.lr)

    def update(self, lambda_t, state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch):

        ### FQI
        q_values = self.policy_net(state_batch).gather(dim = 1, index = action_batch)

        policy_action_batch = self.policy_net(next_state_batch).min(1)[1].unsqueeze(1)

        next_q_values = self.target_net(next_state_batch).gather(dim = 1, index = policy_action_batch).squeeze(1).detach()

        expected_q_values = (obj_cost_batch + lambda_t * con_cost_batch) + self.gamma * next_q_values * (1 - done_batch)

        ### MSE Loss Function
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        ### Huber Loss Function
        # loss = nn.HuberLoss()(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def avg_Q_value_est(self, state_batch):

        q_values = self.policy_net(state_batch)

        avg_q_values = q_values.min(1)[0].unsqueeze(1).mean().item()

        return avg_q_values

    def rl_policy(self, state_batch):

        q_values = self.policy_net(state_batch)

        policy_action_batch = q_values.min(1)[1].unsqueeze(1)

        return policy_action_batch

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'offline_FQI_checkpoint.pth')

curr_path = str(Path().absolute())
parent_path = str(Path().absolute().parent)
sys.path.append(parent_path) # add current terminal path to sys.path
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time

class RLConfig:
    def __init__(self):
        self.algo = "Offlline FQI with Resource Constraint"  # name of algo
        self.result_path = curr_path+"/outputs/" + \
            '/'+curr_time+'/results/'  # path to save results
        self.model_path = curr_path+"/outputs/" + \
            '/'+curr_time+'/models/'  # path to save Offline Reinforcement Learning model
        self.eval_path_con = curr_path+"/outputs/" + \
            '/'+curr_time+'/evals_constraint/'  # path to save FQE model for objective cost
        self.eval_path_obj = curr_path+"/outputs/" + \
            '/'+curr_time+'/evals_objective/'  # path to save FQE model for resource cost

        self.train_eps = int(9e4)  #the number of trainng episodes
        self.test_eps = int(5e6) # the number of testing episodes

        self.gamma = 1.00

        # self.batch_size = 128

        # learning rate for updating dual variable
        self.lr_lam = 3e-9

        # Constraint value for constraint cost function
        self.con_limit = 95

        self.memory_capacity = 1000000  # capacity of Replay Memory
        self.target_update = 100 # update frequency of target net
        self.tau = 0.01

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check gpu

def fqi_agent_config(cfg, seed = 1):

    state_dim = 30
    action_dim = 2
    agent = FQI(state_dim, action_dim, cfg)

    return agent

def fqe_agent_config(cfg, eval_agent, seed = 2):

    state_dim = 30
    action_dim = 2
    agent = FQE(state_dim, action_dim, cfg, eval_agent)

    return agent


def train(cfg, agent_fqi, agent_fqe_obj, agent_fqe_con):
    print('Start to train !')
    print(f'Algorithm:{cfg.algo}, Device:{cfg.device}')

    FQI_loss = []
    FQI_est_values = []

    FQE_loss_obj = []
    FQE_loss_con = []

    est_obj_costs = []
    est_con_costs = []

    lambda_list = []

    train_memory_cont = ReplayBuffer(cfg.memory_capacity)
    train_memory_ext = ReplayBuffer(cfg.memory_capacity)

    # train_memory_suc = ReplayBuffer(cfg.memory_capacity)
    # train_memory_fail = ReplayBuffer(cfg.memory_capacity)
    ######################################## Store the dataset into the buffer ###########################################
    for i in range(len(train_state_table_cont)):
        state = rl_state_var_sc_train_cont.values[i]
        action = train_state_table_cont['EXT'].values[i]
        obj_cost = train_state_table_cont['obj_costs'].values[i]
        con_cost = train_state_table_cont['con_costs'].values[i]
        done = train_state_table_cont['EXT'].values[i]

        idx = train_index_list_cont[i]
        next_state = rl_state_var_sc.loc[idx + 1].values

        train_memory_cont.push(state, action, obj_cost, con_cost, next_state, done)

    for i in range(len(train_state_table_ext)):
        state = rl_state_var_sc_train_ext.values[i]

        action = train_state_table_ext['EXT'].values[i]
        obj_cost = train_state_table_ext['obj_costs'].values[i]
        con_cost = train_state_table_ext['con_costs'].values[i]
        done = train_state_table_ext['EXT'].values[i]

        next_state = terminal_state.copy()

        train_memory_ext.push(state, action, obj_cost, con_cost, next_state, done)
    ####################################################################################################################
    lambda_t = 0.0

    for k in range(cfg.train_eps):
        print('The number of training epochs: ', k)

        loss_list_fqi = []
        loss_list_fqe_obj = []
        loss_list_fqe_con = []

        fqi_est_list = []
        fqe_est_obj = []
        fqe_est_con = []

        # learn the prameterized policy
        # for j in tqdm(range(100)):
        for j in range(100):

            state_batch_1, action_batch_1, \
                obj_cost_batch_1, con_cost_batch_1, \
                next_state_batch_1, done_batch_1 = train_memory_cont.sample(300)

            state_batch_2, action_batch_2, \
                obj_cost_batch_2, con_cost_batch_2, \
                next_state_batch_2, done_batch_2 = train_memory_ext.sample(300)

            state_batch = state_batch_1 + state_batch_2
            action_batch = action_batch_1 + action_batch_2
            obj_cost_batch = obj_cost_batch_1 + obj_cost_batch_2
            con_cost_batch = con_cost_batch_1 + con_cost_batch_2
            next_state_batch = next_state_batch_1 + next_state_batch_2
            done_batch = done_batch_1 + done_batch_2

            state_batch = torch.tensor(np.array(state_batch), device=cfg.device, dtype=torch.float)
            action_batch = torch.tensor(np.array(action_batch), device=cfg.device, dtype=torch.long).unsqueeze(1)
            obj_cost_batch = torch.tensor(np.array(obj_cost_batch), device=cfg.device, dtype=torch.float)
            con_cost_batch = torch.tensor(np.array(con_cost_batch), device=cfg.device, dtype=torch.float)
            next_state_batch = torch.tensor(np.array(next_state_batch), device=cfg.device, dtype=torch.float)
            done_batch = torch.tensor(np.array(done_batch), device=cfg.device)

            ### update the policy NN for learning agent and evaluation agent
            # print(lambda_t)
            loss_rl = agent_fqi.update(lambda_t,
                                       state_batch, action_batch,
                                       obj_cost_batch, con_cost_batch,
                                       next_state_batch, done_batch)

            loss_ev_obj = agent_fqe_obj.update(state_batch, action_batch, obj_cost_batch, next_state_batch, done_batch)
            loss_ev_con = agent_fqe_con.update(state_batch, action_batch, con_cost_batch, next_state_batch, done_batch)
            ##############################################################################################################
            loss_list_fqi.append(loss_rl)
            loss_list_fqe_obj.append(loss_ev_obj)
            loss_list_fqe_con.append(loss_ev_con)

            fqi_est_value = agent_fqi.avg_Q_value_est(state_batch)
            avg_q_value_obj = agent_fqe_obj.avg_Q_value_est(state_batch)
            avg_q_value_con = agent_fqe_con.avg_Q_value_est(state_batch)

            fqi_est_list.append(fqi_est_value)
            fqe_est_obj.append(avg_q_value_obj)
            fqe_est_con.append(avg_q_value_con)
            ################# Update the dual variable: lambda ####################
            lam_update = avg_q_value_con - cfg.con_limit
            # lam_update = 0

            lambda_t = lambda_t + (cfg.lr_lam * lam_update)
            lambda_t = max(0, lambda_t)
            ######################################################################################
            if (j + 1) % cfg.target_update == 0:
                # print(j)
                ### update the target NN for learning agent (FQI)
                for target_param, policy_param in zip(agent_fqi.target_net.parameters(),
                                                      agent_fqi.policy_net.parameters()):
                    target_param.data.copy_(cfg.tau * policy_param.data + (1 - cfg.tau) * target_param.data)

                ### update the target NN for evaluation agent (FQE objective cost)
                for target_param, policy_param in zip(agent_fqe_obj.target_net.parameters(),
                                                      agent_fqe_obj.policy_net.parameters()):
                    target_param.data.copy_(cfg.tau * policy_param.data + (1 - cfg.tau) * target_param.data)

                ### update the target NN for evaluation agent (FQE constraint cost)
                for target_param, policy_param in zip(agent_fqe_con.target_net.parameters(),
                                                      agent_fqe_con.policy_net.parameters()):
                    target_param.data.copy_(cfg.tau * policy_param.data + (1 - cfg.tau) * target_param.data)
            #########################################################################################

        lambda_list.append(lambda_t)

        #         print("lambda after update of One Epoch: ", lambda_t)
        #         print("lambda update: ", lam_update)

        #         print("FQE estimated objective cost: ", np.mean(fqe_est_obj))
        #         print("FQE estimated constraint cost: ", np.mean(fqe_est_con))

        FQI_loss.append(np.mean(loss_list_fqi))
        FQE_loss_obj.append(np.mean(loss_list_fqe_obj))
        FQE_loss_con.append(np.mean(loss_list_fqe_con))

        FQI_est_values.append(np.mean(fqi_est_list))
        est_obj_costs.append(np.mean(fqe_est_obj))
        est_con_costs.append(np.mean(fqe_est_con))

    print('Complete training！')

    return FQI_loss, FQE_loss_obj, FQE_loss_con, FQI_est_values, est_obj_costs, est_con_costs, lambda_list