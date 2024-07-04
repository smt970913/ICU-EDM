import sys
import os

import torch
import numpy as np
from tqdm import tqdm

script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

import offline_fqi_model

import warnings
warnings.filterwarnings('ignore')

class RLTesting:
    def __init__(self, cfg, state_dim, action_dim, hidden_layers, test_agent, data_loader):
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.test_agent = test_agent
        self.data_loader = data_loader

    def fqe_agent_config(self, eval_target, seed = 100):
        agent_fqe_test = offline_fqi_model.FQE(self.cfg, self.state_dim, self.action_dim, self.hidden_layers, self.test_agent, eval_target)
        torch.manual_seed(seed)
        return agent_fqe_test

    def test(self, agent_fqe_obj, agent_fqe_con_list):
        print('Start to test!')
        print(f'Algorithm:{self.cfg.algo}, Device:{self.cfg.device}')

        self.loss_eva_obj = []
        self.FQE_est_value_obj = []

        self.loss_eva_mv_obj = []
        self.FQE_est_value_mv_obj = []
        self.FQE_est_ci_lb_mv_obj = []
        self.FQE_est_ci_ub_mv_obj = []

        self.loss_eva_con = {i: [] for i in range(len(agent_fqe_con_list))}
        self.FQE_est_value_con = {i: [] for i in range(len(agent_fqe_con_list))}

        self.loss_eva_mv_con = {i: [] for i in range(len(agent_fqe_con_list))}
        self.FQE_est_value_mv_con = {i: [] for i in range(len(agent_fqe_con_list))}
        self.FQE_est_ci_lb_mv_con = {i: [] for i in range(len(agent_fqe_con_list))}
        self.FQE_est_ci_ub_mv_con = {i: [] for i in range(len(agent_fqe_con_list))}

        for k in tqdm(range(self.cfg.test_eps)):
            state_batch, action_batch, obj_cost_batch, con_cost_batch_dict, next_state_batch, done_batch = self.data_loader()

            # update the policy agent for test evaluation agent (FQE)
            loss_ev_obj = agent_fqe_obj.update(state_batch, action_batch, obj_cost_batch, next_state_batch, done_batch)
            self.loss_eva_obj.append(loss_ev_obj)

            for m in range(len(agent_fqe_con_list)):
                self.loss_eva_con[m].append(agent_fqe_con_list[m].update(state_batch, action_batch, con_cost_batch_dict[m], next_state_batch, done_batch))

            fqe_obj_est_value = agent_fqe_obj.avg_Q_value_est(state_batch)
            self.FQE_est_value_obj.append(fqe_obj_est_value)

            for m in range(len(agent_fqe_con_list)):
                self.FQE_est_value_con[m].append(agent_fqe_con_list[m].avg_Q_value_est(state_batch))
            ######################################################################################
            if k % self.cfg.target_update == 0:
                ### update the target agent for test agent (FQE objective cost)
                for target_param, policy_param in zip(agent_fqe_obj.target_net.parameters(), agent_fqe_obj.policy_net.parameters()):
                    target_param.data.copy_(self.cfg.tau * policy_param.data + (1 - self.cfg.tau) * target_param.data)
                ### update the target agent for test agent (FQE constraint cost)
                for agent_fqe_con in agent_fqe_con_list:
                    for target_param, policy_param in zip(agent_fqe_con.target_net.parameters(), agent_fqe_con.policy_net.parameters()):
                        target_param.data.copy_(self.cfg.tau * policy_param.data + (1 - self.cfg.tau) * target_param.data)

            if k % 1000 == 0:
                # print(f'Episode:{k}, Loss:{np.mean(self.loss_eva_obj)}, FQE_est_value_obj:{np.mean(self.FQE_est_value_obj)}')
                self.loss_eva_mv_obj.append(np.mean(self.loss_eva_obj))
                self.FQE_est_value_mv_obj.append(np.mean(self.FQE_est_value_obj))
                self.FQE_est_ci_lb_mv_obj.append(np.percentile(self.FQE_est_value_obj, 2.5))
                self.FQE_est_ci_ub_mv_obj.append(np.percentile(self.FQE_est_value_obj, 97.5))

                for m in range(len(agent_fqe_con_list)):
                    # print(f'Episode:{k}, Loss:{np.mean(self.loss_eva_con[m])}, FQE_est_value_con:{np.mean(self.FQE_est_value_con[m])}')
                    self.loss_eva_mv_con[m].append(np.mean(self.loss_eva_con[m]))
                    self.FQE_est_value_mv_con[m].append(np.mean(self.FQE_est_value_con[m]))
                    self.FQE_est_ci_lb_mv_con[m].append(np.percentile(self.FQE_est_value_con[m], 2.5))
                    self.FQE_est_ci_ub_mv_con[m].append(np.percentile(self.FQE_est_value_con[m], 97.5))

        print("Complete Testing!")

        return self.loss_eva_obj, self.FQE_est_value_obj, self.loss_eva_mv_obj, self.FQE_est_value_mv_obj, self.FQE_est_ci_lb_mv_obj, self.FQE_est_ci_ub_mv_obj, self.loss_eva_con, self.FQE_est_value_con, self.loss_eva_mv_con, self.FQE_est_value_mv_con, self.FQE_est_ci_lb_mv_con, self.FQE_est_ci_ub_mv_con

