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

class RLTraining:
    def __init__(self, cfg, state_dim, action_dim, hidden_layers, data_loader):
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers

        self.data_loader = data_loader

    def fqi_agent_config(self, seed = 1):
        agent_fqi = offline_fqi_model.FQI(self.cfg, self.state_dim, self.action_dim, self.hidden_layers)
        torch.manual_seed(seed)
        return agent_fqi

    def fqe_agent_config(self, eval_agent, eval_target, seed = 2):
        agent_fqe = offline_fqi_model.FQE(self.cfg, self.state_dim, self.action_dim, self.hidden_layers, eval_agent, eval_target)
        torch.manual_seed(seed)
        return agent_fqe

    def train(self, agent_fqi, agent_fqe_obj, agent_fqe_con_list, constraint = None):
        print('Start to train!')
        print(f'Algorithm:{self.cfg.algo}, Device:{self.cfg.device}')

        self.FQI_loss = []
        self.FQI_est_values = []

        self.FQE_loss_obj = []
        self.FQE_loss_con = {i: [] for i in range(len(agent_fqe_con_list))}

        self.FQE_est_obj_costs = []
        self.FQE_est_con_costs = {i: [] for i in range(len(agent_fqe_con_list))}

        self.lambda_dict = {i: [] for i in range(len(agent_fqe_con_list))}

        lambda_t_list = [0 for i in range(len(agent_fqe_con_list))]
        lambda_update_list = [0 for i in range(len(agent_fqe_con_list))]

        for k in range(self.cfg.train_eps):
            print(f'The number of training epochs: {k+1}')

            loss_list_fqi = []
            loss_list_fqe_obj = []
            loss_list_fqe_con = {i: [] for i in range(len(agent_fqe_con_list))}

            fqi_est_list = []
            fqe_est_obj = []
            fqe_est_con = {i: [] for i in range(len(agent_fqe_con_list))}

            for j in tqdm(range(self.cfg.train_eps_steps)):

                state_batch, action_batch, obj_cost_batch, con_cost_batch_dict, next_state_batch, done_batch = self.data_loader()

                # update the policy agent for learning agent (FQI) and evaluation agent (FQE)
                loss_rl = agent_fqi.update(lambda_t_list, state_batch, action_batch, obj_cost_batch, con_cost_batch_dict, next_state_batch, done_batch)
                loss_ev_obj = agent_fqe_obj.update(state_batch, action_batch, obj_cost_batch, next_state_batch, done_batch)
                ##############################################################################################################
                loss_list_fqi.append(loss_rl)
                loss_list_fqe_obj.append(loss_ev_obj)

                for m in range(len(agent_fqe_con_list)):
                    loss_list_fqe_con[m].append(agent_fqe_con_list[m].update(state_batch, action_batch, con_cost_batch_dict[m], next_state_batch, done_batch))

                fqi_est_value = agent_fqi.avg_Q_value_est(state_batch)
                avg_q_value_obj = agent_fqe_obj.avg_Q_value_est(state_batch)

                fqi_est_list.append(fqi_est_value)
                fqe_est_obj.append(avg_q_value_obj)

                for m in range(len(agent_fqe_con_list)):
                    fqe_est_con[m].append(agent_fqe_con_list[m].avg_Q_value_est(state_batch))
                ################# Update the dual variable: lambda ####################
                if constraint == None:
                    lambda_update_list = [0 for i in range(len(agent_fqe_con_list))]
                    lambda_t_list = [0 for i in range(len(agent_fqe_con_list))]
                else:
                    for m in range(len(agent_fqe_con_list)):
                        lambda_update_list[m] = agent_fqe_con_list[m].avg_Q_value_est(state_batch) - self.cfg.constraint_limit[m]
                        lambda_t_list[m] = lambda_t_list[m] + (self.cfg.lr_lam[m] * lambda_update_list[m])
                        lambda_t_list[m] = max(0, lambda_t_list[m])
                ######################################################################################
                if j % self.cfg.target_update == 0:

                    ### update the target agent for learning agent (FQI)
                    for target_param, policy_param in zip(agent_fqi.target_net.parameters(), agent_fqi.policy_net.parameters()):
                        target_param.data.copy_(self.cfg.tau * policy_param.data + (1 - self.cfg.tau) * target_param.data)

                    ### update the target agent for evaluation agent (FQE objective cost)
                    for target_param, policy_param in zip(agent_fqe_obj.target_net.parameters(), agent_fqe_obj.policy_net.parameters()):
                        target_param.data.copy_(self.cfg.tau * policy_param.data + (1 - self.cfg.tau) * target_param.data)

                    ### update the target agent for evaluation agent (FQE constraint cost)
                    for agent_fqe_con in agent_fqe_con_list:
                        for target_param, policy_param in zip(agent_fqe_con.target_net.parameters(), agent_fqe_con.policy_net.parameters()):
                            target_param.data.copy_(self.cfg.tau * policy_param.data + (1 - self.cfg.tau) * target_param.data)
                #########################################################################################
            print(f"Epoch {k + 1}/{self.cfg.train_eps}")
            print(f"Average FQI loss of epoch {k + 1}: {np.mean(loss_list_fqi)}")
            print(f"Average FQE estimated objective cost after epoch {k + 1}: {np.mean(fqe_est_obj)}")

            for m in range(len(agent_fqe_con_list)):
                print(f"Average FQE estimated constraint cost of constraint {m} after epoch {k + 1}: {np.mean(fqe_est_con[m])}")
                print(f"Dual variable of constraint {m} after epoch {k + 1}: {lambda_t_list[m]}")
                print(f"Dual variable update of constraint {m} after epoch {k + 1}: {lambda_update_list[m]}")

                self.lambda_dict[m].append(lambda_t_list[m])
                self.FQE_loss_con[m].append(np.mean(loss_list_fqe_con[m]))
                self.FQE_est_con_costs[m].append(np.mean(fqe_est_con[m]))

            self.FQI_loss.append(np.mean(loss_list_fqi))
            self.FQE_loss_obj.append(np.mean(loss_list_fqe_obj))

            self.FQI_est_values.append(np.mean(fqi_est_list))
            self.FQE_est_obj_costs.append(np.mean(fqe_est_obj))

        print("Complete Training!")

        return self.FQI_loss, self.FQE_loss_obj, self.FQE_loss_con, self.FQI_est_values, self.FQE_est_obj_costs, self.FQE_est_con_costs, self.lambda_dict