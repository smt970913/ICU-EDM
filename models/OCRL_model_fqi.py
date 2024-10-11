import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import sys
import datetime
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

curr_path = str(Path().absolute())
parent_path = str(Path().absolute().parent)
sys.path.append(parent_path) # add current terminal path to sys.path
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time

# Define the approximator for FQE class
class FCN_fqe(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers = None):
        super(FCN_fqe, self).__init__()

        if hidden_layers is None:
            self.fc1 = nn.Linear(state_dim, action_dim)
            self.layers = None
        else:
            layers = []
            # add the input layer
            layers.append(nn.Linear(state_dim, hidden_layers[0]))

            # add the hidden layers
            if len(hidden_layers) > 1:
                for i in range(1, len(hidden_layers)):
                    layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

                # add the output layer
                layers.append(nn.Linear(hidden_layers[-1], action_dim))

                # store the layers in the ModuleList
                self.layers = nn.ModuleList(layers)
            else:
                # add the output layer
                layers.append(nn.Linear(hidden_layers[-1], action_dim))
                self.layers = nn.ModuleList(layers)

    def forward(self, x):
        if self.layers is None:
            x = self.fc1(x)
        else:
            # pass the input through the layers
            for layer in self.layers[:-1]:
                x = F.relu(layer(x))

            # pass the output layer
            x = self.layers[-1](x)

        return x

# Define the approximator for FQI class
class FCN_fqi(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers = None):
        super(FCN_fqi, self).__init__()

        if hidden_layers is None:
            self.fc1 = nn.Linear(state_dim, action_dim)
            self.layers = None
        else:
            # initialize the list of layers
            layers = []

            # add the input layer
            layers.append(nn.Linear(state_dim, hidden_layers[0]))

            # add the hidden layers
            if len(hidden_layers) > 1:
                for i in range(1, len(hidden_layers)):
                    layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

                # add the output layer
                layers.append(nn.Linear(hidden_layers[-1], action_dim))

                # store the layers in the ModuleList
                self.layers = nn.ModuleList(layers)
            else:
                # add the output layer
                layers.append(nn.Linear(hidden_layers[-1], action_dim))
                self.layers = nn.ModuleList(layers)

    def forward(self, x):
        if self.layers is None:
            x = self.fc1(x)
        else:
            # pass the input through the layers
            for layer in self.layers[:-1]:
                x = F.relu(layer(x))

            # pass the output layer
            x = self.layers[-1](x)

        return x

# Define the replay buffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, obj_cost, con_cost, next_state, done):

        if not isinstance(con_cost, list) and not isinstance(con_cost, tuple):
            con_cost = [con_cost]

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, obj_cost, con_cost, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, obj_cost, con_cost, next_state, done = zip(*batch)

        con_cost = [list(costs) for costs in zip(*con_cost)]

        return state, action, obj_cost, con_cost, next_state, done

    def extract(self):
        batch = self.buffer
        state, action, obj_cost, con_cost, next_state, done = zip(*batch)

        con_cost = [list(costs) for costs in zip(*con_cost)]

        return state, action, obj_cost, con_cost, next_state, done

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

class FQE:
    def __init__(self, cfg, state_dim, action_dim, hidden_layers, eval_agent, eval_target = 'obj'):

        self.device = cfg.device

        self.gamma = cfg.gamma ### discount factor
        if eval_target == 'obj':
            self.lr_fqe = cfg.lr_fqe_obj
        else:
            self.lr_fqe = cfg.lr_fqe_con[eval_target] ### For constraint cost, specify which constraint to evaluate

        # define policy Q-Estimator
        self.policy_net = FCN_fqe(state_dim, action_dim, hidden_layers).to(self.device)
        # define target Q-Estimator
        self.target_net = FCN_fqe(state_dim, action_dim, hidden_layers).to(self.device)

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer = cfg.optimizer_fqe(self.policy_net.parameters(), lr = self.lr_fqe, weight_decay = cfg.weight_decay_fqe)  # optimizer
        self.loss = cfg.loss_fqe  # loss function

        # input the evaluation agent
        self.eval_agent = eval_agent

    def update(self, state_batch, action_batch, cost_batch, next_state_batch, done_batch):

        # We need to evaluate the parameterized policy
        policy_action_batch = self.eval_agent.rl_policy(next_state_batch)

        # predicted Q-value using policy Q-network
        q_values = self.policy_net(state_batch).gather(dim = 1, index = action_batch)

        # target Q-value calculated by target Q-network
        next_q_values = self.target_net(next_state_batch).gather(dim = 1, index = policy_action_batch).squeeze(1).detach()
        expected_q_values = cost_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = self.loss(q_values, expected_q_values.unsqueeze(1))

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
        q_value = self.policy_net(state_batch).gather(dim = 1, index = policy_action_batch)

        return q_value.mean().item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'FQE_policy_network.pth')
        torch.save(self.target_net.state_dict(), path + 'FQE_target_network.pth')

class FQI:
    def __init__(self, cfg, state_dim, action_dim, hidden_layers):
        self.device = cfg.device

        self.gamma = cfg.gamma # discount factor

        self.policy_net = FCN_fqi(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net = FCN_fqi(state_dim, action_dim, hidden_layers).to(self.device)

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer = cfg.optimizer_fqi(self.policy_net.parameters(), lr = cfg.lr_fqi, weight_decay = cfg.weight_decay_fqi)
        self.loss = cfg.loss_fqi

    def update(self, lambda_t_list, state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch):

        q_values = self.policy_net(state_batch).gather(dim = 1, index = action_batch)
        policy_action_batch = self.policy_net(next_state_batch).min(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_state_batch).gather(dim = 1, index = policy_action_batch).squeeze(1).detach()

        sum_con_cost = 0
        for i in range(len(lambda_t_list)):
            lambda_t = lambda_t_list[i]
            sum_con_cost += lambda_t * con_cost_batch[i]

        expected_q_values = (obj_cost_batch + sum_con_cost) + self.gamma * next_q_values * (1 - done_batch)

        loss = self.loss(q_values, expected_q_values.unsqueeze(1))

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
        torch.save(self.policy_net.state_dict(), path + 'Offline_FQI_policy_network.pth')
        torch.save(self.target_net.state_dict(), path + 'Offline_FQI_target_network.pth')

### Comparatively convenient configuration settings for studying extubation decision-making problem
class RLConfig_default:
    def __init__(self, algo_name, train_eps, test_eps, gamma, lr_fqi, lr_fqe_obj, constraint_num, lr_fqe_con_list, lr_lambda_list, threshold_list):
        self.algo = algo_name  # name of algorithm

        self.train_eps = train_eps  #the number of trainng episodes

        self.test_eps = test_eps # the number of testing episodes

        self.gamma = gamma # discount factor

        # learning rates
        self.lr_fqi = lr_fqi
        self.lr_fqe_obj = lr_fqe_obj

        # constraint threshold
        self.constraint_num = constraint_num
        self.lr_fqe_con = [0 for i in range(constraint_num)]
        self.lr_lam = [0 for i in range(constraint_num)]
        self.constraint_limit = [0 for i in range(constraint_num)]
        for i in range(constraint_num):
            self.lr_fqe_con[i] = lr_fqe_con_list[i]
            self.lr_lam[i] = lr_lambda_list[i]
            self.constraint_limit[i] = threshold_list[i]

        ### some default settings for extubation decision-making study
        self.train_eps_steps = 3000  # the number of steps in each training episode
        self.test_eps_steps = 3000  # the number of steps in each testing episode

        self.batch_size_cont = 400
        self.batch_size_ext = 400

        self.weight_decay_fqi = 1e-5
        self.weight_decay_fqe = 1e-5

        self.optimizer_fqi = torch.optim.Adam
        self.optimizer_fqe = torch.optim.Adam

        self.loss_fqi = nn.MSELoss()
        self.loss_fqe = nn.MSELoss()

        self.memory_capacity = int(1e6)  # capacity of Replay Memory

        self.target_update = 100 # update frequency of target net
        self.tau = 0.01

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU

### Customized configuration settings for studying any medical decision-making problems
class RLConfig_custom:
    def __init__(self, algo_name, train_eps, train_eps_steps,
                 test_eps, test_eps_steps,
                 weight_decay_fqi, weight_decay_fqe,
                 optim_fqi, optim_fqe,
                 loss_fqi, loss_fqe,
                 memory_capacity, target_update, tau,
                 gamma,
                 lr_fqi, lr_fqe_obj,
                 constraint_num, lr_fqe_con_list, lr_lambda_list, threshold_list):

        self.algo = algo_name  # name of algorithm

        self.train_eps = train_eps  #the number of trainng episodes
        self.train_eps_steps = train_eps_steps  # the number of steps in each training episode
        self.test_eps = test_eps # the number of testing episodes
        self.test_eps_steps = test_eps_steps  # the number of steps in each testing episode

        self.weight_decay_fqi = weight_decay_fqi
        self.weight_decay_fqe = weight_decay_fqe

        self.optimizer_fqi = optim_fqi
        self.optimizer_fqe = optim_fqe

        self.loss_fqi = loss_fqi
        self.loss_fqe = loss_fqe
        self.memory_capacity = memory_capacity  # capacity of Replay Memory
        self.target_update = target_update  # update frequency of target net
        self.tau = tau

        self.gamma = gamma # discount factor

        # learning rates
        self.lr_fqi = lr_fqi
        self.lr_fqe_obj = lr_fqe_obj

        # constraint threshold
        self.constraint_num = constraint_num
        self.lr_fqe_con = [0 for i in range(constraint_num)]
        self.lr_lam = [0 for i in range(constraint_num)]
        self.constraint_limit = [0 for i in range(constraint_num)]

        for i in range(constraint_num):
            self.lr_fqe_con[i] = lr_fqe_con_list[i]
            self.lr_lam[i] = lr_lambda_list[i]
            self.constraint_limit[i] = threshold_list[i]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check gpu

class RLConfigurator:
    def input_rl_config(self):
        default = input("Do you want to use the default settings of OCRL for extubation decision-making? (y/n): ")

        if default == 'y':
            algo_name = input("Enter the Algorithm Name: ")
            train_eps = int(float(input("Enter the number of training episodes: ")))
            test_eps = int(float(input("Enter the number of testing episodes: ")))
            gamma = float(input(r"Enter the discount factor $\gamma$: "))
            lr_fqi = float(input("Enter the learning rate of FQI agent: "))
            lr_fqe_obj = float(input("Enter the learning rate of FQE for evaluating the objective cost: "))

            constraint_num = int(input("Enter the number of constraints in the decision-making problem: "))
            lr_fqe_con_list = []
            lr_lambda_list = []
            threshold_list = []
            for i in range(constraint_num):
                lr_fqe_con = float(input(f"Enter the learning rate for FQE Agent of the constraint {i+1}: "))
                lr_lambda = float(input(f"Enter the learning rate for dual variable $\lambda$ {i+1}: "))
                threshold = float(input(f"Enter the value of constraint threshold {i+1}: "))
                lr_fqe_con_list.append(lr_fqe_con)
                lr_lambda_list.append(lr_lambda)
                threshold_list.append(threshold)

            self.config = RLConfig_default(algo_name, train_eps, test_eps, gamma, lr_fqi, lr_fqe_obj, constraint_num, lr_fqe_con_list, lr_lambda_list, threshold_list)

        else:
            algo_name = input("Enter the Algorithm Name: ")

            train_eps = int(float(input("Enter the number of training episodes: ")))
            train_eps_steps = int(float(input("Enter the number of steps in each training episode: ")))
            test_eps = int(float(input("Enter the number of testing episodes: ")))
            test_eps_steps = int(float(input("Enter the number of steps in each testing episode: ")))

            optim_fqi = input("Enter the optimizer for FQI agent (e.g. torch.optim.Adam): ")
            optim_fqe = input("Enter the optimizer for FQE agents (e.g. torch.optim.Adam): ")

            if optim_fqi == 'torch.optim.Adam':
                weight_decay_fqi = float(input("Enter the weight decay for FQI agent: "))
            else:
                weight_decay_fqi = None

            if optim_fqe == 'torch.optim.Adam':
                weight_decay_fqe = float(input("Enter the weight decay for all FQE agents: "))
            else:
                weight_decay_fqe = None

            loss_fqi = input("Enter the loss function for FQI agent (e.g. nn.MSELoss()): ")
            loss_fqe = input("Enter the loss function for FQE agents (e.g. nn.MSELoss()): ")

            memory_capacity = int(float(input("Enter the memory capacity: ")))
            target_update = int(float(input("Enter the target update frequency: ")))
            tau = float(input("Enter the soft update parameter (tau): "))
            gamma = float(input(r"Enter the discount factor $\gamma$: "))

            lr_fqi = float(input("Enter the learning rate of FQI agent: "))
            lr_fqe_obj = float(input("Enter the learning rate of FQE agent for evaluating the objective cost: "))

            constraint_num = int(float(input("Enter the number of constraints: ")))
            lr_fqe_con_list = []
            lr_lambda_list = []
            threshold_list = []

            for i in range(constraint_num):
                lr_fqe_con = float(input(f"Enter the learning rate of FQE Agent for evaluating the constraint {i+1}: "))
                lr_lambda = float(input(f"Enter the learning rate for dual variable $\lambda$ {i+1}: "))
                threshold = float(input(f"Enter the value of constraint threshold {i+1}: "))
                lr_fqe_con_list.append(lr_fqe_con)
                lr_lambda_list.append(lr_lambda)
                threshold_list.append(threshold)

            self.config = RLConfig_custom(algo_name, train_eps, train_eps_steps,
                                          test_eps, test_eps_steps,
                                          weight_decay_fqi, weight_decay_fqe,
                                          optim_fqi, optim_fqe,
                                          loss_fqi, loss_fqe,
                                          memory_capacity, target_update, tau,
                                          gamma,
                                          lr_fqi, lr_fqe_obj,
                                          constraint_num, lr_fqe_con_list, lr_lambda_list, threshold_list)

        return self.config


