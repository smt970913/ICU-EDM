import sys
import os

script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

import warnings
warnings.filterwarnings('ignore')

class RLTesting:
    def __init__(self, cfg, fqe_agent, data_loader):
        self.cfg = cfg
        self.fqe_agent = fqe_agent
        self.data_loader = data_loader

    def test_eval(self):

        state_batch, action_batch, obj_cost_batch, con_cost_batch_dict, next_state_batch, done_batch = self.data_loader()

        test_eval_cost = self.fqe_agent.avg_Q_value_est(state_batch)

        return test_eval_cost