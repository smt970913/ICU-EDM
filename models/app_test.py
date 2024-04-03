from flask import Flask, request, render_template_string
import os
import sys

script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

from safe_fqi_model import RLConfig
import train_set

import DataMIMIC_IV_EFR
from sklearn.preprocessing import StandardScaler

# from celery_worker_test import celery, start_rl_training

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        algo_name = request.form['algo_name']
        train_eps = int(request.form['train_eps'])
        gamma = float(request.form['gamma'])
        lr_fqi = float(request.form['lr_fqi'])
        lr_fqe = float(request.form['lr_fqe'])
        lr_lambda = float(request.form['lr_lambda'])
        threshold = float(request.form['threshold'])

        config = RLConfig(algo_name, train_eps, gamma, lr_fqi, lr_fqe, lr_lambda, threshold)

        data = DataMIMIC_IV_EFR.DataLoader(config,
                                           state_id_path = "../data/state_id_table_0319.csv",
                                           rl_state_path = "../data/rl_state_table_0319.csv",
                                           test_size = 0.20, random_state = 1000, scaler = StandardScaler())

        rl_training = train_set.RLTraining(config, state_dim = 30, action_dim = 2, hidden_layers = None, data_loader = data.data_torch_loader_train)
        # task_id = start_rl_training.delay(rl_training)

        agent_fqi_c0 = rl_training.fqi_agent_config(seed = 1)
        agent_fqe_obj_c0 = rl_training.fqe_agent_config(agent_fqi_c0, seed = 2)
        agent_fqe_con_c0 = rl_training.fqe_agent_config(agent_fqi_c0, seed = 3)

        rl_training.train(agent_fqi_c0, agent_fqe_obj_c0, [agent_fqe_con_c0], safe_constraint = None)

        # feedback = f"Train started with Algorithm: {algo_name}, Train epochs: {train_eps}. Task ID: {task_id}"
        feedback = f"Train started with Algorithm: {algo_name}, Train epochs: {train_eps}."
        return render_template_string('<p>{{feedback}}</p><a href="/">Return</a>', feedback = feedback)

    return '''
        <form method="post">
            Algorithm Name: <input type="text" name="algo_name"><br>
            Train Epochs: <input type="number" name="train_eps"><br>
            Discount Factor (gamma): <input type="text" name="gamma"><br>
            Learning Rate for FQI: <input type="text" name="lr_fqi"><br>
            Learning Rate for FQE: <input type="text" name="lr_fqe"><br>
            Learning Rate for Lambda: <input type="text" name="lr_lambda"><br>
            Safety Constraint Threshold: <input type="text" name="threshold"><br>
            <input type="submit" value="Start Training">
        </form>
    '''


if __name__ == '__main__':
    app.run(debug = True)