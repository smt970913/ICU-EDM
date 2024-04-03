from celery import Celery
from flask import Flask

import DataMIMIC_IV_EFR
from sklearn.preprocessing import StandardScaler

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)
celery = make_celery(app)

@celery.task()
def start_rl_training(rl_training):
    rl_training.fqi_agent_config(seed=1)
    agent_fqe_obj_c3 = rl_training.fqe_agent_config(rl_training.agent_fqi, seed=2)
    agent_fqe_con_c3 = rl_training.fqe_agent_config(rl_training.agent_fqi, seed=3)

    data = DataMIMIC_IV_EFR.DataLoader(state_id_path="C:\\Users\\Arsen\\Downloads\\RL_Computer Science\\state_id_table_0319.csv",
                                       rl_state_path="C:\\Users\\Arsen\\Downloads\\RL_Computer Science\\rl_state_table_0319.csv",
                                       test_size=0.20, random_state=1000, scaler = StandardScaler())

    train_memory_cont, train_memory_ext = rl_training.data_buffer(data.train_state_table_cont,
                                                                  data.train_state_table_ext,
                                                                  data.rl_state_var_sc_train_cont,
                                                                  data.rl_state_var_sc_train_ext,
                                                                  data.rl_state_var_sc,
                                                                  data.train_index_list_cont,
                                                                  data.train_index_list_ext,
                                                                  data.terminal_state)

    rl_training.train(rl_training.agent_fqi, agent_fqe_obj_c3, agent_fqe_con_c3, train_memory_cont, train_memory_ext, safe_constraint = None)

    pass
