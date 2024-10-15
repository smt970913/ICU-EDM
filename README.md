# ICU Extubation Decision-Making 

The repository contains code for an Offline Constrained Reinforcement Learning (RL) based Extubation Decison-Making support
tool utilized in the research effort of us. Code created by Maotong Sun (maotong.sun@tum.de) and Jingui Xie (jingui.xie@tum.de) 
and please **CITE** the following when you are utilizing our results:

Maotong Sun, Jingui Xie. Personalized ICU Extubation Decisions under Resource Constraints: An Offline Constrained Reinforcement Learning Approach, 2024.

This repository contains multiple files, most of them being achieved in the `models` folder:
1. `models/OCRL_model_fqi.py`: The main file for the Offline Constrained Reinforcement Learning based Extubation Decision-Making 
support tool.
2. `models/ORL_model_cql.py`: The file for the Conservative Q-Learning (CQL) algorithm.
2. `models/train_set.py`: The file for the training set generation.
3. `models/test_set.py`: The file for the evaluation of the model.
4. `models/DataMIMIC_IV.py`: The file for modifying the data suitable for the RL training and testing. 
The data is from the MIMIC-IV dataset, and has already been preprocessed. 
The original dataset is not provided in this repository, 
but can be obtained from the MIMIC-IV dataset website. 
The data is preprocessed and saved in the `data` folder as `state_id_table.csv` and `rl_state_table.csv`.

In the following section, we provide the guideline to show the functions of `models/OCRL_model_fqi.py`, 
`models/train_set.py`, `models/test_set.py`, and `models/ORL_model_cql.py`.

## Guideline
### Step 0. Data Preprocessing
```
# Load the data
data = data_selection.load_data()
# Data preprocessing 
data = data_selection.Medical_data_preprocess(data)  
# Data imputation (e.g., using the mean imputation method)
data = data_imputation.imputation(data, method='mean')
```
Use the `data_selection.py` file to preprocess the data and complete the data selection process for you. We provide some 
common selection criterias for critical care study, and those we used in our study. Users can modify the file to fit their 
own needs.

Similarly, use the `data_imputation.py` file to impute the missing values in the data. Also, 
we provide some common imputation methods in healthcare data, and those we used in our study. 
Users can specify the imputation methods they would like to use and set necessary hyperparameters.

### Step 1. Construct the DataLoader
Before deploying RL algorithms to a medical decision-making problem, 
users need to construct a `DataLoader` following these steps. 
Here, we demonstrate using the `MIMIC-IV` dataset from our research, 
with Extubation Failure Rate (EFR) as the objective cost 
and Length-of-Stay (LOS) in ICUs from invasive mechanical ventilation (MV) initiation as the resource constraint.
```
class DataLoader:
    def __init__(self, cfg, state_id_path, rl_state_path, test_size = 0.20, random_state = 1000, scaler = StandardScaler()):
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
        self.process_con_costs()
        self.rl_state_var = self.drop_unwanted_columns(self.rl_state_var)
        self.rl_state_var_sc = self.scale_data(self.rl_state_var)

        # Splitting the data
        self.state_df_id_exp = self.state_df_id.copy()
        self.split_data()
```
The `DataLoader` class is designed to load the data, 
set objective costs and constraints,
construct the state space and action space,
scale the data,
split the data into training and testing sets, 
and store them into the respective training and testing `buffer`.
1) Load the data: Load the data from the `state_id_table.csv` and `rl_state_table.csv` files. The first table contains some 
administration information (e.g., admission_ID, LOS, etc.), and the second table contains the clinical variable to construct the 
feature vector to represent patients' states (e.g., age, weight, respiratory rate, etc.). Users can upload their own data in the same format or from other sources (e.g., 
MySQL, Google Cloud, etc.).
2) Set objective costs and constraints: 
```
def process_con_costs(self):
    # Initialize 'con_costs' to 0 by default
    self.state_df_id['con_costs'] = 0

    # Processing conditions
    condition_1 = (self.state_df_id['EXT'] == 1) & (self.state_df_id['ext_fail'] == 1)
    self.state_df_id.loc[condition_1, 'con_costs'] = 100

    condition_2 = (self.state_df_id['EXT'] == 1) & (self.state_df_id['ext_fail'] != 1)
    self.state_df_id.loc[condition_2, 'con_costs'] = 0
```
3) Construct the state space and action space: 
```
def drop_unwanted_columns(self, df):
    columns_to_drop = ['Respiratory Rate', 'Respiratory Rate (spontaneous)', 'Respiratory Rate (Total)',
                       'Tidal Volume (observed)', 'Tidal Volume (spontaneous)']
    return df.drop(columns=columns_to_drop)
```
4) Data process for model training, we only use scaling here, and other data processing methods can be added here: 
```
def scale_data(self, df):
    return pd.DataFrame(self.scaler.fit_transform(df), columns = df.columns)
```
5) Split the data into training and testing sets: 
```
def split_data(self):
    self.train_ext_id, self.test_ext_id = train_test_split(pd.unique(self.state_df_id_exp['ext_id']), test_size = self.test_size, random_state = self.random_state)
    self.prepare_dataframes()
    self.data_buffer_train()
    self.data_buffer_test()
 
def prepare_dataframes(self):
    ...
```
6) Store the training and testing sets into the respective training and testing `buffer`: 
```
def data_buffer_train(self):
    ...
def data_buffer_test(self):
    ...
```
7) Build the dataloaders for the training and testing sets suitable for the RL training and testing using PyTorch: 
```
def data_torch_loader_train(self):
    ...
def data_torch_loader_test(self):
    ...
```


### Step 2. Set the Hyperparameters for the RL Agent
Users only need to specify several hyperparameters to construct the RL agent, 
with or without constraints. 
The RL agent we use is the Fitted-Q-Iteration (FQI) algorithm modified by us, which is introduced 
in the paper "Personalized Extubation Decision-Making with Resource Constraints".
```
# Start the RL agent construction
configurator = OCRL_model_fqi.RLConfigurator()
# Specify the hyperparameters
config = configurator.input_rl_config()
```
If researchers lean towards studying Constrained RL models, 
customization of more hyperparameters is possible by modifying `OCRL_model_fqi.RLConfigurator()`, 
including the loss function (we default to using `MSE`) 
and the corresponding optimization algorithm (we default to using `SGD`). 
Otherwise, 
the program accommodates more commonly used reinforcement learning hyperparameter settings, 
providing an interface that allows researchers to 
input the number of constraints they require and their corresponding thresholds.

### Step 3. Train the RL Agent
1) Use the 'DataLoader' built in Step 1 to load the training and testing sets.
```
data = DataMIMIC_IV_EFR.DataLoader(config, 
                                   state_id_path = "../data/state_id_table.csv", 
                                   rl_state_path = "../data/rl_state_table.csv", 
                                   test_size = 0.20, random_state = 68, scaler = StandardScaler())
```
You can use 'data.state_df_id' or 'data.rl_state_var' to check the data you loaded.
```
data.state_df_id

data.rl_state_var
```
2) Train the FQI agent using the training sets.
```
rl_training = train_set.RLTraining(cfg, 
                                   state_dim = 30, action_dim = 2, 
                                   hidden_layers = None, 
                                   data_loader = data.data_torch_loader_train)

agent_fqi_c0 = rl_training.fqi_agent_config(seed = 1)
agent_fqe_obj_c0 = rl_training.fqe_agent_config(agent_fqi_c0, eval_target = 'obj', seed = 2)
agent_fqe_con_c0 = rl_training.fqe_agent_config(agent_fqi_c0, eval_target = 0, seed = 3)

rl_training.train(agent_fqi_c0, agent_fqe_obj_c0, [agent_fqe_con_c0], constraint = True)                                
```
Users should create the FQI agent and FQE agents for the training. In this case, 
two FQE agents should be created, 
one for the objective cost $c$ which refers to the EFR, 
and one for the constraint cost $g$ which is the LOS in ICU after the initiation of MV. 
The FQI agent is used to learn the policy, 
and the FQE agents are used to evaluate the learned policy 
and update the dual variables (if constraints exist).

### Step 4. Evaluate (test) the RL Agent
Users can evaluate the trained RL agent using the testing sets and the FQE agents.

### Step 5. Comparison and Visualization
After training and testing the proposed method, 
users can compare the results with other offline RL approaches, 
such as the standard FQI algorithm (without constraints)
and the Conservative Q-Learning (CQL) algorithm.

1) For the standard FQI algorithm without constraints, 
users can simply set the `constraint_num` to 0 in the `RLConfigurator` class 
during Step 2.

2) For the CQL algorithm, 
users can import the `ORL_model_cql.py` file from the `models` folder, 
and then follow similar steps as the FQI algorithm with constraints.
```
from models import ORL_model_cql

configurator = ORL_model_cql.RLConfigurator()
config = configurator.input_rl_config()

# Load the data
data = DataMIMIC_IV_EFR.DataLoader(config, 
                                   state_id_path = "../data/state_id_table.csv", 
                                   rl_state_path = "../data/rl_state_table.csv", 
                                   test_size = 0.20, random_state = 68, scaler = StandardScaler())
                                   
cql_training = train_set.RLTraining_cql(config, state_dim = 30, action_dim = 2, hidden_layers = None, data_loader = data.data_torch_loader_train)

agent_cql_0 = cql_training.cql_agent_config(seed = 1)
agent_fqe_obj_0 = cql_training.fqe_agent_config(agent_cql_0, eval_target = 'obj', seed = 2)
agent_fqe_con_0 = cql_training.fqe_agent_config(agent_cql_0, eval_target = 0, seed = 3)

agent_cql_0.train(agent_fqi_c0, agent_fqe_obj_c0, [agent_fqe_con_c0])                                
```

In the notebook `notebooks/Example_MIMIC_IV_EFR.ipynb`, 
we demonstrate the functionality by executing the above steps.
