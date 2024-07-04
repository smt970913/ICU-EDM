# Offline Constrained Reinforcement Learning for Extubation Decision-Making

The repository contains code for an Offline Constrained Reinforcement Learning based Extubation Decison-Making support
tool utilized in the research effort of us. Code created by Maotong Sun (maotong.sun@tum.de) and Jingui Xie (jingui.xie@tum.de) 
and please **CITE** the following when you are utilizing our results:

Maotong Sun, Jingui Xie. An Offline Constrained Reinforcement Learning based Approach for Safe Extubation Decision-Making in the Intensive Care Units, 2024.

This repository contains multiple files, most of them being achieved in the `models` folder:
1. `models/safe_fqi_model.py`: The main file for the Offline Constrained Reinforcement Learning based Extubation Decision-Making support tool.
2. `models/train_set.py`: The file for the training set generation.
3. `models/test_set.py`: The file for the evaluation of the model.
4. `models/DataMIMIC_IV.py`: The file for the data preprocessing and loading. 
The data is from the MIMIC-IV dataset. 
The data is not provided in this repository, 
but can be obtained from the MIMIC-IV dataset. 
The data is preprocessed and saved in the `data` folder as `state_id_table.csv` and `rl_state_table.csv`.
5. `models/figure_plot.py`: The file for the plotting of the results.

In the following section, we provide the guideline to show the functions of `models/safe_fqi_model.py`, 
`models/train_set.py`, and `models/test_set.py`.

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
with Remaining Length-of-Stay (RLOS) from Invasive Mechanical Ventilation (IMV) initiation as the objective cost 
and Extubation Failure Rate (EFR) as the safety constraint.
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
feature vector to represent patients' states. Users can upload their own data in the same format or from other sources (e.g., 
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
4) Scale the data: 
```
def scale_data(self, df):
    return pd.DataFrame(self.scaler.fit_transform(df), columns = df.columns)
```
5) Split the data: 
```
def split_data(self):
    self.train_ext_id, self.test_ext_id = train_test_split(pd.unique(self.state_df_id_exp['ext_id']), test_size = self.test_size, random_state = self.random_state)
    self.prepare_dataframes()
    self.data_buffer_train()
    self.data_buffer_test()
```

### Step 2. Set the RL Agent
Users only need to specify several hyperparameters to construct the RL agent.
```
# Start the RL agent construction
configurator = safe_fqi_model.RLConfigurator()
# Specify the hyperparameters
config = configurator.input_rl_config()
```
If researchers lean towards studying Constrained RL models, 
customization of more hyperparameters is possible by modifying `safe_fqi_model.RLConfigurator()`, 
including the loss function (we default to using `MSE`) 
and the corresponding optimization algorithm (we default to using `Adam`). 
For medical researchers, 
the program accommodates more commonly used reinforcement learning hyperparameter settings, 
providing an interface that allows medical researchers to 
input the number of safety constraints they require and their corresponding thresholds.

### Step 3. Train the RL Agent

### Step 4. Evaluate (test) the RL Agent