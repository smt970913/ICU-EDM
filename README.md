# Offline Constrained Reinforcement Learning for Safe Extubation Decision-Making

The repository contains code for an Offline Constrained Reinforcement Learning based Extubation Decison-Making support
tool utilized in the research effort of us. Code created by Maotong Sun (maotong.sun@tum.de) and Jingui Xie (jingui.xie@tum.de) 
and please CITE the following when you are utilizing our results:

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
### Step 1. Construct the RL Agent

### Step 2. Train the RL Agent

### Step 3. Evaluate (test) the RL Agent