# PFAL-DRL
This study uses artificial intelligence and computational modeling to analyze the dynamic complexity of plant-environment interactions across ten diverse locations worldwide, each with distinct climate conditions, and the potential to reduce resource consumption in plant factories with artificial lighting (PFALs). There is one folder in this repository:
- codes 
## Locations
Miami (Florida), Phoenix (Arizona), Los Angeles (California), Seattle (Washington), Chicago (Illinois), Milwaukee (Wisconsin), Fargo (North Dakota), Ithaca (New York), Reykjavik (Iceland), and Dubai (United Arab Emirates).

## Codes
This folder contains the Python scripts used in the study. There are six Python scripts and one folder. Details of each file or folder is provided below:

### 
- `log/`: This folder contains the trained neural network model as well as the training results
- `PFALEnv.py`: This file contains the PFAL model, reward function, etc. and follows the gymnasium custom environment conventions
- `baseline_test.py`: This file runs a single outdoor condition case using the baseline
- `conventional_control.py`: This file is Baseline module configuration
- `drl_based_control.py`: This file is the DRL module configuration and contains the training and the online inference functions
- `drl_test.py`: This file runs a single outdoor condition case using DRL
- `main_simulation.py`: This file runs year long simulation for a particular location with either baseline or DRL. The outdoor weather information are provided in this file

## Requirements
To run the codes in this repository, the following Python and core package versions must be installed:
- Python 3.10.9
- Pytorch 2.0.0
- Tianshou 0.5.0
- Gymnasium 0.28.1
- Scipy 1.10.1
- Numpy 1.23.5

## Citation
Please use the following citation when using the data, methods or results of this work:
