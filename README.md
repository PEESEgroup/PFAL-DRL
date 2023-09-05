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
- Matplotlib 3.7.2

## Running the code
### Running code for single growing period (28 days)
The files `drl_test.py` and `baseline_test.py` are used to run a simulation for a single growing period (28 days) using the DRL strategy and the baseline strategy respectively. To run either of the files, simply create an instance of the PFAL environment with the mean monthly outdoor temperature and outdoor relative humidity values, that is `env = PFALEnv(23, 0.79)`, and run the respective file.

### Running code for twelve growing periods (one year)
The file `main_simulation.py` is used to run a year-long simulation for a given location and environmental regulation system. The function `simulate(a,b,c)` where *a* is a LIST containing the monthly outdoor location data, *b* is a STRING indicating either 'drl' or 'baseline', and *c* is a STRING indicating the name of the location for data storage. An example usage is `data = simulate(weather_conditions_ithaca, 'drl', 'ithaca')`.

## Citation
Please use the following citation when using the data, methods or results of this work:

Decardi-Nelson, B., You, F. Harnessing AI to boost energy savings in plant factories for sustainable food production. *Submitted to Nature Food*.
