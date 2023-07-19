# PFAL-DRL
This repository contains code samples of the paper "Harnessing AI to Boost Energy Savings in Plant Factories for Sustainable Food Production"

# Overview
- `log/`: Contains the trained DRL agent
- `PFALEnv.py`: PFAL gym environment
- `baseline_test.py`: Test file for a single case run using baseline
- `conventional_control.py`: Baseline module configuration
- `drl_based_control.py`: DRL module configuration and functions
- `drl_test.py`: Test file for a single case run using DRL
- `main_simulation.py`: Runs year long simulation for a particular location with either baseline or DRL

# Requirements
Codes has been tested using **Python 3.10.9** and the following core package versions
- pytorch 2.0.0+cu118
- tianshou 0.5.0
- gymnasium 0.28.1
- scipy 1.10.1
- numpy 1.23.5
