# Key functions in each file

## PFALEnv.py
- `env = PFALEnv(a,b)` creates an instance of the PFAL environment. *a* and *b* are the outdoor temperature and outdoor relative humidity respectively.
- Please comment out Line 324 when training the DRL agent to avoid keeping the outdoor weather conditions constant at each episode.

## conventional_control.py
- `ctrl = ConventionalCTRL(env)` creates an instance of the baseline strategy with predifined tuning parameters. *env* is an instance of the PFAL environment.
- `u = ctrl.act(x)` is used to determine the next action given the current state `x`.

## drl_based_control.py
- `create_policy` creates a neural network policy
- `train_policy` trains the neural network policy
- `load_policy` loads a trained policy. Use this function when GPU is available on your PC.
- `load_policy_cpu` loads a trained policy. Use this function when GPU is not available on your PC.

## baseline_test.py
- Change the outdoor conditions in the PFALEnv instance and run the file for a single growing period.

## drl_test.py
- Change the outdoor conditions in the PFALEnv instance and run the file for a single growing period.

## main simulation.py
- `data = simulate(weather_conditions_ithaca, 'drl', 'ithaca')` runs multiple month simulations based on the given mean monthly outdoor weather conditions.

