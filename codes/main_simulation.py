# -*- coding: utf-8 -*-

# imports
import numpy as np
from PFALEnv import PFALEnv
from matplotlib import pyplot as plt
from conventional_control import ConventionalCTRL
from tianshou.data import Batch
from copy import deepcopy
from drl_based_control import *


# overall simulation
def simulate(weather_conditions, strategy, loc):
    
    # load controller
    exp = 1
    n_net = 2
    net = [256]*n_net
    gamma = 0.99
    actor_lr = 3E-4
    critic_lr = 3E-4
    alpha_lr = 3E-4
    
    if strategy == "drl":
    # create policy
        policy = create_policy(net, gamma, actor_lr, critic_lr, alpha_lr)
        policy = load_policy_cpu(policy, exp)
        policy.eval()
    
    # else:
        
    
    full_data = []
    
    for wc in weather_conditions:
        print(wc[2])
        
        # create PFAL environment
        env = PFALEnv(wc[0], wc[1])
        month = strategy + wc[2] + loc
        
        # create conventional control
        if strategy == "baseline":
            ctrl = ConventionalCTRL(env)
        
        
        # simulation settings
        dt = env.sampling_time
        
        # initial state
        x = env.reset()
        
        # data storage
        t = []
        X = [x]
        U = []
        R = []
        
        # costs storage
        C = []
        
        # run simulation
        done = False
        k = 0
        info = {}
        
        while not done:
            
            # record current time
            # print(k)
            t += [k*dt]
            
            # compute control action
            
            # compute control action
            if strategy == "drl":
                u = policy(Batch(obs=[x], info=info)).act[0].cpu().detach().numpy()
            else:
                u = ctrl.act(x)
            
            # clipping if outside the range
            u = np.clip(u, env.action_space.low, env.action_space.high)
            
            # u = np.clip(u, env.action_space.low, env.action_space.high)
            # uu = deepcopy(u)
            # u[1] = 1
            if x[6] == 0:
                u[0] = - 1
                
            # record input
            U += [u]
            
            # simulate the next time step
            x, r, done, info = env.step(u)
            
            # store state data
            X += [x]
            
            # record reward data
            R += [r]
            
            # record cost
            C += [ [ info["light"], info["carbon"], info["dehum"], 
                    info["E"], info["vent"] , info["heat_loss"],
                  info["CO2_loss"], info["water_loss"] ]
                  ]
            
            k += 1
            
        # prepare data
        t = np.array(t)/(24*60*60)
        X = env.unscale_states2(np.array(X))
        U = env.unscale_inputs(np.array(U))
        R = np.array(R)
        C = np.array(C)
        
        data = np.array( [ sum(U[:,0]), sum(U[:,1]), sum(U[:,2]), -sum(U[U[:,3]<0,3])+ 
                          sum(U[U[:,3]>0,3]), sum(U[:,4]),  X[-1,0], sum(C[C[:,6]>0,6]),  
                        -sum(C[C[:,6]<0,6]), sum(C[C[:,5]>0,5]),  
                      -sum(C[C[:,5]<0,5]), sum(C[C[:,7]>0,7]), -sum(C[C[:,7]<0,7])
                        ])
        
        # data = data[:,None].T
        
        full_data += [data]
        
        # save files
        np.savez(month, t=t, X=X, U=U,R=R, C=C)
        
    return np.array(full_data)

# run simulation
weather_conditions_reykjavik = [
    (1,	0.79, '_january_'), 
    (0,	0.77, '_february_'), 
    (1,	0.77, '_march_'), 
    (3,	0.74, '_april_'), 
    (7,	0.74, '_may_'), 
    (10,	0.77, '_june_'), 
    (12,	0.80, '_july_'), 
    (11,	0.81, '_august_'), 
    (9,	0.80, '_september_'), 
    (5,	0.78, '_october_'), 
    (2,	0.79, '_november_'), 
    (1,	0.79, '_december_')
    ]

weather_conditions_ithaca = [
    (-5,	0.76, '_january_'), 
    (-4,	0.71, '_february_'), 
    (0,	0.67, '_march_'), 
    (7,	0.63, '_april_'), 
    (14,	0.66, '_may_'), 
    (18,	0.71, '_june_'), 
    (21,	0.71, '_july_'), 
    (20,	0.73, '_august_'), 
    (16,	0.76, '_september_'), 
    (9,	0.73, '_october_'), 
    (4,	0.73, '_november_'), 
    (-2,	0.77, '_december_')
    ]

weather_conditions_dubai = [
    (20,	0.63, '_january_'), 
    (21,	0.62, '_february_'), 
    (23,	0.59, '_march_'), 
    (28,	0.51, '_april_'), 
    (32,	0.46, '_may_'), 
    (34,	0.53, '_june_'), 
    (36,	0.52, '_july_'), 
    (36,	0.51, '_august_'), 
    (34,	0.57, '_september_'), 
    (30,	0.58, '_october_'), 
    (26,	0.59, '_november_'), 
    (22,	0.63, '_december_')
    ]

weather_conditions_austin = [
    (11,	0.63, '_january_'), 
    (13,	0.62, '_february_'), 
    (17,	0.62, '_march_'), 
    (21,	0.63, '_april_'), 
    (25,	0.67, '_may_'), 
    (28,	0.65, '_june_'), 
    (30,	0.61, '_july_'), 
    (30,	0.58, '_august_'), 
    (27,	0.61, '_september_'), 
    (22,	0.63, '_october_'), 
    (16,	0.65, '_november_'), 
    (12,	0.66, '_december_')
    ]

weather_conditions_phoenix = [
    (14,	0.42, '_january_'), 
    (16,	0.40, '_february_'), 
    (19,	0.32, '_march_'), 
    (23,	0.23, '_april_'), 
    (28,	0.18, '_may_'), 
    (33,	0.16, '_june_'), 
    (35,	0.28, '_july_'), 
    (34,	0.32, '_august_'), 
    (32,	0.29, '_september_'), 
    (25,	0.29, '_october_'), 
    (18,	0.34, '_november_'), 
    (13,	0.44, '_december_')
    ]

weather_conditions_miami = [
    (21,	0.71, '_january_'), 
    (22,	0.70, '_february_'), 
    (23,	0.67, '_march_'), 
    (25,	0.66, '_april_'), 
    (27,	0.67, '_may_'), 
    (28,	0.73, '_june_'), 
    (29,	0.72, '_july_'), 
    (29,	0.73, '_august_'), 
    (28,	0.75, '_september_'), 
    (27,	0.72, '_october_'), 
    (24,	0.71, '_november_'), 
    (22,	0.72, '_december_')
    ]

weather_conditions_la = [
    (15,	0.59, '_january_'), 
    (15,	0.64, '_february_'), 
    (16,	0.67, '_march_'), 
    (18,	0.66, '_april_'), 
    (19,	0.69, '_may_'), 
    (21,	0.70, '_june_'), 
    (23,	0.71, '_july_'), 
    (24,	0.69, '_august_'), 
    (23,	0.68, '_september_'), 
    (21,	0.66, '_october_'), 
    (17,	0.57, '_november_'), 
    (14,	0.56, '_december_')
    ]

weather_conditions_seattle = [
    (6,	0.81, '_january_'), 
    (7,	0.76, '_february_'), 
    (9,	0.73, '_march_'), 
    (11,	0.69, '_april_'), 
    (14,	0.66, '_may_'), 
    (17,	0.64, '_june_'), 
    (20,	0.61, '_july_'), 
    (20,	0.62, '_august_'), 
    (17,	0.69, '_september_'), 
    (12,	0.78, '_october_'), 
    (8,	0.81, '_november_'), 
    (6,	0.82, '_december_')
    ]

weather_conditions_chicago = [
    (-3,	0.74, '_january_'), 
    (-2,	0.70, '_february_'), 
    (4,	0.65, '_march_'), 
    (10,	0.60, '_april_'), 
    (16,	0.62, '_may_'), 
    (22,	0.63, '_june_'), 
    (25,	0.63, '_july_'), 
    (24,	0.66, '_august_'), 
    (19,	0.64, '_september_'), 
    (12,	0.64, '_october_'), 
    (6,	0.68, '_november_'), 
    (-1,	0.74, '_december_')
    ]

weather_conditions_milwaukee = [
    (-5,	0.76, '_january_'), 
    (-3,	0.73, '_february_'), 
    (2,	0.70, '_march_'), 
    (8,	0.66, '_april_'), 
    (14,	0.66, '_may_'), 
    (20,	0.68, '_june_'), 
    (23,	0.70, '_july_'), 
    (22,	0.74, '_august_'), 
    (18,	0.73, '_september_'), 
    (11,	0.71, '_october_'), 
    (4,	0.74, '_november_'), 
    (-2,	0.78, '_december_')
    ]

weather_conditions_fargo = [
    (-12,	0.79, '_january_'), 
    (-10,	0.78, '_february_'), 
    (-2,	0.76, '_march_'), 
    (7,	0.61, '_april_'), 
    (14,	0.59, '_may_'), 
    (19,	0.65, '_june_'), 
    (22,	0.68, '_july_'), 
    (21,	0.67, '_august_'), 
    (15,	0.66, '_september_'), 
    (8,	0.66, '_october_'), 
    (-1,	0.75, '_november_'), 
    (-9,	0.80, '_december_')
    ]



data = simulate(weather_conditions_ithaca, 'drl', 'ithaca')
