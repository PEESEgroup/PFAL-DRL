# -*- coding: utf-8 -*-

# imports
import numpy as np
from PFALEnv import PFALEnv
from matplotlib import pyplot as plt
from conventional_control import ConventionalCTRL
from tianshou.data import Batch
from copy import deepcopy
from drl_based_control import *

# create PFAL environment
env = PFALEnv(23, 0.79)
month = 'drl_february_reykjavik'
# create conventional control
ctrl = ConventionalCTRL(env)

# simulation settings
dt = env.sampling_time

exp = 1
n_net = 2
net = [256]*n_net
gamma = 0.99
actor_lr = 3E-4
critic_lr = 3E-4
alpha_lr = 3E-4
nstep_per_episode = 3025
batch_size = 256
epoch = 200

# create policy
policy = create_policy(net, gamma, actor_lr, critic_lr, alpha_lr)
policy = load_policy_cpu(policy, exp)
policy.eval()

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
    t += [k*dt]
    
    
    # compute control action
    u = policy(Batch(obs=[x], info=info)).act[0].cpu().detach().numpy()
    
    # clipping if outside the range
    u = np.clip(u, env.action_space.low, env.action_space.high)
    

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

data = data[:,None].T

# save files
# np.savez(month, t=t, X=X, U=U,R=R, C=C)