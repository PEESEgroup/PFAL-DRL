# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 03:24:46 2023

@author: decar
"""

import numpy as np

class PCTRL():
    
    def __init__(self, x_sp, bias, band, u_lb, u_ub):
        
        self.x_sp = x_sp
        self.gain = 1/band
        self.bias = bias
        self.band = band
        self.u_lb = u_lb
        self.u_ub = u_ub
        
    def act(self, x):
        
        # compute action
        e = self.x_sp - x - 0.5*self.band
        u_p = self.gain * e - self.bias
        

        # clip to range
        u = np.clip(u_p, self.u_lb, self.u_ub)
  
        
        return u
        
class ConventionalCTRL():
    
    def __init__(self, env):
        
        self.env = env
        
        self.xTD_sp = (env.temp_range[1][0] + env.temp_range[1][1])/2
        self.xTD_band = env.temp_range[1][1] - env.temp_range[1][0]
        
        self.xTN_sp = (env.temp_range[0][0] + env.temp_range[0][1])/2
        self.xTN_band = self.xTD_band
        
        self.xC_sp = (env.carbon_range[0] + env.carbon_range[1])/2 
        self.xC_band = env.carbon_range[1] - env.carbon_range[0] - 0.0003
        
        self.xH_sp = (env.humidity_range[0] + env.humidity_range[1])/2
        self.xH_band = env.humidity_range[1] - env.humidity_range[0] #  - 0.003
        
        self.xTD_u = PCTRL(self.xTD_sp, env.ushift[3], self.xTD_band, -1, 1)
        self.xTN_u = PCTRL(self.xTN_sp, env.ushift[3], self.xTN_band, -1, 1)
        self.xC_u = PCTRL(self.xC_sp, env.ushift[1], self.xC_band, -1, 1)
        self.xH_u = PCTRL(self.xH_sp, env.ushift[2], self.xH_band, -1, 1)
        
    def act(self, obs):
        
        x = self.env.unscale_states2(obs)
        
        # carbon dioxide control
        uC = self.xC_u.act(x[1])

        # humidity control
        uH = -(1/self.xH_band)*(self.xH_sp - x[3] - 0.5*self.xH_band) - 1 # self.xH_u.act(x[3])
        uH = np.clip(uH, -1, 1)
        
        # temperature control
        if x[6] == 1:
            uT = self.xTD_u.act(x[2])
            uL = 1
        else:
            uT = self.xTN_u.act(x[2])
            uL = -1
            
        # ventilation [currently 0]
        if x[1] >= self.env.carbon_range[1]:
            uV = -0.95
        elif x[1] > self.xC_sp + 4*self.xC_band:
            uV = -0.95
        else:
            uV = -0.95
        
        return np.array([uL, uC, uH, uT, uV])
            
        