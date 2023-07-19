# -*- coding: utf-8 -*-

# imports
import gymnasium as gym
import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional
from copy import deepcopy

# Plant factory with artificial lighting environment class
class PFALEnv(gym.Env):
    
    def __init__(self, temp=20.15, rel=0.8):
        
        # outdoor for tests
        self.temp = temp
        self.rel = rel
        
        # model parameters
        # crop parameters
        self.c_alpha = 0.68 # [-]
        self.c_beta = 0.8 # [-]
        self.c_bnd = 0.004 # [m/s]
        self.c_car_1 = -1.32E-5 # [m/s/degree Celsius^2]
        self.c_car_2 = 5.94E-4 # [m/s/degree Celsisu]
        self.c_car_3 = -2.64E-3 # [m/s]
        self.c_eps = 17E-9 # [kg/J]
        self.c_fw = 22.5
        self.c_Gamma = 7.32E-5 # [kg/m^3]
        self.c_k = 0.9 # [-]
        self.c_lar_s = 75 # -3 # [m^2/kg]
        self.c_par = 1 # [-]
        self.c_Q10_Gamma = 2 # [-]
        self.c_Q10_gr = 1.6 # [-]
        self.c_Q10_resp = 2. # [-]
        self.c_rad_rf = 1 # [-]
        self.c_r_gr_max = 5E-6 # [1/s]
        self.c_resp_s = 3.47E-7 # [1/s]
        self.c_resp_r = 1.16E-7 # [1/s]
        self.c_stm = 0.007 # [m/s]
        self.c_tau = 0.07 # [-]. in paper it was 0.15
        self.c_d2f = 0.04 # kg DW to kg FW
        
        # climate parameters
        self.c_v_0 = 0.85 
        self.c_v_1 = 611 # [J/m^3]
        self.c_v_2 = 17.4 # [-]
        self.c_v_3 = 239. # [degrees Celsius]
        self.c_a_pl = 62.8 # [m^2/kg]
        self.c_v_pl_ai = 3.6E-3 # [m/s]
        self.mw_water = 18. # [kg/kmol]
        self.c_R = 8314. # [J/K/kmol]
        self.c_T_abs = 273. # [K]
        
        self.c_U = 0.3 # [W/m^2 C]
        self.c_Length = 12.19 # [m]
        self.c_Width = 2.44 # [m]
        self.c_Height = 2.59 # [m]
        self.c_surface_area = (self.c_Length*self.c_Width + 
                               2*(self.c_Width*self.c_Height + 
                                  self.c_Length*self.c_Height))
        self.c_volume = self.c_Length*self.c_Width*self.c_Height
        
        self.c_cap_c = self.c_Height # 4.1 # [m]
        self.c_cap_h = self.c_Height # 4.1 # [m]
        self.c_cap_q = 30000 # [J/m^2/degree Celsius]
        self.c_cap_q_p = 1000 # [J/kg/degree Celsius]
        self.c_cap_q_v = 1290 # [J/m^3/degree Celsius]
        self.c_lat_water = 2256.4 # [kJ/kg]
        
        self.c_CO2 = 0.2 # Wei-Han# 0.45 # 5.5 [$/kg] price of carbon dioxide: 
        self.c_lettuce = 4/.9 # [$/ kg of lettuce]
        self.c_elec = 0.2051 # 0.0786 # 0.2051 # [$/kWh] price of electricity: 
        self.c_led_eff = 0.52 # 
        self.c_grow_area = 80 # m^2
        self.c_COP = 3
        
        
        # equipment capacity and costs
        self.c_dehum_cap = 6/3600/self.c_grow_area # 10*0.5194/3600./self.c_grow_area # kg/m^2.s
        self.c_dehum_eev = 3.0 # kg/kW.h
        self.c_vent_fan_cap = 3000/3600/self.c_grow_area # 1.128012356/self.c_grow_area   # m^3/m^2.s
        self.fan_eff = 1000.*0.00047194745*15 # m^3/s/kW
        
        # simulation settings
        self.sampling_time = 10*60 # s
        
        # auxilliary parameters
        self.eps = 1E-100
        self.tvp = None
        
        self.i = None
        self.i_max = int(24*3600/self.sampling_time) # 1 day operation
        self.j = None
        self.j_max = int(1*24*3600/self.sampling_time) # 1 day operation for when uncertainty is updated
        self.k = None
        self.t_max = int(28*24*3600/self.sampling_time) # 28 days maximum growing 
        
        self.external_conditions = (
            (20.15, self.from_relative_humidity(20.15,0.8), self.from_ppm(400.)),  # July
            (-4.25, self.from_relative_humidity(-4.25,0.86), self.from_ppm(400.)) # January
            )
        self.env_con = None
        
        self.temp_range = ((18,20), (22,25))
        
        self.humidity_range = (self.from_relative_humidity(self.temp_range[0][0],0.7), 
                               self.from_relative_humidity(self.temp_range[1][1],0.8))
        
        self.carbon_range = (self.from_ppm(800), self.from_ppm(1200)) # ppm or umol/mol
        
        self.photo_period = (16, 8)
        
        self.uscale = np.array([
            100./2, # light intensity [W/m^2]
            0.015/3600/2, # 1.2E-6/2, # CO2 supply rate [kg/m^2/s]
            self.c_dehum_cap/2, # dehumidification rate [kg/m^2/s]
            212., # Heating/Cooling rate [W/m^2]
            self.c_vent_fan_cap/2, # ventilation rate []
            ])
        
        self.xscale = np.array([
            1., # non-structural dry weight [kg/m^2]
            1., # structural dry weight [kg/m^2]
            self.from_ppm(3000.), # air CO2 concentration [kg/m^3]
            40.0, # temperature [degrees Celsius]
            self.from_relative_humidity(40., 0.95), # absolute humidity [kg/m^3]
            40., # temperature [degrees Celsius]
            40., # temperature [degrees Celsius]
            1.0, # light on/off [-]
            40., # temperature [degrees Celsius]
            self.from_ppm(3000.), # air CO2 concentration
            self.from_relative_humidity(40., 0.98), # absolute humidity [kg/m^3]
            (24*3600) - self.sampling_time # time [s]
            ])
        
        # gym parameters
        self.observation_space = gym.spaces.Box(
            low = np.zeros(len(self.xscale)-1,dtype=np.float32), 
            high = np.ones(len(self.xscale)-1,dtype=np.float32)
            )
        
        self.action_space = gym.spaces.Box(
            low = -np.ones(len(self.uscale),dtype=np.float32), 
            high = np.ones(len(self.uscale),dtype=np.float32)
            )
        
        self.ushift = np.array([1., 1., 1., 0., 1.])
        
        # true state of the system
        self.state = None
        # observed state of the system
        self.obs_state = None
        
        # flag to use the true state or observed state
        self.uncertain = 0
    
    # step function
    def step(self, action):
        # clip inputs just in case they are outside the range
        uk = np.clip(action, self.action_space.low, self.action_space.high)
        
        # update i
        self.i += 1
        i = self.i%self.i_max
        
        # update j for uncertainty
        self.j += 1
        j = self.j%self.j_max
        
        # current state
        x = deepcopy(self.state[:5])
        
        # activate the photoperiod
        u = deepcopy(uk) # create a copy so it does not point and change the original u
        if self.state[7] == 0:
            u[0] = -1
        
        # initialize reward
        reward = 0
        
        # add external states to the input
        uw = np.concatenate((u,self.state[8:11]))
        
        # next state
        xp, dt, status = self.one_step_sim(x, uw)
        
        # check for violation of hard constraints and penalize
        # done_cons = False
        p_cons = 0
        if status != 0:
            p_cons += 10
            
        # full state
        xp = np.concatenate((xp, self.tvp[i]))
        
        # unscale states and inputs
        xx = self.unscale_states(self.state)
        xxp = self.unscale_states(xp)
        uu = self.unscale_inputs(u)
        
        
        w = xx[8:11]
        # disturbance processing
        xCo = w[1]
        xTo = w[0]
        xHo = w[2]
        
        xC = xx[2]
        xT = xx[3]
        xH = xx[4]
        
        uV = uu[4]
        
        phi_vent_c = uV*(xC - xCo) # CO2 exchange
        Q_vent_q = self.c_cap_q_v*uV*(xT - xTo)
        phi_vent_h = uV*(xH - xHo)
        
        # calculate individual control costs 
        c_CO2 = self.c_CO2 * uu[1] *dt # $/m^2
        c_light = 1E-3 * uu[0] * (dt/3600) # kWh/m^2
        c_dehum = (uu[2]/self.c_dehum_eev) * dt # kWh/m^2
        c_vent = (uu[4]/self.fan_eff) * (dt/3600) # kWh/m^2
        c_E = 0
        if uu[3] >= 0:
            c_E = 1E-3 * uu[3] * (dt/3600)/self.c_COP # kWh/m^2
        else:
            c_E = -1E-3 * uu[3] * (dt/3600)/self.c_COP # kWh/m^2
        
        # control cost
        c_electric = (c_light + c_dehum + c_vent + c_E)
        
        c_control = c_CO2 + c_electric*self.c_elec
        
        # operating zone reward and violation penalty
        p_CO2 = 0
        p_temp = 0
        p_hum = 0
        p_growth = 0
        
        if ((self.carbon_range[0] <= xx[2] <= self.carbon_range[1]) and
            (xx[5] <= xx[3] <= xx[6]) and 
            (self.humidity_range[0] <= xx[4] <= self.humidity_range[1])):
            
            # during light period
            if xx[7]:
                p_growth += ((xxp[0]+xxp[1]) - (xx[0]+xx[1]))/dt
                
        else:
            
            # CO2 deviation
            if xx[7]:
                if xx[2] < self.carbon_range[0]:
                    p_CO2 += (self.carbon_range[0] - xx[2])
                elif xx[2] > self.carbon_range[1]: 
                    p_CO2 +=  xx[2] - self.carbon_range[1]
                
            # temperature deviation
            if xx[3] < xx[5]:
                p_temp += xx[5] - xx[3]
            elif xx[3] > xx[6]:
                p_temp += xx[3] - xx[6]
                
            # humidity deviation
            if xx[7]:
                if xx[4] < self.humidity_range[0]:
                    p_hum += self.humidity_range[0] - xx[4]
                elif xx[4] > self.humidity_range[1]:
                    p_hum += xx[4] - self.humidity_range[1]
        
        # calculate reward
        reward = 0
        
        reward = -1.*c_control - (100000*p_CO2 + 10*p_temp + 10000*p_hum) - 10*p_cons + 1_000_000*p_growth
        
        # info
        info = {"light": c_light, "carbon": c_CO2, "dehum": c_dehum, 
                "E": c_E, "vent": c_vent, 
                "heat_loss":Q_vent_q, "CO2_loss": phi_vent_c, "water_loss": phi_vent_h
                }
        
        # terminal
        done_term = not (
            # xx[0] < self.DESIRED_LETTUCE_WEIGHT
            self.k < self.t_max
            )
        
        done = done_term # or done_cons
        
        # update k
        self.k += 1
        
        # update state in environment
        self.state = deepcopy(xp)
        
        # copy current crop dry weight xD
        xD = self.obs_state[0]
        # xDs = self.obs_state[1]
        
        # update observed state with current state information
        self.obs_state = np.concatenate((np.array([xp[0]+xp[1]]), xp[2:])) # deepcopy(self.state)
        
        # change crop dry weight of observed state to past value if uncertain is true
        if self.uncertain and j !=0:
            self.obs_state[0] = xD
            # self.obs_state[1] = xDs
        
        return deepcopy(self.obs_state), reward/100, done, info
    
    # reset environment
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # initial dry weight
        xDi = 0.72E-3 # [kg/m^2] # Zhang and Kacira (2020)
        
        # randomly select growing season
        env_con_lb = np.array([-12., self.from_relative_humidity(-12., 0.7), 0.000721476])
        env_con_ub = np.array([40., self.from_relative_humidity(40.,0.8), 0.000721476])
        self.env_con = self.np_random.uniform(low=env_con_lb, high=env_con_ub)
        
        # comment if training
        self.env_con = np.array([self.temp, self.from_relative_humidity(self.temp,self.rel), self.from_ppm(400.)])
        
        self.tvp = self.tv_data(self.env_con)
        
        xlb = np.array([xDi*0.25, xDi*0.75, self.from_ppm(1000), 23., self.from_relative_humidity(23.,0.75)] )
        
        xub = np.array([xDi*0.25, xDi*0.75, self.from_ppm(1000), 23., self.from_relative_humidity(23.,0.75)] )
        
        xlb = np.concatenate((xlb/self.xscale[:5], self.tvp[0]))
        xub = np.concatenate((xub/self.xscale[:5], self.tvp[0]))
        
        x = self.np_random.uniform(
            low = xlb,
            high = xub
            )
    
        self.k = 0
        self.i = 0
        self.j = 0
        
        self.state = deepcopy(x)
        self.obs_state = np.concatenate((np.array([x[0]+x[1]]), x[2:]))
        
        return deepcopy(self.obs_state)
        
    def render():
        pass
    
    # seed
    def seed(self, seed):
        np.random.seed(seed)
    
    # odinary differential equation
    def ode(self, t, xs, us):
        
        # unscale states and inputs
        x = xs*self.xscale[:5]
        u = self.unscale_inputs(us[:5])
        w = us[5:]*self.xscale[8:11]
        
        # states processing
        xDn = x[0]
        xDs = x[1]
        xC = x[2]
        xT = x[3]
        xH = x[4]
        
        # input processing
        uL = u[0]
        uC = u[1]
        uD = u[2]
        uQ = u[3]
        uV = u[4]
        
        # disturbance processing
        xCo = w[1]
        xTo = w[0]
        xHo = w[2]
        
        # internal states
        Gamma  = self.c_Gamma*self.c_Q10_Gamma**((xT - 20)/10)
        eps = self.c_eps*((xC - Gamma)/(xC + 2*Gamma))
        sigma_car = self.c_car_1*xT**2 + self.c_car_2*xT + self.c_car_3
        sigma_CO2 = 1/((1/self.c_bnd) + (1/self.c_stm) + (1/sigma_car))
        
        r_gr = self.c_r_gr_max * (xDn/(xDs + xDn)) * self.c_Q10_gr**((xT-20)/10)
        
        phi_resp = ((self.c_resp_s*(1-self.c_tau) + self.c_resp_r*self.c_tau )*
                    xDs*self.c_Q10_resp**((xT-25.)/10))
        
        phi_phot_max = ((eps*self.c_par*self.c_rad_rf*uL*self.c_led_eff*sigma_CO2*(xC - Gamma))/
                        (eps*self.c_par*self.c_rad_rf*uL*self.c_led_eff + sigma_CO2*(xC - Gamma)))
        
        phi_phot  = phi_phot_max*(1 - np.exp(-self.c_k*self.c_lar_s*(1-self.c_tau)*xDs) )
        
        phi_transp_h = (
            (1 - np.exp(-self.c_a_pl*xDs)) * (self.c_v_pl_ai) * (
                ( (self.c_v_0*self.c_v_1*self.mw_water)/(self.c_R*(xT + self.c_T_abs)) )  * 
                 ( np.exp( (self.c_v_2 * xT)/(xT + self.c_v_3) ) ) - xH
                )
            ) 
        
        phi_phot_c = phi_phot - (1/self.c_alpha)*phi_resp - ((1-self.c_beta)/(self.c_alpha*self.c_beta))*r_gr*xDs
        
        phi_vent_c = uV*(xC - xCo)
        Q_vent_q = self.c_cap_q_v*uV*(xT - xTo)
        phi_vent_h = uV*(xH - xHo)
        
        # heat from lighting
        Q_led = uL*(1-self.c_led_eff)
        
        # heat loss or gain from the walls of the plant factory
        Q_wall = self.c_U*(self.c_surface_area/self.c_grow_area)*(xT - xTo)
        
        # latent heat of vapourization
        Q_transp = self.c_lat_water*1000*phi_transp_h # [W/m^2]
        
        # odes
        xDn_dot = (self.c_alpha*phi_phot - r_gr*xDs - phi_resp - 
                   ((1 - self.c_beta)/self.c_beta)*r_gr*xDs) 
        
        xDs_dot = r_gr * xDs
        
        xC_dot = (1/self.c_volume)*(uC - phi_phot_c - phi_vent_c)*self.c_grow_area
        
        xT_dot = (1/self.c_cap_q)*(uQ - Q_vent_q + Q_led - Q_transp - Q_wall)
        
        xH_dot = (1/self.c_volume)*(phi_transp_h - phi_vent_h - uD)*self.c_grow_area
        
        x_dot = [xDn_dot, xDs_dot, xC_dot, xT_dot, xH_dot]
        
        return x_dot
    
    def ode_scaled(self, t, x, u):
        x_dot = np.array(self.ode(t,x,u))
        return x_dot/self.xscale[:5]
    
    def one_step_sim(self, x, u):
        
        e_xC_hi = lambda t, x, u: x[2] - 1 - self.eps
        e_xT_hi = lambda t, x, u: x[3] - 1 - self.eps
        e_xH_hi = lambda t, x, u: x[4] - 1 - self.eps
        
        e_xC_lo = lambda t, x, u: x[2] + self.eps 
        e_xT_lo = lambda t, x, u: x[3] + self.eps
        e_xH_lo = lambda t, x, u: x[4] + self.eps
        
        e_xC_hi.terminal = True
        e_xH_hi.terminal = True
        e_xT_hi.terminal = True
        
        e_xC_lo.terminal = True
        e_xH_lo.terminal = True
        e_xT_lo.terminal = True
    
        try:
            sol = solve_ivp(self.ode_scaled, [0, self.sampling_time], x, method='Radau', events=[e_xH_hi, e_xT_hi, e_xC_hi, e_xH_lo, e_xT_lo, e_xC_lo], args=(u,))
            
            if sol.status == 0:
                return sol.y[:,-1], sol.t[-1], 0
            else:
                return deepcopy(x), self.sampling_time, 1

        except Exception:
            return deepcopy(x), self.sampling_time, -1
    
    # time varying temperature setpoint
    def tv_data(self, s):
        ts = self.sampling_time
        ppd = self.photo_period[0]
        ppn = self.photo_period[1]
        
        iday = int(ppd*3600/ts)
        inight = int(ppn*3600/ts)
        
        x_photo = np.concatenate((
            np.ones( iday ), np.zeros( inight )
            ))/self.xscale[7]
        
        # temporarily changing them to constant upper and lower bounds
        d_lb = np.array([22, 21.33333333, 20.66666667, 20, 19.33333333, 18.66666667, 18])
        u_lb = d_lb[::-1]
        d_ub = np.array([25, 24.16666667, 23.33333333, 22.5, 21.66666667, 20.83333333, 20])
        u_ub = d_ub[::-1]
        
        # day (22) and night (18) lb 
        xT_lb = np.concatenate((
            np.ones( iday )*self.temp_range[1][0], np.ones( inight )*self.temp_range[0][0]
            ))
        xT_lb[0:4] = u_lb[3:]
        xT_lb[93:100] = d_lb[:]
        xT_lb[141:] = u_lb[:3]
        xT_lb = xT_lb/self.xscale[5]
        
        # day (25) and night (20) ub
        xT_ub = np.concatenate((
            np.ones( iday )*self.temp_range[1][1], np.ones( inight )*self.temp_range[0][1]
            ))#
        xT_ub[0:4] = u_ub[3:]
        xT_ub[93:100] = d_ub[:]
        xT_ub[141:] = u_ub[:3]
        xT_ub = xT_ub/self.xscale[5]
        
        # external conditions
        xT_o = np.ones(iday+inight)*s[0]/self.xscale[8]
        xH_o = np.ones(iday+inight)*s[1]/self.xscale[10]
        xC_o = np.ones(iday+inight)*s[2]/self.xscale[9]
        
        # time of day
        xt = np.array(range(iday+inight))*ts/self.xscale[11]
        
        return np.array([xT_lb, xT_ub, x_photo, xT_o, xC_o, xH_o, xt]).T
        
    
    # conversions
    
    def to_ppm(self, x):
        return x/(0.0409*44.1*1E-6)
    
    def from_ppm(self, x):
        return 0.0409*44.1*x*1E-6
    
    def xH_sat(self, xT):
        return (((self.c_v_1*self.mw_water)/(self.c_R*(xT + self.c_T_abs)))*
                (np.exp(self.c_v_2*xT/(xT + self.c_v_3))))
    
    def from_relative_humidity(self, xT, xH):
        return xH*self.xH_sat(xT)
    
    def to_relative_humidity(self, xT, xH):
        return xH/(self.xH_sat(xT))
    
    def scale_states(self, x):
        return x/self.xscale
    
    def unscale_states(self, x):
        return x*self.xscale
    
    def unscale_states2(self, x):
        return x*self.xscale[1:]
        
    def scale_inputs(self, u):
        return (u/self.uscale) - self.ushift
    
    def unscale_inputs(self, u):
        return (u + self.ushift)*self.uscale
        
        
        
