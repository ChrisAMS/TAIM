import numpy as np
import utils as ut
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#function to generate perturbed matrices close to original matrix
def pert_matrix(mat_in,scale,n_rnd_param): 
    #mat_in: squared matrix to be perturbed (N x N)
    #scale: added perturbation between [0,scale]
    #n_rnd_param: number of perturbed matrix generated
    
    #dimension of the problem
    N = mat_in.shape[0]
    
    #eigendecomposition
    w,v = np.linalg.eig(mat_in)
    V   = v
    W   = np.diag(w)

    #generate perturbed matrices
    A_noise = []
    for rng in range(n_rnd_param):
        V_noise = scale*np.random.rand(N,N) + V
        A_noise.append(V_noise @ W @ np.linalg.inv(V_noise))
        
    return A_noise

#function that receives perturbed matrices for A,U and builds the resulting trajectories
def simulate_traj(A_noise, U_noise, x0, offset,std, n_samples, horizon, n_rnd_param, cap_wind, cut_speed, a, b, c, d, g):
    #A_noise: perturbed matrices for A
    #U_noise: perturbed matrices for U
    #cap_wind: installed capacity of each wind farm
    #cut_speed: cut out speed for wind turbines
    
    #sampl_traj: output vector with dimensions (n_samples,horizon,n_rnd_param)
    
    #list of generated trajectories for EACH sampled VAR paremeter
    sampl_traj = np.zeros((n_samples,horizon,n_rnd_param)) 
    for s_samp in range(n_rnd_param):
        A_s = A_noise[s_samp]
        U_s = U_noise[s_samp]
        traj_wind = ut.sim_wind(A_s,U_s,x0,horizon,n_samples)
        traj_wind = offset + (traj_wind * std)
        pow_wind = ut.power_curve(traj_wind, cap_wind, cut_speed, a, b, c, d, g)
        sampl_traj[:,:,s_samp] = np.sum(pow_wind,axis=0)
    
    return sampl_traj
    
def process_traj(sampl_traj,ql,qu,n_samples,n_rnd_param,horizon,flag_hist):
    #sampl_traj: sampled trajectoried from perturbed matrices (simulate_traj)
    #ql,qu: lower and upper quantile (1..100)
    #flag_hist: show comparison of histograms? (example case)
    
    #mean_mix: vector of means of total power generation for each horizon
    #ql_mix: lower quantile vector of total power generation for each horizon
    #qu_mix: lupper quantile vector of total power generation for each horizon
    
    #Mixing of all sampled VAR parameters trajectories
    mixed_traj = np.swapaxes(sampl_traj,1,2)
    mixed_traj = np.reshape(mixed_traj,(n_samples*n_rnd_param,horizon))
    
    #Example figures (only first horizon)
    if(flag_hist):
        print("Example figures for first horizon")
        f, axarr = plt.subplots(5, sharex=True)
        for n in range(4):
            axarr[n].hist(sampl_traj[:,0,n], 50, density=True, facecolor='green', alpha=0.75)
            axarr[n].set_title('VAR Param case'+str(n+1))
        axarr[4].hist(mixed_traj[:,0],50, density=True, facecolor='green', alpha=0.75)
        axarr[4].set_title('Mixed VAR Param case')
    elif(flag_hist and (n_rnd_param>=5)):
        print("ERROR: Examples figures cannot be shown for more than 4 sampled VAR parameters")
        
    #Computation of intervals and mean
    mean_mix = np.mean(mixed_traj,axis=0)
    ql_mix = np.percentile(mixed_traj,ql,axis=0)
    qu_mix = np.percentile(mixed_traj,qu,axis=0)
        
    return [mean_mix,ql_mix,qu_mix]