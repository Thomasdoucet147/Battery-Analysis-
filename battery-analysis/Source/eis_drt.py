

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # ajoute Source au path
import basics  # maintenant importable
import nearest_PD as nPD
import pandas as pd
import numpy as np
import sys
from numpy import log, log10
from math import pi
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# TAKEN FROM : https://github.com/ciuccislab/pyDRTtools
#================================================
# EXPLANATION OF DRT (Distribution of Relaxation Times)
# ===================================================
#
# The DRT is a method that allows decomposing the
# complex impedance response of an electrochemical system (such as a battery)
# into elementary contributions associated with different physical mechanisms,
# each characterized by a relaxation time τ (tau).
#
# The measured signal (Z_im) is modeled as a weighted sum of effects
# at different time scales, represented by the distribution γ(τ).
#
# Mathematically, the DRT involves solving an inverse problem
# to find γ(τ) such that:
#
#    Z_im(ω) ≈ ∫ γ(τ) * Im{1/(1 + jωτ)} d(ln τ)
#
# where ω = 2πf.
#
# This problem is ill-posed (unstable inversion), so regularization
# (alpha, beta) is used to obtain a stable and smooth solution.
#
# Analysis of the DRT:
# - Peaks in γ(τ) indicate distinct phenomena with
#   characteristic relaxation times (e.g., charge transfer,
#   diffusion in the electrolyte, etc.).
# - The horizontal position of the peaks (τ) gives the time scale.
# - The amplitude of the peaks is related to the magnitude of the corresponding phenomenon.
#
# Series resistance (R0):
# - In an EIS spectrum, the series resistance R0 corresponds to the
#   pure ohmic resistance of the system (contacts, electrolyte, etc.).
# - It appears as the starting point of the Nyquist plot on the real axis.
# - The DRT **does not model this R0 resistance**, because it does not represent
#   a time-distributed process but a simple offset.
# - Therefore, R0 must always be considered separately when interpreting the DRT.
#
# ---------------------------------------------------



class EIS_object(object):
    
    # The EIS_object class stores the input data and the DRT result.
      
    def __init__(self, freq, Z_prime, Z_double_prime):
        
        """
        This is EIS_object class 
        Inputs:
            freq: frequency of the EIS measurement
            Z_prime: real part of the impedance
            Z_double_prime: imaginery part of the impedance
        """
        # define an EIS_object
        self.freq = freq
        self.Z_prime = Z_prime
        self.Z_double_prime = Z_double_prime
        self.Z_exp = Z_prime + 1j*Z_double_prime
        
        # keep a copy of the original data
        self.freq_0 = freq
        self.Z_prime_0 = Z_prime
        self.Z_double_prime_0 = Z_double_prime
        self.Z_exp_0 = Z_prime + 1j*Z_double_prime

        self.tau = 1/freq # we assume that the collocation points equal to 1/freq as default
        self.tau_fine  = np.logspace(log10(self.tau.min())-0.5,log10(self.tau.max())+0.5,10*freq.shape[0]) 
        ## select custom collocation

        # tau_fine = np.logspace(tau_min, tau_max, num = N_taus, endpoint=True)   

        self.method = 'none'
    

def simple_run(entry, rbf_type = 'Gaussian', data_used = 'Combined Re-Im Data', induct_used = 1, der_used = '1st order', cv_type = 'GCV', reg_param = 1E-3, shape_control = 'FWHM Coefficient', coeff = 0.5):
    
    
    """
    This function enables to compute the DRT using ridge regression (also known as Tikhonov regression)
    References:
        T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, Influence of the discretization methods on the distribution of relaxation times deconvolution: Implementing radial basis functions with DRTtools, Electrochimica Acta 184 (2015) 483-499.
    Inputs:
        entry: an EIS spectrum
        rbf_type: discretization function
        data_used: part of the EIS spectrum used for regularization
        induct_used: treatment of the inductance part
        der_used: order of the derivative considered for the M matrix
        cv_type: regularization method used to select the regularization parameter for ridge regression
        reg_param: regularization parameter applied when "custom" is used for cv_type 
        shape_control: option for controlling the shape of the radial basis function (RBF) 
        coeff: magnitude of the shape control
    """
    
    # Step 1.1: define the optimization bounds
    N_freqs = entry.freq.shape[0]
    N_taus = entry.tau.shape[0]
    ###
    entry.b_re = entry.Z_exp.real
    entry.b_im = entry.Z_exp.imag
    # Step 1.2: compute epsilon
    entry.epsilon = basics.compute_epsilon(entry.freq, coeff, rbf_type, shape_control)
    
    # Step 1.3: compute A matrix
    ## assemble_A_re(freq_vec, tau_vec, epsilon, rbf_type)
    entry.A_re_temp = basics.assemble_A_re(entry.freq, entry.tau, entry.epsilon, rbf_type)
    entry.A_im_temp = basics.assemble_A_im(entry.freq, entry.tau, entry.epsilon, rbf_type)
    
    # Step 1.4: compute M matrix  assemble_M_1(tau_vec, epsilon, rbf_type)
    if der_used == '1st order':
        entry.M_temp = basics.assemble_M_1(entry.tau, entry.epsilon, rbf_type)
    elif der_used == '2nd order':
        entry.M_temp = basics.assemble_M_2(entry.tau, entry.epsilon, rbf_type)
    
    # Step 2: conduct ridge regularization
    if data_used == 'Combined Re-Im Data': # select both parts of the impedance for the simple run
 
        if induct_used == 0 or induct_used == 2: # without considering the inductance
            N_RL = 1 # N_RL length of resistance plus inductance
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:,N_RL:] = entry.A_re_temp
            entry.A_re[:,0] = 1
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:,N_RL:] = entry.A_im_temp
            
            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
            # optimally select the regularization level
            # initial guess for the hyperparameter
            log_lambda_0 = log(reg_param) # initial guess for lambda
            #
            if cv_type=='custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 
                
            print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
            
            # recover the DRT using cvxopt
            H_combined,c_combined = basics.quad_format_combined(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, entry.lambda_value)
            # enforce positivity constraint # N_RL
            ## bound matrix
            G = matrix(-np.identity(entry.b_re.shape[0]+N_RL))
            h = matrix(np.zeros(entry.b_re.shape[0]+N_RL))
            # Formulate the quadratic programming problem
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_combined), matrix(c_combined),G,h)
            x = np.array(sol['x']).flatten()

            # prepare for HMC sampler, it will be used if needed
            entry.mu_Z_re = entry.A_re@x
            entry.mu_Z_im = entry.A_im@x
            entry.res_re = entry.mu_Z_re-entry.b_re
            entry.res_im = entry.mu_Z_im-entry.b_im

            # only consider std of residuals in both parts
            sigma_re_im = np.std(np.concatenate([entry.res_re,entry.res_im]))
            inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
        
            Sigma_inv = (entry.A_re.T@inv_V@entry.A_re) + (entry.A_im.T@inv_V@entry.A_im) + (entry.lambda_value/sigma_re_im**2)*entry.M
            mu_numerator = entry.A_re.T@inv_V@entry.b_re + entry.A_im.T@inv_V@entry.b_im
           
        elif induct_used == 1: # considering the inductance
            N_RL = 2
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            entry.A_re[:,1] = 1
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            entry.A_im[:,0] = 2*pi*entry.freq

            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
            # optimally select the regularization level
            log_lambda_0 = log(reg_param) # initial guess for lambda
            if cv_type=='custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 
                
            print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
            
            # recover the DRT using cvxopt
            H_combined,c_combined = basics.quad_format_combined(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, entry.lambda_value)
            # enforce positivity constraint # N_RL
            ## bound matrix
            G = matrix(-np.identity(entry.b_re.shape[0]+N_RL))
            h = matrix(np.zeros(entry.b_re.shape[0]+N_RL))
            # Formulate the quadratic programming problem
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_combined), matrix(c_combined),G,h)
            x = np.array(sol['x']).flatten()

            entry.mu_Z_re = entry.A_re@x
            entry.mu_Z_im = entry.A_im@x
            entry.res_re = entry.mu_Z_re-entry.b_re
            entry.res_im = entry.mu_Z_im-entry.b_im

            # only consider std of residuals in both parts
            sigma_re_im = np.std(np.concatenate([entry.res_re,entry.res_im]))
            inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
        
            Sigma_inv = (entry.A_re.T@inv_V@entry.A_re) + (entry.A_im.T@inv_V@entry.A_im) + (entry.lambda_value/sigma_re_im**2)*entry.M
            mu_numerator = entry.A_re.T@inv_V@entry.b_re + entry.A_im.T@inv_V@entry.b_im
            
    elif data_used == 'Im Data': # select the imaginary part of the impedance for the simple run
        
        if induct_used == 0 or induct_used == 2: # without considering the inductance
            N_RL = 0 # N_RL length of resistance plus inductance
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            
            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
            # optimally select the regularization level
            log_lambda_0 = log(reg_param) # initial guess for lambda
            if cv_type=='custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 
                
            print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
            
            # recover the DRT using cvxopt
            H_im, c_im = basics.quad_format_separate(entry.A_im, entry.b_im, entry.M, entry.lambda_value)
            # enforce positivity constraints
            ## bound matrix
            G = matrix(-np.identity(entry.b_im.shape[0]+N_RL))
            h = matrix(np.zeros(entry.b_im.shape[0]+N_RL))
            # Formulate the quadratic programming problem
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_im), matrix(c_im),G,h)
            x = np.array(sol['x']).flatten()

            # prepare for HMC sampler
            entry.mu_Z_re = entry.A_re@x
            entry.mu_Z_im = entry.A_im@x
            entry.res_re = entry.mu_Z_re-entry.b_re
            entry.res_im = entry.mu_Z_im-entry.b_im
            
            # only consider std of residuals in the imaginary part
            sigma_re_im = np.std(entry.res_im)
            inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
            
            Sigma_inv = (entry.A_im.T@inv_V@entry.A_im) + (entry.lambda_value/sigma_re_im**2)*entry.M
            mu_numerator = entry.A_im.T@inv_V@entry.b_im

            
        elif induct_used == 1: # considering the inductance
            N_RL = 1
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            entry.A_im[:,0] = 2*pi*entry.freq
            
            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
            # optimally select the regularization level
            log_lambda_0 = log(reg_param) # initial guess for lambda
            if cv_type=='custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 

            print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
            
            # recover the DRT using cvxopt
            
            H_im, c_im = basics.quad_format_separate(entry.A_im, entry.b_im, entry.M, entry.lambda_value)
            #
            # enforce positivity constraints
            # bound matrix
            G = matrix(-np.identity(entry.b_im.shape[0]+N_RL))
            h = matrix(np.zeros(entry.b_im.shape[0]+N_RL))
            # Formulate the quadratic programming problem
            ##
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_im), matrix(c_im),G,h)
            x = np.array(sol['x']).flatten()

            # prepare for HMC sampler
            entry.mu_Z_re = entry.A_re@x
            entry.mu_Z_im = entry.A_im@x
            entry.res_re = entry.mu_Z_re-entry.b_re
            entry.res_im = entry.mu_Z_im-entry.b_im
            
            # only consider std of residuals in the imaginary part
            sigma_re_im = np.std(entry.res_im)
            inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
            
            Sigma_inv = (entry.A_im.T@inv_V@entry.A_im) + (entry.lambda_value/sigma_re_im**2)*entry.M
            mu_numerator = entry.A_im.T@inv_V@entry.b_im

    elif data_used == 'Re Data': # select the real part of the impedance for the simple run
        N_RL = 1
        entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
        entry.A_re[:, N_RL:] = entry.A_re_temp
        entry.A_re[:,0] = 1
        
        entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
        entry.A_im[:, N_RL:] = entry.A_im_temp

        entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
        entry.M[N_RL:,N_RL:] = entry.M_temp
        
        # optimally select the regularization level
        log_lambda_0 = log(reg_param) # initial guess for lambda
        if cv_type=='custom':
            entry.lambda_value = reg_param
        else:
            entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 

        print('The value of the regularization parameter is', entry.lambda_lambda) # to check the value of lambda
        
        # recover the DRT using cvxopt 
        H_re,c_re = basics.quad_format_separate(entry.A_re, entry.b_re, entry.M, entry.lambda_value)
    
        # enforce positivity constraints
        # ## bound matrix
        G = matrix(-np.identity(entry.b_re.shape[0]+N_RL))
        h = matrix(np.zeros(entry.b_re.shape[0]+N_RL))
        # Formulate the quadratic programming problem
        ###
        # Solve the quadratic programming problem
        sol = solvers.qp(matrix(H_re), matrix(c_re),G,h)
        x = np.array(sol['x']).flatten()

        # prepare for HMC sampler
        entry.mu_Z_re = entry.A_re@x
        entry.mu_Z_im = entry.A_im@x       
        entry.res_re = entry.mu_Z_re-entry.b_re
        entry.res_im = entry.mu_Z_im-entry.b_im
        
        # only consider std of residuals in the real part
        sigma_re_im = np.std(entry.res_re)
        inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
        
        Sigma_inv = (entry.A_re.T@inv_V@entry.A_re) + (entry.lambda_value/sigma_re_im**2)*entry.M
        mu_numerator = entry.A_re.T@inv_V@entry.b_re

    entry.Sigma_inv = (Sigma_inv+Sigma_inv.T)/2
    
    # test if the covariance matrix is positive definite
    if (nPD.is_PD(entry.Sigma_inv)==False):
        entry.Sigma_inv = nPD.nearest_PD(entry.Sigma_inv) # if not, use the nearest positive definite matrix
    
    L_Sigma_inv = np.linalg.cholesky(entry.Sigma_inv)
    entry.mu = np.linalg.solve(L_Sigma_inv, mu_numerator)
    entry.mu = np.linalg.solve(L_Sigma_inv.T, entry.mu)
    # entry.mu = np.linalg.solve(entry.Sigma_inv, mu_numerator)
    
    # Step 3: obtaining the result of inductance, resistance, and gamma  
    if N_RL == 0: 
        entry.L, entry.R = 0, 0        
    elif N_RL == 1 and data_used == 'Im Data':
        entry.L, entry.R = x[0], 0    
    elif N_RL == 1 and data_used != 'Im Data':
        entry.L, entry.R = 0, x[0]
    elif N_RL == 2:
        entry.L, entry.R = x[0:2]
        
    entry.x = x[N_RL:]
    entry.out_tau_vec, entry.gamma = basics.x_to_gamma(x[N_RL:], entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    entry.N_RL = N_RL 
    entry.method = 'simple'
    
    return entry


# this code performs peak deconvolution using either Gaussian, Havriliak-Negami (HN), or ZARC functions

def peak_fct(p, tau_vec, N_peaks, fit='Gaussian'):
    
    """
    This function returns a fit of the peaks in the DRT spectrum

    Inputs:
        p: parameters of the Gaussian functions (sigma_f, mu_log_tau, and inv_sigma for each DRT peak)
        tau_vec: vector of timescales
        N_peaks: number of peaks in the DRT spectrum
        fit: nature of the DRT fit (Gaussian, HN, or ZARC)

    Output:
        gamma_out: sum of Gaussian functions
    """
    
    if fit=='Gaussian': # fit with Gaussian functions
        
        gamma_out = np.zeros_like(tau_vec) # sum of Gaussian functions, whose parameters (the prefactor sigma_f, mean mu_log_tau, and standard deviation 1/inv_sigma for each DRT peak) are encapsulated in p
    
        for k in range(N_peaks):
        
            sigma_f, mu_log_tau, inv_sigma = p[3*k:3*k+3] 
            gaussian_out = sigma_f**2*np.exp(-inv_sigma**2/2*((np.log(tau_vec) - mu_log_tau)**2)) # we use inv_sigma because this leads to less computational problems (no exploding gradient when sigma->0)
            gamma_out += gaussian_out 
            
    elif fit=='Havriliak-Negami': # fit with HN functions
        
        gamma_out = np.zeros_like(tau_vec) # sum of single-ZARC functions, whose parameters (R_ct, log_tau_0, phi for each DRT peak) are encapsulated in p
        
        for k in range(N_peaks):
            
            R_ct, log_tau_0, phi, psi = p[4*k:4*k+4] 
            
            x = np.exp(phi*(np.log(tau_vec)-log_tau_0))
            
            theta = np.arctan(np.abs(np.sin(np.pi*phi)/(x+np.cos(np.pi*phi))))
            
            num = R_ct*x**psi*np.sin(psi*theta)
            denom = np.pi*(1+np.cos(np.pi*phi)*x+x**2)**(psi/2)
            
            DRT_HN_out = num/denom
            
            gamma_out += DRT_HN_out 
            
    else: # fit with ZARC functions
    
        gamma_out = np.zeros_like(tau_vec) # sum of single-ZARC functions, whose parameters (R_ct, log_tau_0, phi for each DRT peak) are encapsulated in p
        
        for k in range(N_peaks):
            
            R_ct, log_tau_0, phi = p[3*k:3*k+3] 
            x = np.exp(phi*(np.log(tau_vec)-log_tau_0))
            DRT_ZARC_out = R_ct*np.sin(np.pi*phi)*x/(1+2*np.cos(np.pi*phi)*x+x**2)
            gamma_out += DRT_ZARC_out 
            
    return gamma_out

def peak_analysis(entry, rbf_type='Gaussian', data_used='Combined Re-Im Data', induct_used=1, der_used='1st order', cv_type='GCV', reg_param=1E-3, shape_control='FWHM Coefficient', coeff=0.5, peak_method='separate', N_peaks=1):
    """
    This function identifies the DRT peaks.
    
    Inputs:
        entry: an EIS spectrum
        rbf_type: discretization function
        data_used: part of the EIS spectrum used for regularization
        induct_used: treatment of the inductance part
        der_used: order of the derivative considered for the M matrix
        cv_type: regularization method used to select the regularization parameter for ridge regression
        reg_param: regularization parameter applied when "custom" is used for cv_type  
        shape_control: option for controlling the shape of the radial basis function (RBF) 
        coeff: magnitude of the shape control
        N_peaks: desired number of peaks
        peak_method: option for the fit of the recovered DRT, either 'separate' to separately fit each peak, or 'combine' to optimize all the peaks at once
    """
    
    # Step 1: define the necessary quantities before the subsequent optimizations
    entry.N_peaks = int(N_peaks)
    
    # Run initial simple analysis
    simple_run(entry, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used, 
               der_used=der_used, cv_type=cv_type, reg_param=reg_param, shape_control=shape_control, coeff=coeff) 
    
    # upper and lower log tau values
    log_tau_min = np.min(np.log(entry.out_tau_vec))
    log_tau_max = np.max(np.log(entry.out_tau_vec))
    
    # Find peaks in the gamma spectrum with a lower threshold for height
    dynamic_threshold = 0.05 * np.mean(entry.gamma)  # Lower threshold for small peak detection
    peak_indices, _ = find_peaks(entry.gamma, height=dynamic_threshold, distance=5)

    # Adjust N_peaks if fewer peaks are found than specified
    N_peaks = min(len(peak_indices), entry.N_peaks)
    num_peaks_found = len(peak_indices)
    
    if entry.N_peaks > num_peaks_found:
        print(f"Warning: Adjusting N_peaks to {num_peaks_found}.")
        entry.N_peaks = num_peaks_found

    # Set bounds for optimization
    bounds = []
    for _ in range(N_peaks):
        bounds.extend([
            (0, 1.3 * np.sqrt(np.max(entry.gamma))),  # Bounds for sigma_f (peak height)
            (log_tau_min, log_tau_max),               # Bounds for mu_log_tau (peak location)
            (1.0 / (log_tau_max - log_tau_min), 10)   # Bounds for inv_sigma (peak width)
        ])
    
    # Define the objective function for optimization
    def objective(params):
        return np.sum((peak_fct(params, entry.out_tau_vec, entry.N_peaks, 'Gaussian') - entry.gamma) ** 2)
    
    # Parameter optimization using differential evolution
    out_fit_tot = differential_evolution(objective, bounds, popsize=30, mutation=(0.5, 1.5), recombination=0.9,
                                         strategy='best1bin', seed=42, maxiter=200, workers=1)
    
    # Refine parameters using L-BFGS-B
    theta_fit_tot = out_fit_tot.x
    out_fit_tot = minimize(objective, theta_fit_tot, bounds=bounds, method='L-BFGS-B')
    theta_fit_tot = out_fit_tot.x
    entry.gamma_fit_tot = peak_fct(theta_fit_tot, entry.out_tau_vec, entry.N_peaks, 'Gaussian')
    
    # Generate individual Gaussian fits for each peak
    gamma_fit_list = [0] * N_peaks
    for k in range(entry.N_peaks):
        params = theta_fit_tot[3 * k:3 * k + 3]
        gamma_fit = peak_fct(params, entry.out_tau_vec, 1, 'Gaussian')
        gamma_fit_list[k] = gamma_fit
    
    if peak_method == 'separate':  # separate fit of the DRT
        entry.out_gamma_fit = gamma_fit_list
        entry.Gaussian = np.array(gamma_fit_list)
        entry.num_vectors = entry.Gaussian.shape[0]
        entry.column_headings = [f'Gaussian_{i+1}' for i in range(entry.num_vectors)]
        entry.df = pd.DataFrame(entry.Gaussian.T, columns=entry.column_headings)
    else:  # combine fit of the DRT
        entry.out_gamma_fit = entry.gamma_fit_tot
    
    entry.method = 'peak'
    
    return entry