#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:21:34 2024

@author: El Mehdi Zahraoui
"""


import numpy as np
import os
from scipy.stats import norm
from scipy.stats import beta
import arviz as az

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import time


def log_plus(x,y):
    
    if x > y:
      summ = x + np.log(1+np.exp(y-x))
    else:
        summ = y + np.log(1+np.exp(x-y))
    return summ

def log_sum(vec):    # Summing log vectors From PMRs' SS R code
    r = -np.Inf
    for i in range(len(vec)):
       #print('element:',vec[i])                         
       r =log_plus(r, vec[i])
       #print(r)
    return r



class Gss_sampler(object):
    
    class lnrefprior(object):
        
        def __init__(self,lnlikefn, lnpriorfn,mu0s,sig0s):
            self.lnlikefn  = lnlikefn
            self.lnpriorfn = lnpriorfn
            self.mu0s = mu0s
            self.sig0s = sig0s
            
            print('defining Reference prior distribution.')
           # print('mu0 :', self.mu0s)
            #print('sig0 :', self.sig0s)
            
        
        def lnref(self,x):      # pi_0: Refrence distribution 
            return  np.sum(norm.logpdf(x , loc = self.mu0s , scale = self.sig0s ))
        
        def ln_pseudo_lik (self,x):   # The GSS new pseudo-likelihood
            return (self.lnlikefn(params = x)+self.lnpriorfn(params = x))-self.lnref(x = x)
        
        def Im_dist(self,x,beta):     # ln_pslike_betaeta: Importance sampling distribution
            lnlike_val = self.ln_pseudo_lik(x)
            p_teta = beta*lnlike_val+self.lnref(x)
            return p_teta , lnlike_val
        
        
    def __init__(self,N_chain):
        
        self.N_chain = N_chain
        print("Running GSS sampling:")

    
    def Gss_mpi(N_chain,ndim,n_samp,beta_dT,dist,mu0s,sig0s,thin,comm = comm,rank = rank, size=size):

        log_r_k = 0
        targ_dist = 0
        
        target = {}

        beta_rank = comm.scatter(beta_dT, root=0)  # Assigning betas to each process

        ln_pslike_beta = Gss_sampler.sampler_mpi(ndim,n_samp,beta_rank,dist,mu0s,sig0s,rank,thin) # running the parrallelized M-H sampler

        results = comm.gather(ln_pslike_beta, root=0) # Receiving the pseudo-likelihood array with the last value being the beta value
            
        comm.barrier()    
        if rank == 0:  # Calculating gss at the end of the sampling 
                
                nu_k = np.zeros(N_chain)
                pse_rk = np.zeros(N_chain)

                for result in enumerate(results):   # Re-arranging the received samples into a dictionnary   
                    target[str(result[-1])] = result[:-1]
                    #print('target',target)
                targ_dist = target

                for i in range(N_chain-1):  # Creating of an Array for nu_max
                    targ_dist_b = targ_dist.get(str(beta_dT[i]))
                    nu_k[i+1] = np.max(targ_dist_b)
                
                #nu_k = np.zeros(N_chain)
                #print('nu_k:', nu_k)
                
                for i in range(1,len(beta_dT)):  # Summing over the r_k ratios
                    

                    targ_dist_b = targ_dist.get(str(beta_dT[i-1]))
                    
                    n_samp = len(targ_dist_b)
                    
                    diff = beta_dT[i] - beta_dT[i-1]
                    
                    #print("beta diff :", diff)
                    
                    stab = targ_dist_b - nu_k[i]
                    
                    pse_rk[i] = diff*nu_k[i] + log_sum(diff*stab) -  np.log(n_samp)
                  
                log_r_k = np.sum(pse_rk)  
                
                print('GSS estimation:', np.sum(pse_rk))
                
        return log_r_k
            
        
        
    
    def sampler_mpi(ndim,n_samp,beta_k,target,mu0s,sig0s,rank,thin):
        #
        

        samples = []
        targ_dist = []

        j = rank

        id = 0

        n_gss = int(n_samp/thin) 

        ln_pslike_beta = np.zeros(n_gss) # Array for saving  thinned_samples

        teta = np.random.normal(mu0s,sig0s,size=(ndim)) # intializing theta from the refrence distribution eliminates the burning period 

        samples.append(teta)

        P_teta , save_pslike = target.Im_dist(teta, beta_k)

        ln_pslike_beta[0] = save_pslike

        while (ln_pslike_beta[0]  == -np.inf or ln_pslike_beta[0]  == np.nan): # Ensuring the starting point is not -np.inf or nan 
           teta = np.random.uniform(low=mu0s-sig0s, high=mu0s+sig0s,size=(ndim))
           #teta = np.random.normal(mu0s,sig0s*val,size=(ndim))
           P_teta , save_pslike = target.Im_dist(teta, beta_k)
           ln_pslike_beta[0] = save_pslike
        
        
        #print('Running MCMC for chain:',j)
        
        for i in range(n_samp):  # looping the MH walks
            
            #print('chain:',j+1,'iterion:',i+1,'out of',n_samp)
            idx = np.random.choice(np.arange(0,ndim,1)) 
            
            new_teta = np.copy(teta)
            #print('tita : ',new_teta)

            for mi in range(int(0.05*ndim)+1):  # Updating only 5% of the total dimensions to solve the rejection problem
                idx = np.random.choice(np.arange(0,ndim,1))
                new_teta[idx] = teta[idx]+np.random.normal(0,sig0s[idx],size=(1))
            #new_teta[idx] = teta[idx]+np.random.normal(0,sig0s[idx]*val,size=(1))


            P_new , im_new  = target.Im_dist(new_teta ,beta_k)

            alpha = min(0, P_new - P_teta)
                    
            u = np.log(np.random.rand(1))

            
            if (u < alpha):
               teta = new_teta  
               P_teta = P_new
               save_pslike = im_new

            if (i+1) % thin == 0: # Saving the thinned_samples 
                #print('id:', id) 
                ln_pslike_beta [id] = save_pslike   
                samples.append(teta)
                id += 1        
        
        samp_ess = np.array(samples)    
        ess=np.zeros(ndim)
        for j in range(ndim):
            ess[j] = az.ess(samp_ess[:,j])
        if rank == 0 :
            print(f'Effective Sample Size: max {max(ess)} min {min(ess)}')  # Printig the effective sample size from the saved samples 
        ln_pslike_beta = np.append(ln_pslike_beta,beta_k)
        targ_dist = ln_pslike_beta
        #print('beta_k:',ln_pslike_beta[-1])
        return targ_dist 
        
    def sampler(N_chain,ndim,n_samp,beta_dT,target,mu0s,sig0s):
        
        targ_dist = {}
        
        
        for j in range(N_chain): 
            
            samples = []
            teta = np.random.normal(mu0s,sig0s,size=(ndim))        #proposal optimization is mine
            
            samples.append(teta)
            ln_pslike_beta = np.zeros(n_samp)
            ln_pslike_beta [0] = target.ln_pseudo_lik(teta)
            
            print('Running MCMC for chain:',j+1)
            
            for i in range(n_samp-1):
                
                print('chain:',j+1,'iterion:',i,'out of',n_samp)
             
                new_teta = np.random.normal(mu0s,sig0s,size=(ndim))
                #new_teta = np.random.uniform(-1,1,size=(ndim))
                #print('tita : ',new_teta)

                P_new = target.Im_dist(new_teta , beta_dT[j])
                P_teta = target.Im_dist(teta , beta_dT[j])
                
                print(P_new)
                #print(P_teta)
                
                #if P_new - P_teta != np.nan :
                alpha = min(0, P_new - P_teta)
                #else:
                #    alpha = -1e8   # hahaha just in case
                #print(len(alpha))
                #print('alpha=',alpha)
                
                u = np.log(np.random.rand(1))
                
                #print ('u :', u)
                #print ('alpha :', pdf(tita[i])/pdf(teta[i]))
                
                
                if (u < alpha):
                   teta = new_teta  
                   
                samples.append(teta)        
                ln_pslike_beta [i+1] = target.ln_pseudo_lik(teta)    
            
            #ln_pslike_beta = [~np.isnan(ln_pslike_beta)]
            print(ln_pslike_beta)
            
            targ_dist[str(beta_dT[j])] = ln_pslike_beta

        #print(targ_dist)
        
        return targ_dist 
        
        
        
        
        # Gss to compute r_k
        
    def sample_is(self,n_samp,mu0s,sig0s,lnlikefn, lnpriorfn, thin=1, parallel = False):
        
        
        
        
        N_chain = self.N_chain
        ndim  = len(mu0s)

        dT = np.linspace(1e-8,1,N_chain)
        
        beta_dT = beta.ppf(dT,0.3,1)

        nu_k = np.zeros(N_chain)
        
        
        pse_rk = np.zeros(N_chain)
    
        
        Im = Gss_sampler.lnrefprior(lnlikefn, lnpriorfn,mu0s,sig0s)
        
        
        if parallel == True:
             
             log_r_k  = Gss_sampler.Gss_mpi(N_chain, ndim, n_samp, beta_dT, Im, mu0s, sig0s,thin)
             
             return log_r_k 
        else:
             
             targ_dist  = Gss_sampler.sampler(N_chain, ndim, n_samp, beta_dT, Im, mu0s, sig0s)
             # Gss to compute r_k
             
             for i in range(N_chain-1):
                 targ_dist_b = targ_dist.get(str(beta_dT[i]))
                 nu_k[i+1] = np.max(targ_dist_b)
             
             #nu_k = np.zeros(N_chain)
             print('nu_k:', nu_k)
             for i in range(1,len(beta_dT)):
                 

                 targ_dist_b = targ_dist.get(str(beta_dT[i-1]))
                 
                 #n_samp = len(targ_dist_b)
                 
                 diff = beta_dT[i] - beta_dT[i-1]
                 
                 #print("beta diff :", diff)
                 
                 stab = targ_dist_b - nu_k[i]
                 
                 pse_rk[i] = diff*nu_k[i] + log_sum(diff*stab) -  np.log(n_samp)
               
             log_r_k = np.sum(pse_rk)  
             
             print('summ pse_rk', np.sum(pse_rk))
             print('log(n)',(len(beta_dT)-1)*np.log(n_samp))
             
             print ("Time for GSS estimation ---", int((time.time()-start_time)), " seconds .---")
             return log_r_k  , targ_dist
        
        



def ref_calibrate(post_samples): # Output an array mu of the mean and an array of std sig of each parameter from posterior dictionnary
        
        li = list(post_samples.keys())
        
        mu = np.zeros(len(li))
        sig = np.zeros(len(li))
        for j in range(len(li)):
              ran_samples = post_samples.get(li[j])
              mu[j] = np.mean(ran_samples)
              sig[j] = np.std(ran_samples)

        return mu , sig

def SS(ln_like): # Marginal likelihood estimation with SS from a dictionnary with Betas as keys for each likelihood chain
    
    log_z =0
    beta_k_list = list(ln_like.keys())
    #print('beta_rank list is:',beta_k_list)
    beta_k = np.array([float(kk) for kk in beta_k_list])
    beta = beta_k
    #print('beta_k_list',beta_k_list)
    #print('beta',beta)
    log_save_like = np.zeros(len(beta)-1)
    
    for i in range(len(beta)-1):
        
        #print(beta[i+1])
        diff = beta[i+1] - beta[i]
        #print('diff',diff)
        chain = ln_like.get(beta_k_list[i])
        #print('chains:', chain)
        #chain = np.exp(chain)
        n = len(chain)
        #print('n:',n)
        log_save_like[i] = log_sum(chain*diff) 
        #print( log_save_like[i])
       
      
    #print('log like sum :', sum(log_save_like))
    #print('log n :',  (len(beta)-1)*np.log(n))
    log_z = np.sum(log_save_like) - (len(beta)-1)*np.log(n)
    return log_z  

def read_post(folder_path,param_name, burn_in): # Read posterior samples from PTMCMC chain file and outputs dictionnary with parameter names as keys 
   
       results = {}
       delimiter=' '
       chain_list = []
       for filename in os.listdir(folder_path):
         if filename.startswith("chain_"):
            chain_list.append(filename)
           #print(chain_list)     
       chain_list = [name.replace('chain_','') for name in chain_list]
       chain_list = [name.replace('.txt','') for name in chain_list]
       chain_list = sorted(chain_list,key=float)
       chain_list = [str('chain_')+name for name in chain_list]
       chain_list = [name+str('.txt') for name in chain_list]
       #print('Calibrating using chains from file',chain_list[0])
       data = []
       filename = chain_list[0]
       full_path = os.path.join(folder_path, filename)
       beta_k = np.genfromtxt(full_path,dtype=float,unpack=True)
       ndim = len(beta_k)-4
       print('Number of dimensions :',ndim)
       for i in range(ndim):
        post = beta_k[i,burn_in:]              
        data.append(post) 
        results[param_name[i]] = post
       print('Posterior samples imported')       
       return results


