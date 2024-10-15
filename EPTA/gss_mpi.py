#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:21:34 2024

@author: elmehdi
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

def log_sum(vec):
    r = -np.Inf
    for i in range(len(vec)):
       #print('element:',vec[i])                         # From Patricio's R code
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
            
        
        def lnpriorref(self,x):
            return  np.sum(norm.logpdf(x , loc = self.mu0s , scale = self.sig0s ))    
            #return  np.log(np.prod((uniform.pdf(x , loc = self.mu0s , scale = self.sig0s ))))
        def new_lnlikefn (self,x):
            return (self.lnlikefn(params = x)+self.lnpriorfn(params = x))-self.lnpriorref(x = x)
        
        def Im_dist(self,x,beta):
            lnlike_val = self.new_lnlikefn(x)
            p_teta = beta*lnlike_val+self.lnpriorref(x)
            return p_teta , lnlike_val
        
        
    def __init__(self,N_chain):
        
        self.N_chain = N_chain
        print("Running GSS sampling:")

    
    def Gss_mpi(N_chain,ndim,n_samp,beta_dT,dist,mu0s,sig0s,thin,comm = comm,rank = rank, size=size):
        log_r_k = 0
        targ_dist = 0
        
        target = {}
            # Define temperatures for each process
        temperatures = beta_dT

            # Scatter temperatures across processes
        temperature = comm.scatter(temperatures, root=0)

            # Generate samples for the assigned temperature
        q_b = Gss_sampler.sampler_mpi(N_chain,ndim,n_samp,temperature,dist,mu0s,sig0s,rank,thin)

        
        results = comm.gather(q_b, root=0)
            
        comm.barrier()    
        if rank == 0:
                
                nu_k = np.zeros(N_chain)
                pse_rk = np.zeros(N_chain)
                for idx, result in enumerate(results):
                    target[str(result[-1])] = result[:-1]
                    
                    #print(f"Results from process {idx}: {result}")
                    #print('target',target)
                targ_dist = target
                for i in range(N_chain-1):
                    targ_dist_b = targ_dist.get(str(beta_dT[i]))
                    nu_k[i+1] = np.max(targ_dist_b)
                
                #nu_k = np.zeros(N_chain)
                print('nu_k:', nu_k)
                
                for i in range(1,len(beta_dT)):
                    

                    targ_dist_b = targ_dist.get(str(beta_dT[i-1]))
                    
                    n_samp = len(targ_dist_b)
                    
                    diff = beta_dT[i] - beta_dT[i-1]
                    
                    #print("beta diff :", diff)
                    
                    stab = targ_dist_b - nu_k[i]
                    
                    pse_rk[i] = diff*nu_k[i] + log_sum(diff*stab) -  np.log(n_samp)
                  
                log_r_k = np.sum(pse_rk)  
                
                print('summ pse_rk', np.sum(pse_rk))
                print('log(n)',(len(beta_dT)-1)*np.log(n_samp))
                
        return log_r_k
            
        
        
    
    def sampler_mpi(N_chain,ndim,n_samp,temp,target,mu0s,sig0s,rank,thin):
        
        
        val = 1
        samples = []
        targ_dist = []
        j = rank
        id = 0
        nsamp_thin = int(n_samp/thin) 
        q_b = np.zeros(nsamp_thin)
        #teta = mu0s
        #teta = np.random.uniform(low=mu0s-sig0s, high=mu0s+sig0s,size=(ndim))
        teta = np.random.normal(mu0s,sig0s*val,size=(ndim))
        samples.append(teta)

        P_teta , save_targ = target.Im_dist(teta, temp)
        q_b[0] = save_targ
        while (q_b[0]  == -np.inf or q_b[0]  == np.nan):
           teta = np.random.uniform(low=mu0s-sig0s, high=mu0s+sig0s,size=(ndim))
           #teta = np.random.normal(mu0s,sig0s*val,size=(ndim))
           P_teta , save_targ = target.Im_dist(teta, temp)
           q_b[0] = save_targ
        
        
        #print('Running MCMC for chain:',j)
        
        for i in range(n_samp):
            
            #print('chain:',j+1,'iterion:',i+1,'out of',n_samp)
            idx = np.random.choice(np.arange(0,ndim,1))
            
            new_teta = np.copy(teta)
            #print('tita : ',new_teta)
            for mi in range(int(0.05*ndim)+1):
                idx = np.random.choice(np.arange(0,ndim,1))
                new_teta[idx] = teta[idx]+np.random.normal(0,sig0s[idx]*val,size=(1))
            #new_teta[idx] = teta[idx]+np.random.normal(0,sig0s[idx]*val,size=(1))


            P_new , im_new  = target.Im_dist(new_teta ,temp)

            alpha = min(0, P_new - P_teta)
                    
            u = np.log(np.random.rand(1))

            
            if (u < alpha):
               teta = new_teta  
               P_teta = P_new
               save_targ = im_new

            if (i+1) % thin == 0:
                #print('id:', id) 
                q_b [id] = save_targ   
                samples.append(teta)
                id += 1        
        
        samp_ess = np.array(samples)    
        ess=np.zeros(ndim)
        for j in range(ndim):
            ess[j] = az.ess(samp_ess[:,j])
        print(f'Effective Sample Size: max {max(ess)} min {min(ess)}')
        q_b = np.append(q_b,temp)
        targ_dist = q_b
        #print('temp:',q_b[-1])
        return targ_dist 
        
    def sampler(N_chain,ndim,n_samp,beta_dT,target,mu0s,sig0s):
        
        targ_dist = {}
        
        
        for j in range(N_chain): 
            
            samples = []
            teta = np.random.normal(mu0s,sig0s,size=(ndim))        #proposal optimization is mine
            
            samples.append(teta)
            q_b = np.zeros(n_samp)
            q_b [0] = target.new_lnlikefn(teta)
            
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
                q_b [i+1] = target.new_lnlikefn(teta)    
            
            #q_b = [~np.isnan(q_b)]
            print(q_b)
            
            targ_dist[str(beta_dT[j])] = q_b

        #print(targ_dist)
        
        return targ_dist 
        
        
        
        
        # Gss to compute r_k
        
    def sample_is(self,n_samp,mu0s,sig0s,lnlikefn, lnpriorfn, thin=1, parallel = False): #from Fan et al. 2010
        
        
        
        
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
             
             stop_time = time.time()
             print ("Time for GSS estimation ---", int((time.time()-start_time)), " seconds .---")
             return log_r_k  , targ_dist
        
        




def direct_samp_post(dim,v,n_samples):
    
    print('Direct sampling from post :',n_samples)
    ln_post = {}
    
    for i in range(dim):
        #p_b = np.random.lognormal(0,v/(v+bet[i]),size = n_samples)  
        p_b = np.random.normal(0,np.sqrt(v/(v+1)),size = (n_samples))
        
        
        ln_post[str(i+1)] = p_b
        
    return ln_post

def ref_calibrate(post_samples):
        
        li = list(post_samples.keys())
        
        mu = np.zeros(len(li))
        sig = np.zeros(len(li))
        for j in range(len(li)):
              ran_samples = post_samples.get(li[j])
              mu[j] = np.mean(ran_samples)
              sig[j] = np.std(ran_samples)

        return mu , sig
def SS(ln_like):
    
    log_z =0
    temp_list = list(ln_like.keys())
    #print('temperature list is:',temp_list)
    temp = np.array([float(kk) for kk in temp_list])
    beta = temp
    #print('temp_list',temp_list)
    #print('beta',beta)
    log_save_like = np.zeros(len(beta)-1)
    
    for i in range(len(beta)-1):
        
        #print(beta[i+1])
        diff = beta[i+1] - beta[i]
        #print('diff',diff)
        chain = ln_like.get(temp_list[i])
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

def read_post(folder_path,param_name, burn_in):
   
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
       print('Calibrating using chains from file',chain_list[0])
       data = []
       filename = chain_list[0]
       full_path = os.path.join(folder_path, filename)
       temp = np.genfromtxt(full_path,dtype=float,unpack=True)
       ndim = len(temp)-4
       print('Number of dimensions :',ndim)
       for i in range(ndim):
        post = temp[i,burn_in:]              
        data.append(post) 
        results[param_name[i]] = post
       print('Posterior samples imported')       
       return results


