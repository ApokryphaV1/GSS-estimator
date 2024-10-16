#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:53:28 2024

@author: elmehdi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:29:27 2024

@author: mqp5725
"""
import glob, json
import numpy as np

from enterprise.signals import parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals import white_signals
from enterprise.signals import gp_signals

from h5pulsar.pulsar import FilePulsar



import time
from mpi4py import MPI

import gss_mpi

if __name__ == "__main__":
    Comm = MPI.COMM_WORLD
    Rank = Comm.Get_rank()
    Size = Comm.Get_size()
    print('imported Enterprise.')


    datadir = './15yr_stochastic_analysis/tutorials/data'

    psrs = []
    for hdf5_file in glob.glob(datadir + '/hdf5/*.hdf5'):
        psrs.append(FilePulsar(hdf5_file))
    print('Loaded {0} pulsars from hdf5 files'.format(len(psrs)))


    ## Get parameter noise dictionary
    noise_ng15 = datadir + '/15yr_wn_dict.json'

    wn_params = {}
    with open(noise_ng15, 'r') as fp:
        wn_params.update(json.load(fp))
        
    # find the maximum time span to set GW frequency sampling
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)

    # white noise parameters
    efac = parameter.Constant() 
    t2equad = parameter.Constant() 
    ecorr = parameter.Constant()
    # we'll set these later with the wn_params dictionary

    # red noise parameters
    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)

    # GW parameters (initialize with names here to use parameters in common across pulsars)
    log10_A_gw = parameter.Uniform(-18, -14)('log10_A_gw')
    gamma_gw = parameter.Uniform(0, 7)('gamma_gw')


    # white noise
    mn = white_signals.MeasurementNoise(efac=efac, log10_t2equad=t2equad, selection=selection)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

    # red noise (powerlaw with 30 frequencies)
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)


    # gwb (no spatial correlations)
    cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)

    # change from 30 to 14 frequencies to line up with the 15 year GWB search
    curn = gp_signals.FourierBasisGP(spectrum=cpl, components=14, Tspan=Tspan, name='gw')

    # for HD spatial correlations you can do...
    # Note that this model is very slow compared to the CURN model above
    #orf=utils.hd_orf()
    #gwb = gp_signals.FourierBasisCommonGP(cpl, orf=utils.hd_orf(), components=14, Tspan=Tspan, name='gw')

    # timing model
    tm = gp_signals.MarginalizingTimingModel(use_svd=True)


    s = tm + mn + ec + rn + curn 

    models = []

    for p in psrs:
        models.append(s(p))

    pta = signal_base.PTA(models)

    pta.set_default_params(wn_params)

    del psrs
    del models 
    # set initial parameters drawn from prior
    x0 = np.hstack([p.sample() for p in pta.params])
    ndim = len(x0)
    print('number of parameters in the model:',ndim)

    
    posterior = {}
    mu0 = []
    sig0 = []
    if Rank == 0 :
        folder_path = './curn_100k/'
        #folder_path = './irn_100k/'
        params = np.genfromtxt(folder_path+'pars.txt',dtype=str,unpack=True)
        posterior = gss_mpi.read_post( folder_path, params, 700000)
        mu0 , sig0 = gss_mpi.ref_calibrate(posterior)
        del posterior
    mu0 = Comm.bcast(mu0,root = 0)
    sig0 = Comm.bcast(sig0,root = 0)

    start_time = time.time()
    n =  100
    nchain = Size #number of chain from N-core
    log_z = np.zeros(n)
    g_samp = 1000
    thin_fac = 100
    for i in range(n):
        
        GSS = gss_mpi.Gss_sampler(N_chain = nchain)

        log_z[i]= GSS.sample_is(n_samp = g_samp,mu0s = mu0 , sig0s= sig0, lnlikefn = pta.get_lnlikelihood, lnpriorfn = pta.get_lnprior, thin= thin_fac,parallel= True)
        z_path = '/fred/oz002/ezahraoui/nano_gss/paper_results/nsamp_comparison/curn_tf'+str(thin_fac)+'_'+str(g_samp)+'s_log_z_'+str(n)+'_'+str(nchain)+'T.txt'
        if Rank == 0 :
            np.savetxt(z_path, log_z, fmt='%1.14e')
    stop_time = time.time()
    if Rank == 0 :
        #ss_z = gss_mpi.SS(q_b)
        np.savetxt(z_path+'final', log_z, fmt='%1.14e')
        print("the GSS_IS istimated log(z):", log_z)
        #print("the SS_IS istimated log(z):", ss_z)
        
        print ("Time for GSS estimation ---", int((time.time()-start_time)), " seconds .---")