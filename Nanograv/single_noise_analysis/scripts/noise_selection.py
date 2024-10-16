import glob
import numpy as np

from enterprise_extensions.models import model_singlepsr_noise

from enterprise_extensions import model_utils



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


    psr = psrs[36]
    tspan = model_utils.get_tspan([psr])

    achrom_freqs = np.linspace(1/tspan,30/tspan,30)

    pta = model_singlepsr_noise(psr, tm_var=False, tm_linear=False,
                          tmparam_list=None,
                          red_var=True, psd='powerlaw', red_select=None, #change RN
                          noisedict=None, tm_svd=True, tm_norm=True,
                          white_vary=True, components=14, upper_limit=False,
                          is_wideband=False, use_dmdata=False, tnequad=False,
                          dmjump_var=False, gamma_val=None, dm_var=False,  #change DM
                          dm_type='gp', dmgp_kernel='diag', dm_psd='powerlaw',
                          dm_nondiag_kernel='periodic', dmx_data=None,
                          dm_annual=False, gamma_dm_val=None,
                          dm_dt=15, dm_df=200,
                          chrom_gp=False, chrom_gp_kernel='nondiag',
                          chrom_psd='powerlaw', chrom_idx=4, chrom_quad=False,
                          chrom_kernel='periodic',
                          chrom_dt=15, chrom_df=200,
                          dm_expdip=False, dmexp_sign='negative',
                          dm_expdip_idx=2,
                          dm_expdip_tmin=None, dm_expdip_tmax=None,
                          num_dmdips=1, dmdip_seqname=None,
                          dm_cusp=False, dm_cusp_sign='negative',
                          dm_cusp_idx=2, dm_cusp_sym=False,
                          dm_cusp_tmin=None, dm_cusp_tmax=None,
                          num_dm_cusps=1, dm_cusp_seqname=None,
                          dm_dual_cusp=False, dm_dual_cusp_tmin=None,
                          dm_dual_cusp_tmax=None, dm_dual_cusp_sym=False,
                          dm_dual_cusp_idx1=2, dm_dual_cusp_idx2=4,
                          dm_dual_cusp_sign='negative', num_dm_dual_cusps=1,
                          dm_dual_cusp_seqname=None,
                          dm_sw_deter=False, dm_sw_gp=False,
                          swgp_prior=None, swgp_basis=None,
                          coefficients=False, extra_sigs=None,
                          psr_model=False, factorized_like=False,
                          Tspan=None, fact_like_gamma=13./3, gw_components=10,
                          fact_like_logmin=None, fact_like_logmax=None,
                          select='backend', tm_marg=True, dense_like=False, ng_twg_setup=False, wb_efac_sigma=0.25)

    print('Done pta model')
    del psrs
    posterior = {}
    mu0 = []
    sig0 = []
    model_id = '_rn'

    if Rank == 0 :
        folder_path = './single_pulsar/posterior/'+psr.name+model_id
        posterior = gss_mpi.read_post( folder_path,pta.param_names, 20000)
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
        z_path = './single_pulsar/paper_results/'+psr.name+model_id+'_'+str(g_samp)+'s_tf'+str(thin_fac)+'_'+str(n)+'log_z_'+str(nchain)+'T.txt'
        if Rank == 0 :
            np.savetxt(z_path, log_z, fmt='%1.14e')
    stop_time = time.time()
    if Rank == 0 :
        #ss_z = gss_mpi.SS(q_b)
        #np.savetxt(z_path+'final', log_z, fmt='%1.14e')
        print("the GSS_IS istimated log(z):", log_z)
        #print("the SS_IS istimated log(z):", ss_z)
        
        print ("Time for GSS estimation ---", int((time.time()-start_time)), " seconds .---")
