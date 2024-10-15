import numpy as np
import os, glob, json
import optparse

import enterprise
from enterprise.pulsar import Pulsar

from enterprise.signals.parameter import function
from enterprise.signals.deterministic_signals import Deterministic
from enterprise.signals import parameter, selections
from enterprise.signals.selections import Selection

from enterprise_extensions import models
import time
from mpi4py import MPI
import gss_mpi

if __name__ == "__main__":
    Comm = MPI.COMM_WORLD
    Rank = Comm.Get_rank()
    Size = Comm.Get_size()
    print('imported Enterprise.')
    parser = optparse.OptionParser()
    parser.add_option('--datadir', action='store', dest='datadir', default='./', type='string')
    parser.add_option('--outdir', action='store', dest='outdir', default='./report/gwb_run/', type='string')
    parser.add_option('--noisedir', action='store', dest='noisedir', default='./noisefiles/', type='string')
    parser.add_option('--orf', action='store', dest='orf', default='crn', type='string')
    parser.add_option('--orf_bins', action='store', dest='orf_bins', default=None, type='string')
    parser.add_option('--common_psd', action='store', dest='common_psd', default='powerlaw', type='string')
    parser.add_option('--common_components', action='store', dest='common_components', default=30, type='int')
    parser.add_option('--gamma_common', action='store', dest='gamma_common', default=None, type='float')
    parser.add_option('--red_components', action='store', dest='red_components', default=0, type='int')
    parser.add_option('--dm_components', action='store', dest='dm_components', default=0, type='int')
    parser.add_option('--chrom_components', action='store', dest='chrom_components', default=0, type='int')
    parser.add_option('--num_dmdips', action='store', dest='num_dmdips', default=2, type='int')
    parser.add_option('--bayesephem', action='store_true', dest='bayesephem', default=False)
    parser.add_option('--common_sin', action='store_true', dest='sin_wave', default=False)
    parser.add_option('--psrname', action='store', dest='psrname', default=None, type='string')
    parser.add_option('--resume', action='store_true', dest='resume', default=False)
    parser.add_option('--emp', action='store', dest='emp', default=None, type='string')
    parser.add_option('--number', action='store', dest='num', default=1e7, type='float')
    parser.add_option('--thin', action='store', dest='thin', default=100, type='int')
    (options,args) = parser.parse_args()

    # load par and tim files
    datadir = options.datadir
    outdir = options.outdir
    noisedir = options.noisedir

    parfiles = sorted(glob.glob(datadir + '/J*/*.par'))
    timfiles = sorted(glob.glob(datadir + '/J*/*_all.tim'))
    noisefiles = sorted(glob.glob(noisedir + '/*.json'))
    # filter to one set of par+tim+noisefile per pulsar
    PsrList = np.loadtxt(datadir + 'psrlist.txt',dtype=str)

    parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in PsrList]
    timfiles = [x for x in timfiles if x.split('/')[-1].split('_')[0] in PsrList]
    noisefiles = [x for x in noisefiles if x.split('/')[-1].split('_')[0] in PsrList]

    psrs = []
    for p, t in zip(parfiles, timfiles):
        psr = Pulsar(p, t, ephem='DE440')
        psrs.append(psr)
    
    # set reference time for the sin wave to the earliest TOA in the data set
    dataset_tmin = np.min([p.toas.min() for p in psrs])
    dataset_tmax = np.max([p.toas.max() for p in psrs])

    # sin wave model
    @function
    def sine_wave(toas, flags, A = -9, f = -9, phase = 0.0):
        return 10 ** A * np.sin(2 * np.pi * (10 ** f) * (toas - dataset_tmin) + phase)
    
    def sine_signal(A, f, phase, selection=Selection(selections.no_selection), name = ""):
        return Deterministic(sine_wave(A = A, f = f, phase = phase), selection = selection, name = name)

    if options.sin_wave:
        m1 = sine_signal(A = parameter.Uniform(-9, -4)('common_sin_log10_A'),
                     f = parameter.Uniform(-9, -7.7)('common_sin_log10_f'),
                     phase = parameter.Uniform(0, 2 * np.pi)('common_sin_phase'), name='common_sin')
    else:
        m1 = None

    # load noise models and files
    params = {}
    for nf in noisefiles:
        with open(nf, 'r') as fin:
            params.update(json.load(fin))

    if not options.red_components:
        try:
            red_dict = {}
            with open(noisedir + 'red_dict.json','r') as rd:
                red_dict.update(json.load(rd))
        except:
            raise UserWarning('Custom pulsar red noise frequency components not set.')
    else:
        red_dict = options.red_components

    if not options.dm_components:
        try:
            dm_dict = {}
            with open(noisedir + 'dm_dict.json','r') as dd:
                dm_dict.update(json.load(dd))
        except:
            raise UserWarning('Custom pulsar DM noise frequency components not set.')
    else:
        dm_dict = options.dm_components

    if not options.chrom_components:
        try:
            chrom_dict = {}
            with open(noisedir + 'chrom_dict.json','r') as cd:
                chrom_dict.update(json.load(cd))
        except:
            raise UserWarning('Custom pulsar scattering noise frequency components not set.')
    else:
        chrom_dict = options.chrom_components

    try:
        gamma_common = float(options.gamma_common)
    except:
        gamma_common = None

    if options.psrname is not None:
        dropout = True
    else:
        dropout = False

    if options.orf_bins is not None:
        orf_bins = np.loadtxt(options.orf_bins)
    else:
        orf_bins = None

    # setup model
    pta = models.model_general(psrs, noisedict=params, orf=options.orf,
                           orf_bins=orf_bins, common_psd=options.common_psd,
                           common_components=options.common_components,
                           gamma_common=gamma_common,
                           bayesephem=options.bayesephem,
                           sat_orb_elements=True, tnequad=True,
                           tm_svd=True, tm_marg=True,
                           red_var=True, red_components=red_dict,
                           dm_var=True, dm_components=dm_dict,
                           chrom_var=True, chrom_components=chrom_dict,
                           chrom_kernel='diag', tndm=True,
                           num_dmdips=options.num_dmdips,
                           dmpsr_list=['J1713+0747'], dm_expdip_idx=[1,4],
                           dm_expdip_tmin=[57490,54650],
                           dm_expdip_tmax=[57530,54850], extra_sigs=m1,
                           dropout=dropout, dropout_psr=options.psrname)

    # initial sample, set zeros for the initial ORF
    x = [np.array(p.sample()).ravel().tolist() for p in pta.params]
    x0 = np.array([p for sublist in x for p in sublist])
    if options.orf == 'bin_orf':
        if orf_bins is not None:
            orf_idx = len(orf_bins)-1
        else:
            orf_idx = 7
    elif options.orf == 'chebyshev_orf':
        orf_idx = 4
    elif options.orf == 'legendre_orf':
        orf_idx = 6
    else:
        pass
    if 'orf' in options.orf:
        for i in range(orf_idx):
            if options.bayesephem:
                x0[-i-13] = 0.
            else:
                x0[-i-1] = 0.

    del psrs
    mu0 = []
    sig0 = []
    model_name = 'curn'
    folder_path = '/fred/oz103/ezahraoui/epta_posteriors/DR2new/'+model_name+'/'
    if Rank == 0 :
        posterior = {}
        params = np.genfromtxt(folder_path+'pars.txt',dtype=str,unpack=True)
        posterior = gss_mpi.read_post( folder_path, params, 50000)
        mu0 , sig0 = gss_mpi.ref_calibrate(posterior)
        del posterior
    mu0 = Comm.bcast(mu0,root = 0)
    sig0 = Comm.bcast(sig0,root = 0)

    start_time = time.time()
    n =  100
    nchain = Size #number of chain from N-core
    log_z = np.zeros(n)
    g_samp = 2000
    thin_fac = 100
    new_path = folder_path.replace(model_name+'/','paper_results/')
    for i in range(n):
        
        GSS = gss_mpi.Gss_sampler(N_chain = nchain)

        log_z[i]= GSS.sample_is(n_samp = g_samp,mu0s = mu0 , sig0s= sig0, lnlikefn = pta.get_lnlikelihood, lnpriorfn = pta.get_lnprior, thin= thin_fac,parallel= True)
        z_path = new_path+model_name+'_'+str(g_samp)+'s_tf'+str(thin_fac)+'_'+str(n)+'_log_z_'+str(nchain)+'T.txt'
        if Rank == 0 :
            np.savetxt(z_path, log_z, fmt='%1.14e')
    stop_time = time.time()
    if Rank == 0 :
        #ss_z = gss_mpi.SS(q_b)
        #np.savetxt(z_path+'final', log_z, fmt='%1.14e')
        print("the GSS_IS istimated log(z):", log_z)
        #print("the SS_IS istimated log(z):", ss_z)
        
        print ("Time for GSS estimation ---", int((time.time()-start_time)), " seconds .---")