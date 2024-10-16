import glob, json, sys
import numpy as np

from enterprise.signals import parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise_extensions.models import model_singlepsr_noise

from enterprise_extensions import sampler as ee_sampler

from enterprise_extensions import model_utils

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from la_forge import core, diagnostics

from h5pulsar.pulsar import FilePulsar


import la_forge.diagnostics as dg
import la_forge.core as co
from la_forge.rednoise import plot_rednoise_spectrum, plot_free_spec
from la_forge.utils import epoch_ave_resid


datadir = './15yr_stochastic_analysis/tutorials/data'
psrs = []
for hdf5_file in glob.glob(datadir + '/hdf5/*.hdf5'):
    psrs.append(FilePulsar(hdf5_file))
print('Loaded {0} pulsars from hdf5 files'.format(len(psrs)))

psr = psrs[65]
tspan = model_utils.get_tspan([psr])

achrom_freqs = np.linspace(1/tspan,30/tspan,30)

pta = model_singlepsr_noise(psr, tm_var=False, tm_linear=False,
                          tmparam_list=None,
                          red_var=False, psd='powerlaw', red_select=None,
                          noisedict=None, tm_svd=True, tm_norm=True,
                          white_vary=True, components=14, upper_limit=False,
                          is_wideband=False, use_dmdata=False, tnequad=False,
                          dmjump_var=False, gamma_val=None, dm_var=False,
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

x0 = np.hstack([p.sample() for p in pta.params])
ndim = len(x0)


cov = np.diag(np.ones(ndim) * 0.01**2)

# set the location to save the output chains

outDir = './single_pulsar/'+psr.name+'_wn'


sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, 
                 outDir=outDir)

np.savetxt(outDir + '/achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')


N = int(1e6)
print('start sampling.')

x0 = np.hstack([p.sample() for p in pta.params])

sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, )


