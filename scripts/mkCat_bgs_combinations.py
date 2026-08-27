#Script to read in existing LSS catalog information for some tracer and create a sub-sample
#Expectation is that users will copy this and create their own criteria, following the example

import sys
import os
import numpy as np
import fitsio
import argparse
from astropy.table import Table,join,unique,vstack
import LSS.common_tools as common
from astropy.coordinates import SkyCoord
import astropy.units as u
import logging
from pycorr.utils import setup_logging
logger = logging.getLogger('mkCat')

#LSS code, requires it git clone
import LSS.main.cattools as ct
import LSS.common_tools as common
from LSS.globals import main


parser = argparse.ArgumentParser()
parser.add_argument("--zmin",help="a redshift cut to apply when making clustering catalogs, default is to apply what is in globals.py", default=None, type=float)
parser.add_argument("--zmax",help="a redshift cut to apply when making clustering catalogs, default is to apply what is in globals.py", default=None, type=float)
parser.add_argument("--overwrite",help="whether to overwrite existing files", choices=['n', 'y'], default='n')

#arguments to find input data
parser.add_argument("--basedir", help="base directory for input, a versioning structure is expected under it", default='/global/cfs/cdirs/desi/survey/catalogs/')
parser.add_argument("--outdir", help="directory for output, the catalogs will be saved directly under it", default=os.environ['SCRATCH'])
parser.add_argument("--version", help="catalog version for input", default='v2')
parser.add_argument("--survey", help="e.g., Y1, DA2", default='DA2')
parser.add_argument("--verspec", help="version for redshifts", default='loa-v1')
parser.add_argument("--use_map_veto", help="string to include in full file name denoting whether map veto was applied", default='_HPmapcut')
#parser.add_argument("--extra_clus_dir", help="an optional extra layer of directory structure for clustering catalog",default='')

parser.add_argument("--compmd", choices=['not_altmtl', 'altmtl', 'n'], help="use altmtl to use PROB_OBS for completeness weights in clustering catalogs", default='not_altmtl')

#what steps to run (set all to y to get NGC/SGC clustering catalogs output)
parser.add_argument("--mkfulldat", choices=['n', 'y'], help="whether to make the initial cut file that gets used throughout", default='n')
parser.add_argument("--clusd", choices=['n', 'y'], help="make the 'clustering' catalog intended for paircounts", default='n')
parser.add_argument("--clusran", choices=['n', 'y'], help="make the random clustering files; these are cut to a small subset of columns", default='n')
parser.add_argument("--minr", help="minimum number for random files", default=0, type=int)
parser.add_argument("--maxr", help="maximum number for random files (plus one), 18 (0 through 17) are available (it is worth running all in parallel, see the option below)", default=18, type=int)
parser.add_argument("--nz", choices=['n', 'y'], help="get n(z) for type and all subtypes", default='n')
parser.add_argument("--splitGC", choices=['n', 'y'], help="convert to NGC/SGC catalogs", default='n')

#options for linear imaging systematic regressions
parser.add_argument("--imsys_clus", choices=['n', 'y'], help="add weights for imaging systematics using eboss method, applied to clustering catalogs?", default='n')
parser.add_argument("--imsys_clus_ran", choices=['n', 'y'], help="add weights for imaging systematics using eboss method, applied to clustering catalogs, to randoms?", default='n')
parser.add_argument("--nran4imsys", help="number of random files to using for linear regression", default=1, type=int)
parser.add_argument("--usemaps", help="the list of maps to use; defaults to what is set by globals", type=str, nargs='*', default=None)
parser.add_argument("--imsys_nside", help="healpix nside used for imaging systematic regressions", default=256, type=int)
parser.add_argument("--imsys_zbin", choices=['n', 'y', 'hi'], help="use separate redshift bins for imaging systematic regressions? (or a wider redshift range for BGS)", default='n')

parser.add_argument("--par", choices=['y', 'n'], help="run different random numbers in parallel?", default='y')

#options for the third-level (property) adaptive binning, applied within each mag bin x quiescent/star-forming split
parser.add_argument("--splitprop", help="name of the column to use for the adaptive sub-binning done within each mag bin x quiescent/star-forming split", type=str, required=True)
parser.add_argument("--splitpropfile", help="path to a file with TARGETID and the --splitprop column to merge onto the group catalog", type=str, default='/global/cfs/cdirs/desi/users/ianw89/private/DATA/BGS_LOA/ian_BGS_Y3_merged.fits')

setup_logging()

from astropy.cosmology import z_at_value, FlatLambdaCDM
from astropy import units as u
_cosmo_h = FlatLambdaCDM(H0=100, Om0=0.315192, Ob0=0.045, Tcmb0=2.725, Neff=3.04) 
def get_max_observable_z(abs_mags, fluxlimit): # this should use un-kcorrected absolute magnitudes in H=100 cosmology
    d_l = (10 ** ((fluxlimit - abs_mags + 5) / 5)) / 1e6 # luminosity distance in Mpc
    return z_at_value(_cosmo_h.luminosity_distance, d_l*u.Mpc) # Use this cosmology because it needs to match the absolute magnitude calculation cosmology (fastspecfit)

#function to take a file and split it NGC/SGC
def splitGC(flroot,datran='.dat',rann=0):
    app = 'clustering'+datran+'.fits'
    if datran == '.ran':
        app = str(rann)+'_clustering'+datran+'.fits'

    outf_ngc = flroot+'NGC_'+app
    outf_sgc = flroot+'SGC_'+app
    
    # If they both already exist we we aren't ovewriting, return early
    if os.path.exists(outf_ngc) and os.path.exists(outf_sgc) and args.overwrite != 'y':
        return

    fn = Table(fitsio.read(flroot +app))
    sel_ngc = common.splitGC(fn)#gc.b > 0
    common.write_LSS_scratchcp(fn[sel_ngc],outf_ngc,logger=logger)
    common.write_LSS_scratchcp(fn[~sel_ngc],outf_sgc,logger=logger)


args = parser.parse_args()
args.input_tracer = 'BGS_BRIGHT' 
common.printlog(str(args),logger)

basedir = args.basedir
survey = args.survey
version = args.version
specrel = args.verspec
rm = int(args.minr)
rx = int(args.maxr)

#get some parameters that depend on the tracer
mainp = main(args.input_tracer,args.verspec,survey=survey) #parameters contained in mainp

dchi2 = mainp.dchi2 #used for good redshift selection when making clustering catalogs
#clustering catalogs will have these redshift bounds:

if survey != 'SV3' and survey != 'Y1' and survey != 'DA2' and survey != 'Y3':
    common.printwarn('survey not recognized, directory structure may not be regonized', logger)

#set up input directory
maindir = basedir +'/'+survey+'/LSS/'
ldirspec = maindir+specrel+'/'
dirin = ldirspec+'LSScats/'+version+'/'
if version.endswith('pip'):
    lssmapsdir = ldirspec+'LSScats/'+version.rstrip("pip")+'/'+'/hpmaps/' #maps for imaging systematics regressions
else:
    lssmapsdir = dirin+'/hpmaps/' #maps for imaging systematics regressions
    
if not os.path.exists(dirin):
    sys.exit('issue with '+dirin+' it does not exist')

#parse arguments to toggle options
mkclusdat = False
mkclusran = False
if args.clusd == 'y':
    mkclusdat = True
    common.printlog('Will make clustering catalog for data',logger)
if args.clusran == 'y':
    mkclusran = True
    common.printlog('Will make clustering catalog for randoms, files '+str(rm)+ ' through '+str(rx),logger)
    common.printlog('(if running all, consider doing in parallel)',logger)  

tracer_out = args.input_tracer

rand_tbls = []
if mkclusran:
    ran_cols = ['RA','DEC','TARGETID','TILEID','NTILE','PHOTSYS']
    if args.compmd != 'altmtl':
        ran_cols.append('FRAC_TLOBS_TILES')
    for i in range(rm,rx):
        ranin = dirin + args.input_tracer +'_' +str(i)+'_full'+args.use_map_veto+'.ran.fits'
        ran_tbl = Table(fitsio.read(ranin, columns=ran_cols))
        common.printlog(f'{len(ran_tbl)} random rows read from {ranin}', logger)
        rand_tbls.append(ran_tbl)

fulldat = Table(fitsio.read(dirin+args.input_tracer+'_full'+args.use_map_veto+'.dat.fits'))
common.printlog(f'{len(fulldat)} full data rows read from {dirin+args.input_tracer}_full{args.use_map_veto}.dat.fits', logger)

# Use group catalog to get centrals and quality cuts
path = '/global/cfs/cdirs/desi/users/ianw89/groupcatalogs/BGS_Y3/v0.8/GROUP_CATALOG_BGS_Y3_1PASS_v0.8.fits'
grpcat = Table(fitsio.read(path, columns=['TARGETID', 'Z_ASSIGNED_FLAG', 'ABS_MAG_R', 'G_R', 'IS_SAT', 'QUIESCENT']))
common.printlog(f'{len(grpcat)} group catalog rows read from {path}', logger)

sel = grpcat['Z_ASSIGNED_FLAG'] == 0 # DESI spectroscopic data only
sel &= ~grpcat['IS_SAT'] # Centrals only
grpcat = grpcat[sel]
grpcat.remove_columns(['Z_ASSIGNED_FLAG', 'IS_SAT'])
common.printlog(f'{len(grpcat)} group catalog rows after cuts (DESI spec + centrals only)', logger)

# Merge with fulldat, inner join
fulldat = join(fulldat, grpcat, keys=['TARGETID'], join_type="inner")
common.printlog(f'{len(fulldat)} full data rows after joining with group catalog', logger)

# Now read in merged file and get the extra column we need for the third-level split, if it's not in fulldat already
if args.splitprop not in fulldat.colnames:
    extra = Table(fitsio.read(args.splitpropfile, columns=['TARGETID', args.splitprop]))
    common.printlog(f'{len(extra)} rows read from {args.splitpropfile} for property {args.splitprop}', logger)
    fulldat = join(fulldat, extra, keys=['TARGETID'], join_type="inner")
    common.printlog(f'{len(fulldat)} full data rows after joining with {args.splitprop} property file', logger)

# Use my automated binnings strategy
mag_bins = common.make_adaptive_density_bins(fulldat['ABS_MAG_R'], n_bins=10, n_tail=0, alpha=0.5, limit=0.01)
mag_bins = np.round(mag_bins, 2)

def get_binned_mean(data, bin_edges):
    """Bin the data using the provided edges and compute the mean value in each bin."""
    
    indices = np.digitize(data, bin_edges)
    sums = np.bincount(indices, weights=data, minlength=len(bin_edges) + 1)
    counts = np.bincount(indices, minlength=len(bin_edges) + 1)
    means = sums / np.where(counts == 0, 1, counts)
    means[counts == 0] = np.nan

    # Ignore data to the left of leftmost edge and right of rightmost edge
    return means[1:len(bin_edges)]

mag_bins_means = get_binned_mean(fulldat['ABS_MAG_R'], mag_bins)

# Save off the numpy array of mag_bins
np.save(args.outdir+'/'+args.input_tracer+'_mag_bins.npy', mag_bins)
np.save(args.outdir+'/'+args.input_tracer+'_mag_bins_means.npy', mag_bins_means)

for i in range(len(mag_bins)-1):
    mag_mask = (fulldat['ABS_MAG_R'] >= mag_bins[i]) & (fulldat['ABS_MAG_R'] < mag_bins[i+1])
    dat_inmagbin = fulldat[mag_mask]

    for is_quiescent in [True, False]:
        qsf_label = 'Q' if is_quiescent else 'SF'
        q_mask = dat_inmagbin['QUIESCENT'] == is_quiescent
        dat_inqbin = dat_inmagbin[q_mask]
        # Some properties we should go all the way to the extremal values, but others we should not
        limit = 0.01

        prop_in_bin = dat_inqbin[args.splitprop]

        if args.splitprop == 'c9050':
            limit = 0.0
        if args.splitprop == 'LOGSSFR':
            dat_inqbin[args.splitprop][prop_in_bin < -16] = -16 # A floor value of -16 for finding bins since its ~0 SFR when this low, I assume differences are noise

        # Remove nans
        dat_inqbin = dat_inqbin[~np.isnan(prop_in_bin)]
        prop_in_bin = dat_inqbin[args.splitprop]

        # Check how much data is in this mag x QvsSF bin and print it
        common.printlog(f'Mag bin {i}: {mag_bins[i]:.2f} to {mag_bins[i+1]:.2f}, ({qsf_label}) count: {len(prop_in_bin)}', logger)

        # Faintest Q bin gets ~10000 galaxies, and the uneven binning can make some samples so small (< 1000 or so) that
        # pycorr crashes. Let's just go for 3 bins in that case
        if len(prop_in_bin) < 15000:
            n_bins = 3
        else:
            n_bins = 5

        prop_bins = common.make_adaptive_density_bins(prop_in_bin, n_bins=n_bins, n_tail=0, alpha=0.5, limit=limit)

        if args.splitprop == 'LOGSSFR':
            # Change the label for the leftmost bin to -99 instead of ~ -16 for clarity
            prop_bins[0] = -99

        prop_bins = np.round(prop_bins, 3)
        prop_bins_means = get_binned_mean(prop_in_bin, prop_bins)

        np.save(args.outdir+'/'+args.input_tracer+'_'+args.splitprop+'_bins_formagbin'+str(i)+'_'+qsf_label+'.npy', prop_bins)
        np.save(args.outdir+'/'+args.input_tracer+'_'+args.splitprop+'_bins_means_formagbin'+str(i)+'_'+qsf_label+'.npy', prop_bins_means)

        for j in range(len(prop_bins)-1):
            prop_mask = (prop_in_bin >= prop_bins[j]) & (prop_in_bin < prop_bins[j+1])

            common.printlog(f'Mag bin {i}: {mag_bins[i]:.4f} to {mag_bins[i+1]:.4f}, {qsf_label}, {args.splitprop} bin {j}: {prop_bins[j]:.4f} to {prop_bins[j+1]:.4f}', logger)

            dat_inallbins = dat_inqbin[prop_mask]

            # When calling xirunpc, a zmin and zmax is also provided. Should that cut just happen there since it's N vs S specific?
            zmin = 0.001 # Reference measurement uses 0.005 but there is no need to match. For my faintest bin it only goes up to 0.005 zmax
            zmax_S = float(get_max_observable_z(mag_bins[i+1], 19.5).round(4))
            zmax_N = float(get_max_observable_z(mag_bins[i+1], 19.54).round(4))
            zmax = min(zmax_N, zmax_S)
            zmax_arr = np.where(dat_inallbins['PHOTSYS'] == b'N', zmax_N, zmax_S)

            dz = zmax - zmin # TODO simplification for n(z) calculation...
            common.printlog(f'  zmin = {zmin}, zmax_S = {zmax_S}, zmax_N = {zmax_N}', logger)

            #write output to new "full" catalog at your defined location
            tracer_out = args.input_tracer + "_CEN" + f'_mag{mag_bins[i]:.2f}to{mag_bins[i+1]:.2f}' + f'_{qsf_label}' + f'_{args.splitprop}{prop_bins[j]:.3f}to{prop_bins[j+1]:.3f}'
            fout = args.outdir+'/'+tracer_out+'_full'+args.use_map_veto+'.dat.fits'

            if args.mkfulldat == 'y':
                if os.path.exists(fout) and args.overwrite == 'n':
                    common.printlog(f'Output file {fout} already exists, skipping', logger)
                else:
                    common.write_LSS_scratchcp(dat_inallbins, fout, logger=logger)

            #create "clustering" catalogs for data with no NGC/SGC split or FKP weights 
            #needs to happen before randoms so randoms can get z and weights
            weightileloc=True
            if args.compmd == 'altmtl':
                weightileloc = False
            if mkclusdat:
                if os.path.exists(args.outdir+'/'+tracer_out+'_clustering.dat.fits') and args.overwrite == 'n':
                    common.printlog(f'clustering catalog {args.outdir}/{tracer_out}_clustering.dat.fits already exists, skipping', logger)
                else:
                    # Pass in data so it doesn't bother re-reading what we just wrote
                    ct.mkclusdat(args.outdir+'/'+tracer_out, weightileloc, tp=tracer_out, dchi2=dchi2, zmin=zmin, zmax=zmax_arr, use_map_veto=args.use_map_veto, data=dat_inallbins)

            nzcompmd = 'ran'
            if args.compmd == 'altmtl':
                nzcompmd = args.compmd
            # Columns to resample from the data
            rcols=['Z','WEIGHT','WEIGHT_SYS','WEIGHT_COMP','WEIGHT_ZFAIL','TARGETID_DATA'] #columns to make sure are in the randoms
            inds = np.arange(rm,rx)

            #make clustering catalogs for randoms
            if mkclusran:
                clus_arrays = [fitsio.read(args.outdir+'/'+tracer_out+'_clustering.dat.fits')]

                def _parfun_cr(ii):
                    ranin = rand_tbls[ii]
                    ranout = args.outdir+'/'+tracer_out+'_'+str(ii)+'_clustering.ran.fits'
                    if os.path.exists(ranout) and args.overwrite == 'n':
                        common.printlog(f'ranout {ranout} already exists, skipping', logger)
                        return
                    ct.mkclusran(ranin,args.outdir+'/'+tracer_out+'_',ii,rcols=rcols,clus_arrays=clus_arrays,use_map_veto=args.use_map_veto,compmd=nzcompmd,logger=logger,tp=args.input_tracer)
                
                if args.par == 'y':
                    from multiprocessing import Pool
                    with Pool() as pool:
                        res = pool.map(_parfun_cr, inds)

                else:
                    for ii in inds:#range(rm,rx):
                        _parfun_cr(ii)


            nran = rx-rm
            regions = ['NGC', 'SGC']

            dirout = args.outdir #just because original copied code used dirout

            #split catalogs NGC/SGC
            # this reads in previously written cluster cat, maybe don't do that to save I/O TODO
            if args.splitGC == 'y':
                fb = dirout+'/'+tracer_out+'_'
                splitGC(fb,'.dat')
                def _spran(rann):
                    splitGC(fb,'.ran',rann)
                inds = np.arange(nran)
                if args.par == 'y':
                    from multiprocessing import Pool
                    with Pool() as pool:
                        res = pool.map(_spran, inds)
                else:
                    for rn in inds:#range(rm,rx):
                        _spran(rn)

            #this 1) calculates the n(z) for the sample
            #2) adds the n(z) as a function of completeness to the catalogs (column NX)
            #3) calculates the FKP weights based on NX
            #4) refactors the weights (section 8.2 of the KP3 paper arXiv:2411.12020)
            #5) writes the NGC/SGC clustering catalogs back out for data and randoms
            #It needs to take inputs for completeness and mean weight as a function of NTILE, of the non-subsampled catalog, so that the angular upweighting option can remain consistent
            def get_ntile_info(clus_orig_fname):
                fd = fitsio.read(clus_orig_fname, columns=['NTILE', 'WEIGHT_COMP', 'FRAC_TLOBS_TILES'])
                weight_ntl = np.bincount(fd['NTILE']-1, weights=fd['WEIGHT_COMP']) / np.bincount(fd['NTILE']-1) # mean of WEIGHT_COMP for each (positive integer) NTILE in the data. Note that the NTILE values are shifted down by 1 to avoid guaranteed division by zero for NTILE=0
                comp_ntl = 1 / weight_ntl # the completeness is the inverse of the mean weight (for each NTILE). Indexed by NTILE-1
                if args.compmd != 'altmtl':
                    fttl = np.bincount(fd['NTILE']-1, weights=fd['FRAC_TLOBS_TILES']) / np.bincount(fd['NTILE']-1) # mean of FRAC_TLOBS_TILES for each (positive integer) NTILE in data (although shouldn't this be computed in randoms?). Note that the NTILE values are shifted down by 1 to avoid guaranteed division by zero for NTILE=0
                    comp_ntl *= fttl # if not using altmtl, also multiply by the mean FRAC_TLOBS_TILES for each NTILE to get the completeness. Both are indexed by NTILE-1
                return comp_ntl, weight_ntl # both are indexed by NTILE-1 as common.addnbar expects
            
            if args.nz == 'y':
                for reg in regions:#allreg:
                    #file names
                    fb = dirout+'/'+tracer_out+'_'+reg
                    fcr = fb+'_0_clustering.ran.fits'
                    fcd = fb+'_clustering.dat.fits'
                    fout = fb+'_nz.txt'

                    # If fout already exists and we aren't overwriting, skip making n(z)
                    if os.path.exists(fout) and args.overwrite != 'y':
                        continue

                    #make n(z)
                    common.mknz(fcd,fcr,fout,bs=dz,zmin=zmin,zmax=zmax,compmd=nzcompmd)
                    #do steps 2-5 above
                    # For Y3 this is dir structure
                    if survey == 'DA2':
                        extra_dir = 'nonKP'
                        if args.compmd == 'altmtl':
                            extra_dir = 'PIP'
                        clus_orig = dirin+'/'+extra_dir+'/'+args.input_tracer+'_'+reg+'_clustering.dat.fits'
                    else:
                        clus_orig = dirin+'/'+args.input_tracer+'_clustering.dat.fits'
                    comp_ntl, weight_ntl = get_ntile_info(clus_orig)
                    common.addnbar(fb,bs=dz,zmin=zmin,zmax=zmax,P0=7000,nran=nran,par=args.par,compmd=nzcompmd,comp_ntl=comp_ntl,weight_ntl=weight_ntl,logger=logger)

            # determine linear weights for imaging systematics
            # this is new for doing after the fact based on clustering catalogs
            skip_imsys = False
            if args.imsys_clus == 'y':
                zmin_imaging = 0.001
                zmax_imaging = 1.0

                figname1 = dirout+'/'+tracer_out+'_N_'+str(zmin_imaging)+str(zmax_imaging)+'_linclusimsysfit.png'
                figname2 = dirout+'/'+tracer_out+'_S_'+str(zmin_imaging)+str(zmax_imaging)+'_linclusimsysfit.png'

                if os.path.exists(figname1) and os.path.exists(figname2) and args.overwrite != 'y':
                    common.printlog('Imaging systematic weight figures already exist and not overwriting; will skip.', logger)
                    skip_imsys = True

                if skip_imsys:
                    continue

                common.printlog('Calculating imaging systematic weights' ,logger)
                #import package
                from LSS.imaging import densvar 

                #get maps to regress against
                if args.usemaps == None:
                    fit_maps = mainp.fit_maps
                else:
                    fit_maps = [mapn for mapn in args.usemaps]

                #get NGC/SGC catalogs and stack them (weights will be fit splitting the data into different photometric regions)
                fname = os.path.join(dirout, tracer_out+'_NGC_clustering.dat.fits')
                dat_ngc = Table(fitsio.read(fname))
                fname = os.path.join(dirout, tracer_out+'_SGC_clustering.dat.fits')
                dat_sgc = Table(fitsio.read(fname))
                dat = vstack([dat_sgc,dat_ngc])

                #get randoms
                ranl = []
                for iran in range(0,args.nran4imsys):
                    ran = fitsio.read(os.path.join(dirout, tracer_out+'_NGC_'+str(iran)+'_clustering.ran.fits')) 
                    ranl.append(ran)
                    ran = fitsio.read(os.path.join(dirout, tracer_out+'_SGC_'+str(iran)+'_clustering.ran.fits')) 
                    ranl.append(ran)
                rands = np.concatenate(ranl)
                
                #fiducial column name for weights, initialize as 1.
                syscol = 'WEIGHT_IMLIN_CLUS' 
                dat[syscol] = np.ones(len(dat))
                #photometric regions
                regl = ['S','N']
                if args.input_tracer == 'QSO':
                    regl = ['DES','SnotDES','N']
                
                #do fit looping over regions and redshift bins
                for reg in regl:
                    #handling for loading maps of potential systematics to regress against
                    regu = reg
                    if reg == 'DES' or reg == 'SnotDES':
                        regu = 'S'
                    mptr = args.input_tracer
                    if args.input_tracer[:3] == 'BGS':
                        mptr = 'BGS_BRIGHT' #bright and any/faint cover the same footprint so the same map is used for both
                    pwf = lssmapsdir+mptr+'_mapprops_healpix_nested_nside256_'+regu+'.fits'
                    sys_tab = Table.read(pwf)
                    cols = list(sys_tab.dtype.names)
                    for col in cols:
                        #apply extinction corrections to depth to get total depth estimate
                        if 'DEPTH' in col:
                            bnd = col.split('_')[-1]
                            sys_tab[col] *= 10**(-0.4*common.ext_coeff[bnd]*sys_tab['EBV'])
                    #Delta EBV using DESI EBV maps
                    debv = common.get_debv()
                    for ec in ['GR','RZ']:
                        if 'EBV_DIFF_'+ec in fit_maps: 
                            sys_tab['EBV_DIFF_'+ec] = debv['EBV_DIFF_'+ec]
                    
                    selr = rands['PHOTSYS'] == reg

                    common.printlog('Getting weights for region '+reg+' and '+str(zmin_imaging)+'<z<'+str(zmax_imaging), logger)
                    if args.input_tracer == 'LRG' and args.usemaps == None:
                        if reg == 'N':
                            fitmapsbin = fit_maps
                        else:
                            if zmax == 0.6:
                                fitmapsbin = mainp.fit_maps46s
                            if zmax == 0.8:
                                fitmapsbin = mainp.fit_maps68s
                            if zmax == 1.1:
                                fitmapsbin = mainp.fit_maps81s
                    else:
                        fitmapsbin = fit_maps
                    use_maps = fitmapsbin
                    #now, everything in place to actually perform regression for this reg and zbin
                    figname = dirout+'/'+tracer_out+'_'+reg+'_'+str(zmin_imaging)+str(zmax_imaging)+'_linclusimsysfit.png'
                    wsysl = densvar.get_imweight(dat,rands,zmin_imaging,zmax_imaging,reg,fitmapsbin,use_maps,sys_tab=sys_tab,zcol='Z',modoutname = dirout+'/'+tracer_out+'_'+reg+'_'+str(zmin_imaging)+str(zmax_imaging)+'_linfitparam.txt',figname=figname,wtmd='clus', logger=logger)
                    #we want to update the weights for the selection of data just input to the regression
                    sel = wsysl != 1 
                    common.printlog(f'sel sum {np.sum(sel)} out of len {len(wsysl)}',logger)
                    dat[syscol][sel] = wsysl[sel]
                #attach data to NGC/SGC catalogs, write those out
                #we will do a join
                dat.keep_columns(['TARGETID',syscol])
                
                if syscol in dat_ngc.colnames:
                    dat_ngc.remove_column(syscol)
                dat_ngc = join(dat_ngc,dat,keys=['TARGETID'])
                #apply weight to final weight columns, remove any previous weighting
                dat_ngc['WEIGHT_SYS_OLD'] = dat_ngc['WEIGHT_SYS'] #keep the old sys weight
                dat_ngc['WEIGHT'] /= dat_ngc['WEIGHT_SYS']
                dat_ngc['WEIGHT_SYS'] = dat_ngc[syscol]
                dat_ngc['WEIGHT'] *= dat_ngc['WEIGHT_SYS']
                #write out NGC
                common.write_LSS_scratchcp(dat_ngc,os.path.join(dirout, tracer_out+'_NGC_clustering.dat.fits'),logger=logger)
                #do SGC
                if syscol in dat_sgc.colnames:
                    dat_sgc.remove_column(syscol)
                dat_sgc = join(dat_sgc,dat,keys=['TARGETID'])
                #apply weight to final weight columns
                dat_sgc['WEIGHT_SYS_OLD'] = dat_sgc['WEIGHT_SYS'] #keep the old sys weight
                dat_sgc['WEIGHT'] /= dat_sgc['WEIGHT_SYS']
                dat_sgc['WEIGHT_SYS'] = dat_sgc[syscol]
                dat_sgc['WEIGHT'] *= dat_sgc['WEIGHT_SYS']
                #write out SGC
                common.write_LSS_scratchcp(dat_sgc,os.path.join(dirout, tracer_out+'_SGC_clustering.dat.fits'),logger=logger)

            #column needs to be added to randoms
            if args.imsys_clus_ran == 'y' and not skip_imsys:
                #do randoms
                syscol = 'WEIGHT_IMLIN_CLUS'
                fname = os.path.join(dirout, tracer_out+'_NGC_clustering.dat.fits')
                dat_ngc = Table(fitsio.read(fname,columns=['TARGETID',syscol]))
                fname = os.path.join(dirout, tracer_out+'_SGC_clustering.dat.fits')
                dat_sgc = Table(fitsio.read(fname,columns=['TARGETID',syscol]))
                dat = vstack([dat_sgc,dat_ngc])
                dat.rename_column('TARGETID','TARGETID_DATA') #randoms have their weights modulated based on the data used for the redshift
                regl = ['NGC','SGC']
                def _add2ran(rann):
                    for reg in regl:
                        ran_fn = os.path.join(dirout, tracer_out+'_'+reg+'_'+str(rann)+'_clustering.ran.fits')
                        ran = Table(fitsio.read(ran_fn))
                        if syscol in ran.colnames:
                            ran.remove_column(syscol)
                        ran = join(ran,dat,keys=['TARGETID_DATA'])
                        ran['WEIGHT_SYS_OLD'] = ran['WEIGHT_SYS'] #keep the old sys weight
                        ran['WEIGHT'] /= ran['WEIGHT_SYS'] #remove effect of any original weighting
                        ran['WEIGHT'] *= ran[syscol]
                        ran['WEIGHT_SYS'] = ran[syscol]
                        common.write_LSS_scratchcp(ran,ran_fn,logger=logger)

                if args.par == 'y':
                    from multiprocessing import Pool
                    with Pool() as pool:
                        res = pool.map(_add2ran, inds)
                else:
                    for rn in inds:#range(rm,rx):
                        _add2ran(rn)


common.printlog('Script Finished',logger)