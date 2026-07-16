#Script to read in existing LSS catalog information for some tracer and create a sub-sample
#Expectation is that users will copy this and create their own criteria, following the example

# Ian Example:
#srun -N 1 -C cpu -t 04:00:00 --qos interactive --account desi python scripts/mkCat_subsamp_minef.py --input_tracer BGS_BRIGHT --mkfulldat y --clusd y --clusran y --nz y --splitGC y --ccut FSFABSmag-R-19-20 --extracut=Q  --survey Y1 --verspec iron --version v1.5pip  --imsys_clus y --imsys_clus_ran y --outdir /global/cfs/cdirs/desi/users/ianw89/newclustering/Y1/LSS/iron/LSScats/v1.5pip

import sys
import os
import numpy as np
import fitsio
import argparse
from astropy.table import Table,join,unique,vstack

import logging
from pycorr.utils import setup_logging
logger = logging.getLogger('mkCat')

#LSS code, requires it git clone
import LSS.main.cattools as ct
import LSS.common_tools as common
from LSS.globals import main


parser = argparse.ArgumentParser()
parser.add_argument("--centrals",help="True/False", default='n')
parser.add_argument("--extracut",help="Q (quiescent) or SF (star-forming) or ALL", default='ALL')
parser.add_argument("--passcut",help="a value of NTILE_MINE to cut to", default=0, type=int)
parser.add_argument("--ccut",help="a string that is used define your subsample", default='') 
parser.add_argument("--zmin",help="a redshift cut to apply when making clustering catalogs, default is to apply what is in globals.py", default=None, type=float)
parser.add_argument("--zmax",help="a redshift cut to apply when making clustering catalogs, default is to apply what is in globals.py", default=None, type=float)
parser.add_argument("--pcut",help="another gal property to cut on", default='') #COLUMNNAME-LOWERBOUND-UPPERBOUND

#arguments to find input data
parser.add_argument("--input_tracer", help="tracer type that subsample will come from", required=True)
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

setup_logging()

args = parser.parse_args()
common.printlog(str(args),logger)

basedir = args.basedir
survey = args.survey
version = args.version
specrel = args.verspec
ccut = args.ccut
pcut = args.pcut
rm = int(args.minr)
rx = int(args.maxr)

#get some parameters that depend on the tracer
mainp = main(args.input_tracer,args.verspec,survey=survey) #parameters contained in mainp

dchi2 = mainp.dchi2 #used for good redshift selection when making clustering catalogs
#clustering catalogs will have these redshift bounds:
zmin = mainp.zmin
zmax = mainp.zmax
if args.zmax != None:
    zmax = args.zmax
if args.zmin != None:
    zmin = args.zmin
common.printlog(f'zmin = {zmin}, zmax = {zmax}', logger)

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

common.printlog(f'running subsampling on {dirin}{args.input_tracer} catalogs according to {ccut} {pcut}',logger)

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

def get_FSF_Y1_old(indata, fsf_cols, prog='bright'):
    fsf_dir='/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/'
    fsf = fitsio.read(fsf_dir+f'fastspec-iron-main-{prog}.fits',columns = fsf_cols)
    ol = len(indata)
    indata = join(indata,fsf,keys=['TARGETID']) #note, anything missing from fastspecfit will now be missing
    del fs
    common.printlog('length before/after Y1 old fastspecfit join '+str(ol)+' '+str(len(indata)),logger)
    return indata

def get_FSF(indata, fsf_cols, prog='bright'):
    # BUG Missing galaxies...
    if survey == 'Y1':
        fsf_dir='/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs/'
        fsl = []
        for hp in range(0,12):
            fsi = fitsio.read(fsf_dir+f'fastspec-iron-main-{prog}-nside1-hp'+str(hp).zfill(2)+'.fits', ext='SPECPHOT', columns = fsf_cols)
            fsl.append(fsi)
        fs = np.concatenate(fsl)
        del fsl
    else:
        fsf_dir = '/global/cfs/cdirs/desi/vac/dr2/fastspecfit/loa/v1.0/catalogs/'
        fsl = []
        for hp in range(0,12):
            fsi = fitsio.read(fsf_dir+f'fastspec-loa-main-{prog}-nside1-hp'+str(hp).zfill(2)+'.fits',ext='SPECPHOT',columns = fsf_cols)
            fsl.append(fsi)
        fs = np.concatenate(fsl)
        del fsl

    ol = len(indata)
    common.printlog('fsf columns: '+str(fs.dtype.names),logger)
    for c in fsf_cols:
        if c in indata.dtype.names and c != 'TARGETID':
            indata.remove_column(c) 
    indata = join(indata, fs, keys=['TARGETID'], join_type='left')
    del fs
    common.printlog('length before/after fastspecfit join '+str(ol)+' '+str(len(indata)),logger)
    return indata

def get_extra_columns(indata, specrel, extra_cols:list):
    extra_cols.append('TARGETID') 

    if specrel == 'fuji': 
        # This is the version with 18 of the 20 patches
        extra_fn = '/global/cfs/cdirs/desi/users/ianw89/private/DATA/BGS_FUJI/ian_BGS_SV3_merged_noY3.fits'
    elif specrel == 'iron':
        extra_fn = '/global/cfs/cdirs/desi/users/ianw89/private/DATA/BGS_IRON/ian_BGS_merged.fits'
    elif specrel == 'loa-v1':
        extra_fn = '/global/cfs/cdirs/desi/users/ianw89/private/DATA/BGS_LOA/ian_BGS_Y3_merged.fits'
    else:
        common.printwarn('specrel not recognized, cannot get extra columns', logger)
        sys.exit(1)
    
    extra = fitsio.read(extra_fn, columns=extra_cols)
    orig_len = len(indata)
    indata = join(indata, extra, keys=['TARGETID'])
    common.printlog(f'length before/after extra columns join {orig_len} {len(indata)}', logger)
    return indata

pcut_col = None
tracer_out = args.input_tracer
if args.ccut != '':
    if 'FSFABSmag' in args.ccut:
        csplit = args.ccut.split('-')
        bnd = csplit[1]
        magcolname = 'ABSMAG01_SDSS_'+bnd    
    tracer_out += f'_{args.ccut.replace("FSFABSmag-", "")}'
if args.pcut != '':
    tracer_out += f'_{args.pcut}'
    pcut_col = args.pcut.split('-')[0]
if args.extracut != 'ALL':
    tracer_out += f'_{args.extracut}'
if args.passcut > 0:
    tracer_out += f'_{args.passcut}p'
if args.centrals == 'y':
    tracer_out += '_CEN'

#tracer_out = args.input_tracer

if args.mkfulldat == 'y':

    common.printlog('reading full data file '+dirin+args.input_tracer+'_full'+args.use_map_veto+'.dat.fits',logger)
    fulldat = fitsio.read(dirin+args.input_tracer+'_full'+args.use_map_veto+'.dat.fits')

    # Use group catalog to get centrals (if desired) and quality cuts
    # Including these quality cuts is a key benefit of using my script
    path = '/global/cfs/cdirs/desi/users/ianw89/groupcatalogs/BGS_Y3/v0.8/GROUP_CATALOG_BGS_Y3_1PASS_v0.8.fits'
    grpcat = Table(fitsio.read(path, columns=['TARGETID', 'Z_ASSIGNED_FLAG', 'ABS_MAG_R', 'G_R', 'IS_SAT']))
    common.printlog(f'{len(grpcat)} group catalog rows read from {path}', logger)

    grpcat_sel = grpcat['Z_ASSIGNED_FLAG'] == 0 # DESI spectroscopic data only
    if args.centrals == 'y':
        grpcat_sel &= ~grpcat['IS_SAT'] # Centrals only
    grpcat = grpcat[grpcat_sel]
    grpcat.remove_columns(['Z_ASSIGNED_FLAG', 'IS_SAT'])
    common.printlog(f'{len(grpcat)} group catalog rows after cuts (DESI spec + centrals only)', logger)

    # Merge with fulldat, inner join
    fulldat = join(fulldat, grpcat, keys=['TARGETID'], join_type="inner")
    common.printlog(f'{len(fulldat)} full data rows after joining with group catalog', logger)

    
    sel = np.ones(len(fulldat),dtype=bool) #initialize selection to all true
    
    ian_columns_needed = []
    if (args.extracut == 'Q' or args.extracut == 'SF') and 'QUIESCENT' not in fulldat.dtype.names:
        ian_columns_needed.extend(['QUIESCENT'])
    if args.passcut > 0 and 'NTILE_MINE' not in fulldat.dtype.names:
        ian_columns_needed.append('NTILE_MINE')
    if args.pcut != '' and pcut_col not in fulldat.dtype.names:
        ian_columns_needed.append(pcut_col)
    if args.ccut != '' and magcolname not in fulldat.dtype.names:
        ian_columns_needed.append(magcolname)

    if len(ian_columns_needed) > 0:
        fulldat = get_extra_columns(fulldat, specrel, ian_columns_needed)
        # Allowing the extra columns to remove galaxies; THIS IS INTENTIONAL FOR SV3 ANALYSIS
        sel = np.ones(len(fulldat),dtype=bool) #initialize selection to all true

    if 'Q' in args.extracut:
        sel &= fulldat['QUIESCENT'] == 1
    if 'SF' in args.extracut:
        sel &= fulldat['QUIESCENT'] == 0

    if args.passcut > 0:
        sel &= fulldat['NTILE_MINE'] >= args.passcut
            
    #for selections based on fastspecfit; other cases can be written similarly
    if 'FSFABSmag' in args.ccut:
        #this is an example that can make subsamples based on fastspecfit absolute magnitudes
        #other critera can be added
        csplit = args.ccut.split('-')
        bnd = csplit[1]
        min_mag = None 
        max_mag = None 
        if len(csplit) >= 3:
            min_mag = -float(csplit[2])
        if len(csplit) >= 4:
            max_mag = -float(csplit[3])
        # TODO
        # If the column already exists, print off how many nan's
        if magcolname in fulldat.dtype.names:
            nans = np.sum(np.isnan(fulldat[magcolname]))
            common.printlog(f'Column {magcolname} already exists with {nans} NaNs', logger)
        else:
            common.printwarn("Need abs mag, but not present!", logger)
            sys.exit(1)
            #common.printlog('about to get columns from fastspecfit', logger)
            #fulldat = get_FSF(fulldat,['TARGETID',colname])
        ecorr = np.zeros(len(fulldat))
        if 'ecorr' in args.ccut:
            ecorr = -0.8*(fulldat['Z_not4clus']-0.1) #seemed best here for getting constant n(z) /global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/BGS_explore.ipynb
        if min_mag is not None:
            sel &= fulldat[magcolname] < min_mag + ecorr
            common.printlog('length after min Absmag selection '+str(np.sum(sel)),logger)
        if max_mag is not None:
            sel &= fulldat[magcolname] > max_mag + ecorr
            common.printlog('length after max Absmag selection '+str(np.sum(sel)),logger)

        #add any additional selections here
    
    if args.pcut != '':
        pcut_split = args.pcut.split('-')
        if len(pcut_split) != 3:
            common.printwarn(f'pcut {args.pcut} does not have enough elements', logger)
            sys.exit(1)
        min_val = float(pcut_split[1])
        max_val = float(pcut_split[2])
        sel &= fulldat[pcut_col] > min_val
        common.printlog(f'length after min {pcut_col} selection {np.sum(sel)}', logger)
        sel &= fulldat[pcut_col] < max_val
        common.printlog(f'length after max {pcut_col} selection {np.sum(sel)}', logger)

    #write output to new "full" catalog at your defined location
    fout = args.outdir+'/'+tracer_out+'_full'+args.use_map_veto+'.dat.fits'
    common.write_LSS_scratchcp(fulldat[sel],fout,logger=logger)


    
#create "clustering" catalogs for data with no NGC/SGC split or FKP weights 
#needs to happen before randoms so randoms can get z and weights
weightileloc=True
if args.compmd == 'altmtl':
    weightileloc = False
if mkclusdat:
    ct.mkclusdat(args.outdir+'/'+tracer_out,weightileloc,tp=tracer_out,dchi2=dchi2,zmin=zmin,zmax=zmax,use_map_veto=args.use_map_veto)

nzcompmd = 'ran'
if args.compmd == 'altmtl':
    nzcompmd = args.compmd
rcols=['Z','WEIGHT','WEIGHT_SYS','WEIGHT_COMP','WEIGHT_ZFAIL','TARGETID_DATA'] #columns to make sure are in the randoms
inds = np.arange(rm,rx)

#make clustering catalogs for randoms
if mkclusran:
    clus_arrays = [fitsio.read(args.outdir+'/'+tracer_out+'_clustering.dat.fits')]

    def _parfun_cr(ii):
        if args.passcut > 0:
            common.printlog(f'', logger)
            # Replace source path to one that has my NTILE_MINE column
            inpath = f'/global/cfs/cdirs/desi/users/ianw89/newclustering/{survey}/LSS/{specrel}/LSScats/{version}/{args.input_tracer}_{ii}_full{args.use_map_veto}.ran.fits' 
            ran_cols = ['RA','DEC','TARGETID','TILEID','NTILE','PHOTSYS','NTILE_MINE']
            if args.compmd != 'altmtl':
                ran_cols.append('FRAC_TLOBS_TILES')
            ranin = Table(fitsio.read(inpath,columns=ran_cols))
            # Apply the passcut selection
            ol = len(ranin)
            rsel = ranin['NTILE_MINE'] >= args.passcut
            ranin = ranin[rsel]
            common.printlog(f'randoms-{ii} length before/after passcut {ol} {np.sum(rsel)}', logger)
        else:
            ranin = dirin + args.input_tracer +'_'
    
        ct.mkclusran(ranin,args.outdir+'/'+tracer_out+'_',ii,rcols=rcols,clus_arrays=clus_arrays,use_map_veto=args.use_map_veto,compmd=nzcompmd,logger=logger,tp=args.input_tracer)
    
    if args.par == 'y':
        from multiprocessing import Pool
        with Pool() as pool:
            res = pool.map(_parfun_cr, inds)

    else:
        for ii in inds:#range(rm,rx):
            _parfun_cr(ii)

#define P0 value used for fiducial FKP weights and dz used for creating n(z)
if tracer_out[:3] == 'QSO':
    dz = 0.02
    P0 = 6000    
else:    
    dz = 0.01

if tracer_out[:3] == 'LRG':
    P0 = 10000
if tracer_out[:3] == 'ELG':
    P0 = 4000
if tracer_out[:3] == 'BGS':
    P0 = 7000

nran = rx-rm
regions = ['NGC', 'SGC']

#function to take a file and split it NGC/SGC
def splitGC(flroot,datran='.dat',rann=0):
    import LSS.common_tools as common
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    app = 'clustering'+datran+'.fits'
    if datran == '.ran':
        app = str(rann)+'_clustering'+datran+'.fits'

    fn = Table(fitsio.read(flroot +app))
    sel_ngc = common.splitGC(fn)#gc.b > 0
    outf_ngc = flroot+'NGC_'+app
    common.write_LSS_scratchcp(fn[sel_ngc],outf_ngc,logger=logger)
    outf_sgc = flroot+'SGC_'+app
    common.write_LSS_scratchcp(fn[~sel_ngc],outf_sgc,logger=logger)


dirout = args.outdir #just because original copied code used dirout

#split catalogs NGC/SGC
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
        common.addnbar(fb,bs=dz,zmin=zmin,zmax=zmax,P0=P0,nran=nran,par=args.par,compmd=nzcompmd,comp_ntl=comp_ntl,weight_ntl=weight_ntl,logger=logger)

# determine linear weights for imaging systematics
# this is new for doing after the fact based on clustering catalogs
if args.imsys_clus == 'y':
    #import package
    from LSS.imaging import densvar 
    
    #setup redshift bins to use for regressions; you might find something else works better!
    if args.input_tracer[:3] == 'ELG':
        if args.imsys_zbin == 'y':
            zrl = [(0.8,1.1),(1.1,1.6)]
        else:
            zrl = [(0.8,1.6)]
    if args.input_tracer[:3] == 'QSO':
        if args.imsys_zbin == 'y':
            zrl = [(0.8,1.3),(1.3,2.1),(2.1,3.5)] 
        else:
            zrl = [(0.8,3.5)]   
    if args.input_tracer[:3] == 'LRG':
        if args.imsys_zbin == 'y':
            zrl = [(0.4,0.6),(0.6,0.8),(0.8,1.1)] 
        else:
            zrl = [(0.4,1.1)]  
    if args.input_tracer[:3] == 'BGS': # Ian's custom z range for group catalog
        zrl = [(0.001,0.5)]

    
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
    for i in range(0,args.nran4imsys):
        ran = fitsio.read(os.path.join(dirout, tracer_out+'_NGC_'+str(i)+'_clustering.ran.fits')) 
        ranl.append(ran)
        ran = fitsio.read(os.path.join(dirout, tracer_out+'_SGC_'+str(i)+'_clustering.ran.fits')) 
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

        for zr in zrl:
            zmin = zr[0]
            zmax = zr[1]
            
            common.printlog('getting weights for region '+reg+' and '+str(zmin)+'<z<'+str(zmax),logger)
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
            wsysl = densvar.get_imweight(dat,rands,zmin,zmax,reg,fitmapsbin,use_maps,sys_tab=sys_tab,zcol='Z',modoutname = dirout+'/'+tracer_out+'_'+reg+'_'+str(zmin)+str(zmax)+'_linfitparam.txt',figname=dirout+'/'+tracer_out+'_'+reg+'_'+str(zmin)+str(zmax)+'_linclusimsysfit.png',wtmd='clus')
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
if args.imsys_clus_ran == 'y':
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

