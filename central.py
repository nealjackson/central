import numpy as np,njj,warnings,os,sys,time,glob,cProfile
try:
    import emcee
    import corner
    have_emcee = True
except:
    have_emcee = False
import scipy as sp; from scipy import special,optimize,signal,ndimage
from scipy.optimize import fmin
from scipy.ndimage import measurements,filters,gaussian_filter
import matplotlib; from matplotlib import pyplot as plt,rc
import matplotlib.patches as patches
warnings.simplefilter('ignore')
plt.rcParams['image.origin']='lower'
plt.rcParams['image.interpolation']='nearest'
lims = np.loadtxt('vla_lims')
merlinlims = np.array([[100.,100.,10000.],[125.,50.,50.0]])
fitalpha, LOGSTART, LOGEND, LNUM = False, -7. ,1.0,300
TABLE1 = np.asarray(np.loadtxt('table1',dtype='S')[:,2:],dtype='float')
FRAT,FERR,ABSEP,ASEP,BSEP,DERR = 11.7,1.2,1.60,1.444,0.110,0.01

def movie (prefix='gamma_rb',fps=4):
    for i in np.sort(glob.glob(prefix+'*.png')):
        os.system('convert '+i+' '+i.replace('png','jpg'))
    command = "mencoder \"mf://%s*.jpg\" -mf fps=%d -o amp_video.avi -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=4800" % (prefix,fps)
    os.system(command)

def findim (s,gal,bh,r,defl,ddefl,logstart,logend):
    t,im = defl-r-s, np.array([])
    for i in range(len(r)-1):
        if (t[i]<0.0 and t[i+1]>0.0) or (t[i]>0.0 and t[i+1]<0.0):
            x = abs(t[i])/(abs(t[i])+abs(t[i+1]))
            rret = r[i]+(r[i+1]-r[i])*x
            dret = defl[i]+(defl[i+1]-defl[i])*x
            ddret = ddefl[i]+(ddefl[i+1]-ddefl[i])*x
            mu = 1./((1.-dret/rret)*(1.-ddret))
            new = np.array([rret,dret,ddret,mu])
            try:
                im = np.vstack((im,new))
            except:
                im = np.array([np.copy(new)])
    return im,r,defl,ddefl

def setup (logstart,logend,n,gal,bh,s,stype='nuker'):
    r = np.logspace (logstart,logend,n)
    defl = defr_nuker (gal,r,bh) if stype=='nuker' else defr_cusp (gal,r,bh)
    r = np.append(-r[::-1],r)
    defl = np.append(-defl[::-1],defl)
    ddefl = np.gradient(defl) / np.gradient(r)
    return findim(s,gal,bh,r,defl,ddefl,logstart,logend)

def defr_nuker (gal,r,bh):
    (sigma,alpha,beta,gamma,kb,rb)=gal
    a,b = (2.0-gamma)/alpha, (beta-gamma)/alpha
    rb,kb = 10.**rb, 10.**kb
    out = 2.0**(1.+b)*kb*rb*(r/rb)**(1.-gamma)* \
          special.hyp2f1 (a,b,1.+a,-(r/rb)**alpha)/(2.-gamma) \
          + bh*0.686E-12*pow(sigma,3.75)/r
#          + bh*0.841E-12*pow(sigma,3.75)/r
#          + bh*1.1587E-12*pow(sigma,3.75)/r
    return out

def defr_cusp (gal,r,bh):    # cusp: beta=2. isothermal, max defl = 2.pi.kb
    (sigma,alpha,beta,gamma,kb,rb)=gal
    a=(beta-3.)/2.
    rb,kb = 10.**rb, 10.**kb
    X=1.+r*r/(rb*rb)
    out = (2.0*kb*rb/r) * ( special.beta(a,0.5*(3.-gamma)) \
          - special.beta(a,1.5)*X**(0.5*(3.-beta))* \
          special.hyp2f1(a,0.5*gamma,0.5*beta,1./X) ) \
          + bh*0.841E-12*pow(sigma,3.75)/r
#          + bh*1.1587E-12*pow(sigma,3.75)/r
    return out

def getrrad (r,defl, ddefl):
    for i in range (len(r)-1):    # want rightmost one, so no break in if loop
        if ddefl[i]>1.0 and ddefl[i+1]<1.0:
            x=abs(ddefl[i]-1.)/(abs(ddefl[i+1]-1.)+abs(ddefl[i]+1))
            rrad = r[i]+(r[i+1]-r[i])*x
            arrad = defl[i]+(defl[i+1]-defl[i])*x
    return 0.0,0.0 if rrad<1.0E-18 else rrad, arrad

def cenmag (gal,bh,name,s=-1000.0):
    im,r,defl,ddefl = setup (LOGSTART,LOGEND,LNUM,gal,bh,s)
    if s==-1000.0:
        rrad,arrad = getrrad (r, defl, ddefl)
        return ((arrad/rrad -1.0)**-2. if rrad else 0.0)
    else:
        return im[:,3]

def pmucore (gid,ulow=-4.0,uhigh=-1.0,dobh=False): # check vs Keeton fig 2-3 
    gal = TABLE1[gid,:6]
    for s in np.arange(0,10,0.01):
        images,r,defl,ddefl = setup (LOGSTART,LOGEND,LNUM,gal,dobh,s)
        # need to deal with the case of the bright image and two just around 
        # the BH, where we would only have 1 image in the absence of BH
        if len(images)<3 or abs(images[-1,-1])<0.001:
            smax=s
            break
    x,y = np.random.random(1000)*smax, np.random.random(1000)*smax
    ssamp = np.hypot(x,y); ssamp = ssamp[ssamp<smax]
    mu = np.array([])
    for s in ssamp:
        images,r,defl,ddefl = setup (LOGSTART,LOGEND,LNUM,gal,dobh,s)
        if len(images)>2:
            mu = np.append(mu,images[-2,-1])
    print (TABLE1[gid,0],':mu_mean',np.log10(np.nanmean(mu)))
    mu = np.log10(np.sort(mu))
    p = np.array([])
    for i in np.arange(ulow,uhigh,0.1):
        p = np.append(p,float(len(mu[mu>i]))/float(len(mu)))
    plt.show()
    
def mkplot (gid,s,bh,gal=[],fname='',x1=0,x2=0,y1=0,y2=0,noisy=False,axes=False):
    if not len(gal) and gid>=0:
        gal = TABLE1[gid,:6]
    if not bh:
        gal[0] = 0.0
    images,r,defl,ddefl = setup (LOGSTART,LOGEND,LNUM,gal,True,s)
    margin=0.2*(defl[r>0.001].max()-defl[r<-0.001].min())
    (x1,x2)=(r.min(),r.max()) if not x1 and not x2 else (x1,x2)
    if not y1 and not y2:
        y1,y2 = defl[r<-0.001].min()-margin, defl[r>0.001].max()+margin
#    ax=plt.subplot(111) ##
    plt.plot(r,defl)
    plt.grid(True)
    plt.plot(r-s,r)
    tstring = 'sig:%d a=%.2f b=%.2f gam=%.2f k=%.2f r=%.2f ' % \
            (int(gal[0]) if bh else 0,gal[1],gal[2],gal[3],gal[4],gal[5])
    if noisy:
        for i in range(images.shape[0]):
            string = '%.3f %.3f %.3g' % (images[i,0],images[i,1],images[i,3])
            plt.text(x1+0.05*(x2-x1),y2-0.05*float(i+1)*(y2-y1),string)
    plt.xlim(x1,x2); plt.ylim(y1,y2)
    if axes:
        plt.plot([x1,x2],[0,0],'k-',linewidth=1.5)
        plt.plot([0,0],[y1,y2],'k-',linewidth=1.5)
    plt.xlabel('Radius/arcsec')
    plt.ylabel('Deflection angle /arcsec')
#    ax.add_patch(patches.Rectangle((-0.1,1.0),0.4,0.8,fill=False))
    plt.title(tstring)
    if not fname:
        plt.show()
    else:
        plt.savefig(fname)
    plt.clf()

def compare_mag (bh):
    a=np.array([])
    table=np.loadtxt('table1',dtype='S')
    tnum = np.asarray(table[:,2:],dtype='float')
    krrad=np.asarray(table[:,10],dtype='float')
    krrbh=np.asarray(table[:,11],dtype='float')
    for i in range(table.shape[0]):
        mmag = cenmag (tnum[i,:6],bh,table[i,0])
        a=njj.nstack(a,np.array([np.log10(mmag),krrad[i]]))
        print ('%s %.2f %.2f ' % (table[i,0],np.log10(mmag),-krrbh[i]))

def chisq_s (x0, *x):
    images,r,defl,ddefl = setup (LOGSTART,LOGEND,LNUM,x,True,x0)
    if not len(images) or images.shape[0]==1:
        return 1.0E9
    elif images[0,0]>-0.01:  # missing the bright image
        return 1.0E+9
    else:
        oer = abs(images[-1,0]-images[0,0])
        chisq_er = (oer-(ASEP+BSEP))**2 / DERR**2
        ofrat = abs(images[0,3]/images[-1,3])
        chisq_frat = (ofrat-FRAT)**2 / FERR**2
        osrat = abs(images[0,0]/images[-1,0])
        chisq_srat =  (osrat-ASEP/BSEP)**2 / 1.0**2
    return chisq_er+chisq_frat+chisq_srat

def chisq_fit (x0, *x):
    x1 = np.array([0.1])
    if len(x0)==5:  # isothermal, insert beta=1.0 as third
        x0 = np.insert (x0, 2, 1.0)
    fjunk=open('junk.log','w')
    save_stdout = sys.stdout; sys.stdout = fjunk
    xopt,fopt,ai,gn,wa = fmin(chisq_s,x1,args=tuple(x0),full_output=1)
#   prior on gamma
    if x0[3]<0.0:
        fopt += 1.0E6*abs(x0[3])
    sys.stdout = save_stdout; fjunk.close()
    val = -fopt/2.0 if x[0] else fopt
    f = open('mcmc.log','a')
    for i in range(len(x0)):
        f.write('%.6f '%x0[i])
    f.write('%.6f %.6f\n'%(xopt,fopt))
    f.close()
    if x[0] and (val>0.0 or np.isinf(val) or np.isnan(val)):
        return -1.0E9
    return val

def shist (x,y,z,yticks,arr,lims,title,loc=2):
    plt.subplot(x,y,z,yticks=yticks)
    plt.hist(arr,range=lims);plt.legend([title],loc=loc)

def mcmc2data (infile='mcmc.log',outfile='mcmc2data.log',mknew=True):
    if not have_emcee:
        return
    if mknew:
        fi = open(infile)
        f = open(outfile,'w')
        for line in fi:
            i = np.asarray(line.split(),dtype='f')
            gal = i[:6]
            im,r,defl,ddefl = setup (LOGSTART,LOGEND,LNUM,gal,True,i[6])
            if not len(im) or not im.shape[1]:
                continue
            flux = np.sort(abs(im[:,3]))
            if len(flux)>2:
                frat = np.log10(flux[-3]/flux[-1])
            else:
                continue
#   output:  0=sig 1=alpha 2=beta 3=gamma 4=rb 5=kb 6=s 7=chisq 8=f3/f1 9=B-C
            f.write ('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n' % \
            (i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],frat,im[-1,0]-im[-2,0]))
            f.flush()
        f.close()
        fi.close()
    mcmclabels = [r'$\log_{10}M(BH)/M_{\odot}$',r'$\alpha$',r'$\beta$',r'$\gamma$',r'$\log_10\kappa_b$',r'$\log_{10}r_b$/arcsec']
    mcmclimits = np.array([[5.0,10.5],[-0.5,5.0],[0.5,2.0],[0.,2.],[-0.5,2.5],[-4.,0.]])
    mcmclabels_i = [r'$\log_{10}M(BH)/M_{\odot}$',r'$\alpha$',r'$\gamma$',r'$\log_10\kappa_b$',r'$\log_{10}r_b$/arcsec']
    mcmclimits_i = np.array([[5.0,10.5],[-0.5,5.0],[0.,2.],[-0.5,2.5],[-4.,0.]])
    a = np.load(outfile) if outfile[-3:]=='npy' else np.loadtxt(outfile)
    a[:,0] = np.log10(0.00126*a[:,0]**4.8)
    a = a[a[:,7]<100000]
    a = a[~np.isnan(a[:,7])]
    for i in range(len(mcmclimits)):
        a = a[a[:,i]>mcmclimits[i,0]]
        a = a[a[:,i]<mcmclimits[i,1]]
    b = a[a[:,8]>-3.92]
    c = a[a[:,8]<-3.92]
    matplotlib.rcParams.update({'font.size': 15})
    matplotlib.rcParams.update({'axes.labelsize': 25})
    matplotlib.rcParams.update({'axes.titlesize': 25})
    figure = corner.corner(a[:,:6],labels=mcmclabels,title_args={"fontsize":12},bins=10,range=mcmclimits)
    figure.savefig(outfile.split('.')[0]+'_bc.png')
    figure = corner.corner(c[:,:6],labels=mcmclabels,title_args={"fontsize":12},bins=10,range=mcmclimits)
    figure.savefig(outfile.split('.')[0]+'_fc.png')

def mkfig5 (fqvfile='fqv.log'):
    a = np.loadtxt(fqvfile)
    a = a[~np.isnan(a[:,1])]
    plt.clf()
    fluxes = a[:,1] - np.log10(FRAT)  # convert C/B to C/A
    fluxes[fluxes<-5] = -6.0
    plt.hist(fluxes,bins=14,range=[-7,0])
    print (len(fluxes[fluxes<-5.0]),len(fluxes[(fluxes>-4.9)&(fluxes<-3.92)]),\
             len(fluxes[fluxes>=-3.92]))
    plt.plot([-3.92,-3.92],[0,250],'k-')
    plt.xlabel('log(Flux density C/A)');plt.ylabel('Number')
    plt.savefig('fig5a.png',bbox_inches='tight')
    aa = a[a[:,1]>-6.0]
    matplotlib.rcParams.update({'font.size': 10})
    plt.clf()
    plt.semilogx(18000.*10**aa[:,1],1000.*aa[:,2]*BSEP/aa[:,5],'b.')
    plt.semilogx(lims[:,1],lims[:,0],'r-')
    plt.semilogx(merlinlims[0],merlinlims[1],'r-')
    plt.xlabel ('Flux density of image C / microJy')
    plt.ylabel ('B-C image separation / mas')
    plt.xlim(1,15000);plt.ylim(0,120)
    plt.savefig('fig5b.png',bbox_inches='tight')

def fitall(gid=-1,nparam=6):
    x0 = np.array([226,1.88,1.0,0.3,1.43,-1.0]) if gid==-1 else TABLE1[gid,:6]
    x = (False,nparam)
    f = open('junk.log','w')
    save_stdout = sys.stdout; sys.stdout = f
    x0opt,fopt,ai,gn,wa = fmin(chisq_fit,x0,args=x,full_output=1)
    print('***',x0opt,fopt,'\n')
    x1 = np.array([0.1])
    xopt,fopt,ai,gn,wa = fmin(chisq_s,x1,args=tuple(x0opt),full_output=1)
    sys.stdout = save_stdout; f.close()
    return fopt,x0opt

def getstart():
    a=np.loadtxt('table1',dtype='S')
    fo=open('getstart.log','a')
    for i in range(20,len(a)):
        fopt, gal = fitall(i)
        fo.write ('%.7f %.7f %.7f %.7f %.7f %.7f %f\n' % \
            (gal[0],gal[1],gal[2],gal[3],gal[4],gal[5],fopt))
        fo.flush()
        sys.stdout.write ('%d %.2f %.2f %.2f %.2f %.2f %f\n' % \
            (int(gal[0]),gal[1],gal[2],gal[3],gal[4],gal[5],fopt))
    fo.close()
  
def mcmc (scale=1.04,nwalkers=500,nburn=50,niter=10000,nparam=5,startfile='getstart_5param.log'):
    if not have_emcee:
        return None
    start = np.loadtxt(startfile,usecols=range(nparam))
    smc = emcee.EnsembleSampler (nwalkers,nparam,chisq_fit,a=scale,args=(True,nparam))
    p0 = [start[i%len(start),:nparam] for i in xrange(nwalkers)]
    pos,prob,state = smc.run_mcmc(p0,nburn)
    smc.reset()
    print ('\n\n**** main walking: ****\n\n')
    smc.run_mcmc(pos,niter,rstate0=state)
    return smc

def match_source (d1,d2,gamma,kb,rb):
    beta = 1.0
    alpha = np.arange(0.1,4.0,0.1)
    posgood = np.zeros_like(alpha)
    for i in range(len(alpha)):
        gal=[200,alpha[i],beta,gamma,kb,rb]
        sa,images=match_ratio(-1,False,gal)
        gotd1,gotd2=abs(images[0,0]),abs(images[-1,0])
        posgood[i]=(gotd1-d1)**2.0+(gotd2-d2)**2.0
    plt.plot(alpha,posgood)
    plt.show()

def match_ratio (gid,bh,gal=[],smin=0.0,doplot=False,dorat=True,dolrat=False,dolength=False):
    gal = TABLE1[gid,:6] if gal==[] else gal
    if gal[0]<smin:
        return 0.0
    s=np.linspace(0.0,5.0,1000)
    ratio = np.zeros_like(s)
    asep, bsep, frat = np.array([]), np.array([]), np.array([])
    for i in range(len(s)):
        images,r,defl,ddefl = setup(LOGSTART,LOGEND,LNUM,gal,bh,s[i])
        asep = np.append(asep,-images[0,0])
        bsep = np.append(bsep,images[-1,0])
        frat = np.append(frat,abs(images[0,3]/images[-1,3]))
    goodrat = (frat-FRAT)**2/FERR**2
    goodlength = (asep-ASEP)**2/DERR**2 + (bsep-BSEP)**2/DERR**2
    goodlrat = ((asep/bsep)-(ASEP/BSEP))**2 / (DERR**2*(ASEP**2/BSEP**4+1/BSEP**2))
    goodcrit = dorat*goodrat + dolength*goodlength + dolrat*goodlrat
    goodcritmin = np.nanmin(goodcrit)
    idxcritmin = np.argwhere(goodcrit==goodcritmin)[0][0]
    sa = s[idxcritmin]
    images,r,defl,ddefl = setup(LOGSTART,LOGEND,LNUM,gal,bh,sa)
    return sa,images,goodcritmin

def findequiv (nsim=100,dolength=False,dolrat=True,dorat=True,domc=False,\
               erlim=[-0.3,0.1],glim=1000,dovirgo=False):
    b=np.asarray(np.loadtxt('table1',dtype='S')[:,1:],dtype='f')
    b=b[b[:,7]>erlim[0]]; b=b[b[:,7]<erlim[1]]
    nsim = nsim if domc else len(b) 
    mean,scat,iloop = [],[], 0
    for i in range (1,7):
        mean.append(np.median(b[:,i]))
        bsort = np.sort(b[:,i])
        lb = len(bsort)
        scat.append(0.5*(bsort[2*lb/3]-bsort[lb/3]))
    results = np.zeros((nsim,9))
    mags,dists,goods = [], [], []
    f = open('fqv.log','w')
    if dovirgo:
        virgo = np.loadtxt('virgo.samples.phy')
        nsim = len(virgo)
    while True:
        if iloop==nsim:
            break
        if dovirgo:
            parms = virgo[iloop]
        else:
            parms = np.random.randn(6)*scat+mean if domc else b[iloop,1:7]
        sa,im,goodmin =  match_ratio (-1,True,gal=parms,dolength=dolength,\
                                      dorat=dorat,dolrat=dolrat)
        if domc and goodmin>glim:
            print ('Rejected point at number:',iloop)
            continue
        goods.append(goodmin)
        mkplot(-1,0. if np.isnan(sa) else sa,True,gal=parms,\
               fname='plots/fqv_%03d.png'%iloop,noisy=True)
        if len(im)==3:
            thismag = np.log10(abs(im[1,-1]/im[2,-1]))
            thisdist = im[2,0]-im[1,0]
        elif len(im)==5:
            v = im[1:4,-1]
            brightest_middle = 1+np.argwhere(v==v.max())[0][0]
            thismag = np.log10(abs(im[brightest_middle,-1]/im[-1,-1]))
            thisdist = im[-1,0]-im[brightest_middle,0]
        else:
            thismag = thisdist = np.nan
        f.write ('%.3f %7.3f %.3f %6d %.3f %.3f %5.2f   %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %d\n' % \
        (sa,thismag,thisdist,int(goodmin),im[0,0],im[-1,0],-im[0,-1]/im[-1,-1],\
         parms[0],parms[1],parms[2],parms[3],parms[4],parms[5],len(im)))
        sys.stdout.write('.'); sys.stdout.flush()
        if not iloop%50:
            sys.stdout.write('%d'%iloop)
        iloop+=1
    print ('Median goodness was',np.nanmedian(goods))
    f.close()
    constraints_fig('fqv.log')

def func_opt_gamma_rb (x0,*x):
    global fitalpha
    if fitalpha:
        sigma,beta,gamma,rb = x
        alpha,s,kappa = x0
    else:
        sigma,alpha,beta,gamma,rb = x
        s,kappa = x0
    if alpha>15.0:      # prior on alpha
        return 1.0E+9
    gal = [sigma,alpha,beta,gamma,kappa,rb]
    im,r,defl,ddefl = setup (LOGSTART,LOGEND,LNUM,gal,True,s)
    if len(im)>2:
        mu = np.sort(abs(im[:,-1]))
        fAB = np.log10(mu[-1]/mu[-2])
        sepAB = abs(im[-1,0]-im[0,0])
        sepA,sepB = abs(im[0,0]),abs(im[-1,0])
        goodness = 100.*((np.log10(FRAT)-fAB)**2 + (ASEP-sepA)**2/DERR**2 + (BSEP-sepB)**2/DERR*2)
#        goodness = 100.*((np.log10(FRAT)-fAB)**2 + (ASEP+BSEP-sepAB)**2)
    else:
        goodness = 1.0E9
    return goodness

# For a given gamma, rb, flux ratio and separation, find the magnification ratio C/A
# by fitting for s, alpha and kappa. Beta and sigma are fixed throughout.
def opt_gamma_rb (gamma,rb,alpha=2.0,sigma=0.0,beta=1.0):
    global fitalpha
    s,kappa = plot_gamma_rb (gamma,rb,alpha,sigma,beta,False)
    if fitalpha:
        x = (sigma,beta,gamma,rb)
        x0 = np.array([alpha,s,kappa])
        xopt = fmin (func_opt_gamma_rb,x0,args=x,ftol=0.0001,xtol=0.0001)
        x = (sigma,beta,gamma,rb)
    else:
        x = (sigma,alpha,beta,gamma,rb)
        x0 = np.array([s,kappa])
        xopt = fmin (func_opt_gamma_rb,x0,args=x,ftol=0.0001,xtol=0.0001)
        x = (sigma,alpha,beta,gamma,rb)
    goodness = func_opt_gamma_rb (xopt, *x)
    alpha = xopt[0] if fitalpha else alpha
    s = xopt[1] if fitalpha else xopt[0]
    kappa = xopt[2] if fitalpha else xopt[1]
    gal = [sigma,alpha,beta,gamma,kappa,rb]
    im,r,defl,ddefl = setup (LOGSTART,LOGEND,LNUM,gal,True,s)
    if len(im)>2 and goodness < 5.0: # this goodness excludes lots of fits
        mu = np.sort(abs(im[:,-1]))
        nplot = len(glob.glob('gamma_rb*.png'))
        fn = 'gamma_rb_%04d.png'%nplot
        mkplot(-1,s,True,[sigma,alpha,beta,gamma,kappa,rb],noisy=True,\
               fname=fn,x1=-1.5,x2=1.5)
        f = open('gamma_rb.log','a')
        f.write('%.3f %.2f %.3f %.3f %.3f %.3f %.3f  %.1f %.3f %.4f\n' % \
                  (s,sigma,alpha,beta,gamma,kappa,rb,mu[-1]/mu[-2],\
                   np.log10(mu[-3]/mu[-1]),goodness))
        f.close()
        return np.log10(mu[-3]/mu[-1])
    else:
        f = open('gamma_rb.log','a')
        f.write('%.3f %.2f %.3f %.3f %.3f %.3f %.3f  NaN NaN  %.4f\n' % \
                  (s,sigma,alpha,beta,gamma,kappa,rb,goodness))
        f.close()
        return np.nan
    
def grid_gamma_rb (MBH=2.7e8,doalpha=True,glim=[0.0,0.025,40],rblim=[-3.0,0.05,40],alpha=2.0):
    global fitalpha
    if not doalpha:
        fitalpha = False
    os.system('rm gamma_rb.log')
    sigma = (MBH/0.00126)**(1/4.8)
    g=np.zeros((rblim[2],glim[2]))
    for irb in range(rblim[2]):
        for ig in range(glim[2]):
            g[irb,ig] = opt_gamma_rb (glim[0]+glim[1]*ig,rblim[0]+rblim[1]*irb,\
                                      sigma=sigma,alpha=alpha)
        plot_gamma_rb_fig(glim=glim,rblim=rblim)
        plt.savefig('gamma_rb_temp.png')

def func_rusinma (x0, *x):
    gal = [x[0],1.0,x[1],x0[1],0.01,x0[1]]
    im,r,defl,ddefl = setup (LOGSTART,LOGEND,LNUM,gal,True,x0[0])
    if len(im)>2:
        mu = np.sort(abs(im[:,-1]))
        fAB = np.log10(mu[-1]/mu[-2])
        sepAB = abs(im[-1,0]-im[0,0])
        sepA,sepB = abs(im[0,0]),abs(im[-1,0])
        goodness = 100.*((np.log10(FRAT)-fAB)**2 + \
                   (ASEP-sepA)**2/DERR**2 + (BSEP-sepB)**2/DERR*2)
    else:
        goodness = 1.0E9    
    return goodness

def grid_rusinma (MBH=2.7e8,glim=[0.0,0.025,40]):
    sigma = (MBH/0.00126)**(1/4.8)
    rm_AC = glim*0.0
    for g in glim:   # optimise s, kb
        x0,x = np.array([0.001,1.0]), (sigma, g)
        xopt = fmin (func_rusinma, x0, args=x)

def plot_gamma_rb_fig (logfile='gamma_rb.log',glim=[0.0,0.025,40],\
         rblim=[-3.0,0.05,40],c=[-1.,-2.,-3.,-4.,-5.],ls='solid',\
         lw=1,doylab=True,doclab=True,smooth=0.0):
    GAMMA,RB,VAL = 4,6,8
    gticks = np.array([0.0,0.5,1.0])
    gticks = gticks[(gticks>=glim[0])&(gticks<=glim[1]*glim[2])]
    rbticks = np.array([-3.0,-2.5,-2.0,-1.5,-1.0])
    rbticks = rbticks[(rbticks>=rblim[0])&(rbticks<=rblim[1]*rblim[2])]
    gscale = np.arange(glim[0],glim[0]+glim[1]*glim[2],glim[1])
    rbscale = np.arange(rblim[0],rblim[0]+rblim[1]*rblim[2],rblim[1])
    g = np.ones((len(rbscale),len(gscale)))*np.nan
    data = np.loadtxt(logfile)
    for d in data:
        rbidx = int(round((d[RB]-rblim[0])/rblim[1]))
        gidx = int(round((d[GAMMA]-glim[0])/glim[1]))
        if rbidx<0 or gidx<0 or rbidx>=len(rbscale) or gidx>=len(gscale):
            continue
        g[rbidx,gidx] = d[VAL]
    if smooth!=0.0:
        g=gaussian_filter(g,smooth)
    CS = plt.contour(g,c,colors='k',linestyles=ls,linewidth=lw)
    zc = CS.collections[0]
    plt.setp (zc,linewidth=lw)
    if doclab:
        plt.clabel(CS,inline=1,fontsize=10)
    plt.xticks((gticks-glim[0])/glim[1],gticks)
    plt.yticks((rbticks-rblim[0])/rblim[1],rbticks)
    plt.xlabel('Inner power law index')
    if doylab:
        plt.ylabel('log(Break radius/arcsec)')

def plot_twofig_gamma_rb():
    plt.subplot(121)
    plot_gamma_rb_fig('gamma_rb_runs/log_noBH_1.0',c=[-2.,-3.,-4.,-5],ls='dashed')
    plot_gamma_rb_fig('gamma_rb_runs/log_BH_1.0',c=[-2.],ls='solid')
    plot_gamma_rb_fig('gamma_rb_runs/log_BH_1.0',c=[-5.],ls='solid',doclab=False,lw=4)
    plt.plot([-2.99,-2.99],[0.01,0.01],'k-',linestyle='dashed',label='Without SMBH')
    plt.plot([-2.99,-2.99],[0.01,0.01],'k-',label='With SMBH')
    plt.plot([-2.99,-2.99],[0.01,0.01],'k-',linewidth=4,label='High SMBH demagnification')
    plt.xlim(0.0,40.0)
    plt.legend(fontsize=9,handlelength=5,loc=2)
    ax=plt.subplot(122)
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    plt.ylabel('log (Break radius/parsec)')
    plot_gamma_rb_fig('gamma_rb_runs/log_noBH_2.0',c=[-2.,-3.,-4.,-5],ls='dashed',doylab=False)
    plot_gamma_rb_fig('gamma_rb_runs/log_BH_2.0',c=[-2.,-3],ls='solid',doylab=False)
    plot_gamma_rb_fig('gamma_rb_runs/log_BH_2.0',c=[-5.],ls='solid',doclab=False,doylab=False,lw=4)
    ax.set_yticks([3.4,13.4,23.4,33.4])
    ax.set_yticklabels(['1.0','1.5','2.0','2.5'])
    plt.savefig('gamma_rb_twofig.png',bbox_inches='tight',clobber=True)

def plot_gamma_rb (gamma,rb,alpha=2.0,sigma=0.0,beta=1.0,doplot=False):
    gridsize = 25
    nim, fAB, fCA, sepAB = (np.nan*np.zeros((gridsize,gridsize)) for i in range(4))
    goodness = 1.0E9*np.ones((gridsize,gridsize))
    k0, kinc, s0, sinc, nk, ns = 0.0, 3./gridsize, 0.0, 2./gridsize, gridsize,gridsize
    for pk in range(nk):
        k = k0+kinc*pk
        for ps in range(ns):
            s = s0+sinc*ps
            gal = [sigma,alpha,beta,gamma,k,rb]
            im,r,defl,ddefl = setup (LOGSTART,LOGEND,LNUM,gal,True,s)
            nim[pk,ps]=len(im)
            if len(im)>2:
                mu = np.sort(abs(im[:,-1]))
                fAB[pk,ps] = np.log10(mu[-1]/mu[-2])
                fCA[pk,ps] = np.log10(mu[-3]/mu[-1])
                sepAB[pk,ps] = abs(im[-1,0]-im[0,0])
                goodness[pk,ps] = (sepAB[pk,ps]-ABSEP)**2 + \
                                  (fAB[pk,ps]-np.log10(FRAT))**2

    if doplot:
        plt.subplot(221)
        plt.imshow(goodness,vmin=0.0,vmax=10.0);plt.colorbar()
        plt.subplot(222)
        plt.imshow(fAB,vmax=min(2.0,fAB.max()));plt.colorbar()
        plt.title('log fA/fB')
        plt.subplot(223)
        plt.imshow(fCA);plt.colorbar()
        plt.title('log fC/fA')
        plt.subplot(224)
        plt.imshow(sepAB);plt.colorbar()
        plt.title('Sep A-B')
        plt.show() 
    np.putmask(goodness,np.isnan(goodness),1.0E9)           
    minwhere = measurements.minimum_position(goodness)
    return minwhere[1]*sinc,minwhere[0]*kinc

def constraints_fig(bfile):
    b = np.loadtxt(bfile)    # flux, separation, galparms
    b[:,5] = 116.0*b[:,2]/b[:,5]
    b[:,1] = 16000.0*10**b[:,1]
    b = b[b[:,1]>0.1]
    plt.semilogx(b[:,1],b[:,5],'bo')
    plt.semilogx(lims[:,1],lims[:,0],'r-')
    plt.semilogx(merlinlims[0],merlinlims[1],'r-')
    plt.xlabel ('Flux density of image C / microJy')
    plt.ylabel ('B-C image separation / mas')
    plt.xlim(0.5*b[:,1].min(),2.0*b[:,1].max())
    plt.ylim(0,125.0)
    plt.savefig('constraints_fig.png')
    plt.clf()
    a=np.load('result1.npy')
    plt.imshow(abs(a-23.0),cmap=matplotlib.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(8),(3.0+np.arange(8)))
    plt.yticks(np.arange(9),40.0+20.0*np.arange(9))
    plt.xlabel('SNR of image C')
    plt.ylabel('B-C image separation / mas')
    plt.savefig('constraints_col.png')
    plt.clf()
    
def comparison_fig(filename,params,dosame=False):
#  0 = source position  1 = log C/B  2 = raw dist B-C   3 = chisq  4 = raw G-A
#  5 = raw G-B  6 = FA/FB  7 = sigma  8 = alpha  9 = beta  10 = gamma  11 = kb  12 = rb 13=nim
#    xticks = [[],[],[],[],[],[],[],[7.5,8.0,8.5,9.0],[1.,1.5,2.0,2.5],[1.0,1.2,1.4,1.6,1.8,2.0],[-0.1,0.0,0.1,0.2,0.3],[1.,1.5,2.],[-2.5,-2.,-1.5,-1.]]
    xticks = [[],[],[],[],[],[],[],[7.5,8.0,8.5,9.0],[0.,1.,2.,3.,4.,5],[1.0,1.5,2.0],[0.0,0.2,0.4,0.6,0.8],[1.,1.5,2.],[-3.,-2.,-1.,0.]]
    matplotlib.rcParams['legend.handlelength'] = 0
    matplotlib.rcParams['legend.handleheight'] = 0
    b = np.loadtxt(filename)
    b[:,7]=8+np.log10(3.1*(b[:,7]/200.)**4)
    c0,c1 = b[b[:,1]<-10.0], b[b[:,1]>-10.0]
    p = ['','','','','','','','log (M(BH)/M(sun))','alpha','Outer index','Inner index',\
         'Convergence','log10(Break radius/arcsec)']
    ny = len(params)/2 + len(params)%2 if dosame else len(params)
    plotnum = 1
    for i in range(len(params)):
        matplotlib.rcParams.update({'font.size':10})
        plt.subplot(ny,2,plotnum)
        plt.hist (c1[:,params[i]])
        plt.yticks([])
        plt.xticks(xticks[params[i]])
        if not dosame:
            plt.subplot(ny,2,plotnum)
        plt.hist (c0[:,params[i]])
        matplotlib.rcParams.update({'font.size':10})
        plt.legend ([p[params[i]]])
        plotnum += 1
    plt.show()
    plt.clf()

def mktable2():
    cols=['sigma','alpha','beta','gamma','kappa','rb','s','chisq','ca','dist']
    a = np.load('mcmcruns/mcmc10_data.npy')
    aiso = np.load('mcmcruns/mcmc11_data.npy')
    a[:,0] = 4.8*np.log10(a[:,0])-2.9
    aiso[:,0] = 4.8*np.log10(aiso[:,0])-2.9
    a=a[a[:,8]<100000]
    a=a[~np.isnan(a[:,7])]
    aiso=aiso[aiso[:,8]<100000]
    aiso=aiso[~np.isnan(aiso[:,7])]    
    b=a[a[:,8]>-3.92]
    biso=aiso[aiso[:,8]>-3.92]
    c=a[a[:,8]<-3.92]
    ciso=aiso[aiso[:,8]<-3.92]
#    UP,LO=0.75,0.25
    UP,LO=0.8333,0.1667
    for i in range(len(cols)):
        A, Ai = a[:,i][~np.isnan(a[:,i])], aiso[:,i][~np.isnan(aiso[:,i])]
        B, Bi = b[:,i][~np.isnan(b[:,i])], biso[:,i][~np.isnan(biso[:,i])]
        C, Ci = c[:,i][~np.isnan(c[:,i])], ciso[:,i][~np.isnan(ciso[:,i])]
        A, Ai, C, Ci = np.sort(A), np.sort(Ai), np.sort(C), np.sort(Ci)
        B, Bi = np.sort(B), np.sort(Bi)
        fmt = '$%.2f^{+%.2f}_{-%.2f}$ & '
        allfmt = '$\\%s$ & '+6*fmt+'\n'
        print (allfmt % (cols[i],\
         np.median(A),A[int(UP*len(A))]-np.median(A),\
         np.median(A)-A[LO*len(A)],\
         np.median(B),B[int(UP*len(B))]-np.median(B),\
         np.median(B)-B[LO*len(B)],\
         np.median(C),C[int(UP*len(C))]-np.median(C),\
         np.median(C)-C[LO*len(C)],\
         np.median(Ai),Ai[int(UP*len(Ai))]-np.median(Ai),\
         np.median(Ai)-Ai[LO*len(Ai)],\
         np.median(Bi),Bi[int(UP*len(Bi))]-np.median(Bi),\
         np.median(Bi)-Bi[LO*len(Bi)],\
         np.median(Ci),Ci[int(UP*len(Ci))]-np.median(Ci),\
         np.median(Ci)-Ci[LO*len(Ci)]))
