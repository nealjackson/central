import numpy as np
import scipy as sp
import os,time
import pickle
from scipy import interpolate
from scipy import ndimage
from scipy import optimize
from scipy.optimize import fmin
from scipy.ndimage import measurements
import matplotlib.pyplot as plt
from scipy import fftpack as ff
import sys
try:
    import astLib
    from astLib import astCoords
    import pyfits
    from pyfits import getdata
except:
    pass

def addarray (a, apos, b, bpos, ismult=False):
    dpos = np.asarray(apos,dtype='float')-np.asarray(bpos)
    dfloor = np.floor(dpos)
    dceil = np.floor(dpos)+np.asarray(b.shape[::-1])
    frac = dpos-dfloor
    b1 = ndimage.shift(b,frac[::-1])
    c = np.copy(a)
    if any(dceil)<0.0 or dfloor[0]>a.shape[1] or dfloor[1]>a.shape[0]:
        print ('Warning: PSF found outside image', dfloor, '->', dceil)
        return c
    while dceil[0]>a.shape[1]:
        dceil[0]-=1.0
        b1=b1[:,:-1]
    while dceil[1]>a.shape[0]:
        dceil[1]-=1.0
        b1=b1[:-1,:]
    while dfloor[0]<0:
        dfloor[0]+=1.0
        b1=b1[:,1:]
    while dfloor[1]<0:
        dfloor[1]+=1.0
        b1=b1[1:,:]
    if ismult:
        c[dfloor[1]:dceil[1],dfloor[0]:dceil[0]]*=b1
        return c[dfloor[1]:dceil[1],dfloor[0]:dceil[0]].sum()
    else:
        c[dfloor[1]:dceil[1],dfloor[0]:dceil[0]]+=b1
        return c


def arraymap (a, new, order=1):
#   maps a 2-D array a to a new size: NB x=fast index
#   rescales fluxes to get same total pixel flux
    ax, ay = float(a.shape[1]), float(a.shape[0])
    nx, ny = float(new[0]), float (new[1])
    xc = np.tile (np.arange(0.0, ax-0.5*ax/nx, ax/nx)+0.5*ax/nx, ny)
    yc = np.repeat (np.arange(0.0, ay-0.5*ay/ny, ay/ny)+0.5*ay/ny, nx)
#    print ('Dimensions of xc ',xc.shape)
#    print ('Dimensions of yc ',yc.shape)
    allc = np.vstack ((yc,xc))
    newa = sp.ndimage.map_coordinates (a, allc, 
order=order).reshape(new[1],new[0]) \
           * float(ax*ay)/float(nx*ny)
    return newa


def arrayplot (a, nx=0, ny=0):
    if not nx or not ny or nx*ny>nim:
        nx=1+int(np.sqrt(len(a)))
        ny=1+len(a)/nx
    icou=0
    for ix in range(nx):
        for iy in range(ny):
            plt.subplot(nx,ny,icou+1)
            try:
                plt.imshow(getdata(a[icou]))
            except:
                pass
            icou+=1
            

def convgauss (im, fwhm, axrat=1.0, angle=0.0, ignore=4.0, flux=1.0):
    gsize = int(fwhm*ignore)+3
    gsize = 2*(gsize/2)
    a = mkgauss ([gsize,gsize],[gsize/2,gsize/2],flux,fwhm,axrat,angle,ignore)
    return sp.ndimage.convolve(im,a)


def convolve (im1, im2, mode=0, crit=1.0E-10):
# crit is the thing which stops the stuff getting noisy in the middle
# when deconvolving. Check that the default gives reasonable values!

# need to zero-pad up to the next power of 2 before using this!!
    im1 = zeropad(im1,0,0)
    im2 = zeropad(im2,im1.shape[0],im1.shape[1])

    fftim1 = ff.fft2 (im1)
    fftim2 = ff.fft2 (im2)
    fftout = np.copy (fftim2)
    if (mode):
        fftout.real = (fftim1.real*fftim2.real+fftim1.imag*fftim2.imag)/ \
                      (fftim2.real*fftim2.real+fftim2.imag*fftim2.imag)
        fftout.imag = (fftim1.imag*fftim2.real-fftim1.real*fftim2.imag)/ \
                      (fftim2.real*fftim2.real+fftim2.imag*fftim2.imag)
    else:
        fftout.real = fftim1.real*fftim2.real-fftim1.imag*fftim2.imag
        fftout.imag = fftim1.imag*fftim2.real+fftim1.real*fftim2.imag
    for i in range (fftout.shape[0]):
        for j in range (fftout.shape[1]):
            if ((i%2 and not j%2) or (not i%2 and j%2)):
                fftout[i,j] *= -1.0
            if abs(fftim2[i,j])<crit*abs(fftim2[0,0]):
                fftout[i,j] = 0.0
    np.putmask (fftout.imag, np.isnan(fftout.imag), 0.0)
    np.putmask (fftout.real, np.isnan(fftout.imag), 0.0)
    np.putmask (fftout.imag, np.isnan(fftout.real), 0.0)
    np.putmask (fftout.real, np.isnan(fftout.real), 0.0)
    return (ff.ifft2(fftout)).real

def edgenoise (a, n=10):
    b=a[a<np.median(np.ravel(a))]
    return np.sqrt(2.0)*np.std(b)
#    return (0.25*(np.std(a[0:n,:])+np.std(a[:,0:n])+\
#          np.std(a[-n:-1,:])+np.std(a[:,-n:-1])), \
#          0.25*(np.median(a[0:n,:])+np.median(a[:,0:n])+\
#          np.median(a[-n:-1,:])+np.median(a[:,-n:-1])),
#          0.25*(np.mean(a[0:n,:])+np.mean(a[:,0:n])+\
#          np.mean(a[-n:-1,:])+np.mean(a[:,-n:-1])))
 
def first_download (ra,dec,outfile='',gif=0,fits=1,imsize=2.0,\
    imserver='third.ucllnl.org'):
    format = 'hms'
    try:
        if np.isreal(ra):
            format = 'decimal'
    except:
        pass
    if format=='decimal':
        ra=astCoords.decimal2hms(ra,' ')
        dec=astCoords.decimal2dms(dec,' ')
    if outfile=='':
        outfile=ra.replace(' ','')+dec.replace(' ','')
        outfile=outfile + ('.fits' if fits==1 else '.gif')
    ra=ra.split()
    dec=dec.split()
    command = ('wget -O %s "http://%s/cgi-bin/firstimage?RA=%s%%20%s%%20%s%%20%s%%20%s%%20%s&Dec=&Equinox=J2000&ImageSize=%.1f&MaxInt=10&GIF=%d&FITS=%d&Download=1"'%(outfile,imserver,ra[0],ra[1],ra[2],dec[0],dec[1],dec[2],imsize,gif,fits))
    print (command)
    os.system(command)
    
    
#third.ucllnl.org/cgi-bin/firstimage?RA=08%2014%2025.89%20%2B29%2041%2015.6&
#Dec=&Equinox=J2000&ImageSize=4.5&MaxInt=10&FITS=1&Download=1

# ==================================================================
#
#   fitgauss inputs:
#        infile     input fits file
#        gu         guesses (ngauss x 4)  4 are x,y,flux,width
#                   default, use TV or estimate. Anything zero is estimated.
#        o          which things to optimise e.g. [[0,3],[1,3]] for widths
#                   of 1st 2 gaussians. Default: everything
#        docrt      show fits afterwards; -1 for no print
#        dotv       user to give cursor hits for initial gauss positions
#        ngauss     used if dotv=0 for points to try and find. Not very robust.
#        tiewidth
#
def fitgauss (infile,gu=np.array([]),o=[],docrt=0,dotv=1,ngauss=1,tiewidth=0):
    a = getdata (infile)
    b = edgenoise(a,5)
    a -= b[1]

    if (dotv):
        plt.imshow(a)
        cur = np.around(pcurs())
        ngauss = cur.shape[0]
    else:            # find brightest ngauss separated points, will be
                     # overwritten if user has supplied something
        aa = np.copy (a)
        temp = measurements.maximum_position (aa)
        cur=np.array([[np.float(temp[1]),np.float(temp[0])]])
        for i in range (1,ngauss):
            for j in range (aa.shape[0]):
                for k in range (aa.shape[1]):
                    xdist = cur[i-1][0]-k
                    ydist = cur[i-1][1]-j
                    if np.sqrt(xdist*xdist+ydist*ydist)<3.0:
                        aa[j,k]=0.0
            temp = measurements.maximum_position (aa)
            cur=np.vstack((cur,[temp[1],temp[0]]))
    if docrt > -1:
        print (cur)
        
# ----------------------------------------------------------------
    fluxes = np.zeros_like(cur[:,0])
    widths = np.zeros_like(fluxes)
    for i in range (ngauss):
        widths[i] = np.median([a[cur[i,1]+1,cur[i,0]],\
            a[cur[i,1]-1,cur[i,0]],a[cur[i,1],cur[i,0]+1],\
            a[cur[i,1],cur[i,0]-1]])
        widths[i] = np.sqrt(1.0/np.log(a[cur[i,1],cur[i,0]]/widths[i]))
        if widths[i]<1.5 or np.isnan(widths[i]):
            widths[i]=2.0
        fluxes[i] = a[cur[i,1],cur[i,0]]*np.pi*widths[i]*widths[i]
        
    g = np.hstack ((cur, np.array(fluxes,ndmin=2).transpose(),\
                         np.array(widths,ndmin=2).transpose()))
    for i in range (gu.shape[0]):
        for j in range (4):
            try:
                if gu[i,j] != 0.0:
                    g[i,j]=gu[i,j]
            except:
                pass

    if docrt > -1:
        print ('Initial guesses:',g)

    if o==[]:
        o=[0,0]
        for i in range(ngauss):
            for j in range(4):
                if i or j:
                    o=np.vstack((o,[i,j]))

    x0=[]
    for i in range (len(o)):
        x0 = np.append(x0,g[o[i][0],o[i][1]])
# ----------------------------------------------------------------
    xopt = fmin (fg_func,x0,args=[a.ravel(),a.shape,g,b,o,tiewidth],disp=0)
    m = np.zeros_like(a)
    for i in range (g.shape[0]):
        for j in range (len(o)):
            if o[j][0]==i:
                g[i][o[j][1]]=xopt[j]
        m = m+mkgauss(a.shape[::-1], (g[i][0],g[i][1]), g[i][2], g[i][3])

    if (docrt>0):
        print (xopt, ((a-m)*(a-m)).sum(), (a-m).max(),(a-m).min())
        plt.subplot(221);plt.imshow(a);plt.colorbar()
        plt.subplot(222);plt.imshow(m);plt.colorbar()
        plt.subplot(223);plt.imshow(a-m);plt.colorbar();plt.show()
# ----------------------------------------------------------------
    if docrt > -1:
        print ('Final array:',g)
    return xopt, g

def fg_func (x0, *x):
    ny = x[1][0]
    nx = x[1][1]
    a = (np.array(x[0])).reshape(ny,nx) 
    g = x[2]
    m = np.zeros_like(a)
    penalty = 0.0
    
    for i in range (g.shape[0]):
        p = g[i]
        for j in range (len(x[4])):
            if x[4][j][0]==i:
                p[x[4][j][1]]=x0[j]
        m = m+mkgauss(a.shape[::-1], (p[0],p[1]), p[2], p[3])
        if i and x[5]>0:
            penalty += 100.0*np.exp((g[i][3]-g[i-1][3])*(g[i][3]-g[i-1][3]))
 
    goodness = ((a-m)*(a-m)).sum()/(x[3][0]*x[3][0]*float(nx)*float(ny))
    return goodness+penalty

# ==================================================================

def forcefloat (a,defaultval=-1.0):
#   forcibly change an array to float, substituting strings with defaultval
    b=np.copy(a.ravel())
    c=np.zeros(b.shape,dtype=np.float32)+defaultval
    for i in range (b.size):
        try:
            c[i]=float(b[i].strip())
        except:
            pass
    c=c.reshape(a.shape)
    return c


def getradec (filename, startcol, ncol=0, deletechars=''):
# read ncol characters from a file where coordinates start at startcol
    if not ncol:
        ncol=startcol+6
    a=np.genfromtxt(filename,dtype='S20',usecols=np.arange(ncol),\
                    deletechars=deletechars)
    b=forcefloat(a)
    ra=np.fabs(b[:,startcol])*15.0+b[:,startcol+1]/4.0+b[:,startcol+2]/240.0
    dec=np.fabs(b[:,startcol+3])+b[:,startcol+4]/60.0+b[:,startcol+5]/3600.0
    for i in range(len(dec)):
        if (a[i,startcol+3].rfind('-')>-1):
            dec[i]*=-1.0
    return ra,dec,a,b

def getdegperpix (hdu):
    try:
         cdelt2 = hdu[0].header['CDELT2']
    except:
         try:
             cdelt2 = hdu[0].header['CD1_2']*hdu[0].header['CD1_2']+ \
                         hdu[0].header['CD1_1']*hdu[0].header['CD1_1']
         except:
             pass
    cdelt1 = -cdelt2
    crval1 = hdu[0].header['CRVAL1']
    crval2 = hdu[0].header['CRVAL2']
    crpix1 = hdu[0].header['CRPIX1']
    crpix2 = hdu[0].header['CRPIX2']
    naxes=[hdu[0].header['NAXIS1'],hdu[0].header['NAXIS2']]
    return cdelt1, cdelt2, crval1, crpix1, crval2, crpix2, naxes
 
def jiggle (im1, im2, docrt=0):
    x = (im1,im2)
    x0 = [0.0,0.0,1.0]
    xopt = fmin (jiggle_func, x0, args=x, disp=0)
    shiftim = xopt[2]*ndimage.shift (im2, (xopt[0],xopt[1]))
    if (docrt):
        print ('Optimal jiggle', xopt[0],xopt[1],xopt[2],jiggle_func(xopt,*x))
    return shiftim

def jiggle_func (x0, *x):
    shiftim = x0[2]*ndimage.shift (x[1], (x0[0],x0[1]))
    diff = x[0]-shiftim
    return (diff*diff).sum()

def maxfit_func1 (x0, *x):
    goodness = 0.0
    if x0[0]<0.5 or x0[0]>1.5 or x0[1]<0.5 or x0[1]>1.5:
        return 1.0E9
    expect=np.zeros((3,3))
    data = np.asarray(x).reshape(3,3)
    for j in range (0,3):
        for i in range (0,3):
            dist = np.hypot(i-x0[0],j-x0[1])
            expect[j,i] = x0[2] - x0[3]*dist**2
            goodness += (data[j,i]-expect[j,i])**2
#    print (x0,goodness)
#    print (expect)
    return goodness

def maxfit (a, noisy=True, guess=[0,0], maxoff=5.0):
    if guess==[0,0]:
        guess=[a.shape[1]/2,a.shape[0]/2]
#   guess in (x,y) order; pos in (y,x)
    pos = np.array([guess[1]-maxoff,guess[0]-maxoff]) + \
         np.array(measurements.maximum_position\
     (a[guess[1]-maxoff:guess[1]+maxoff,guess[0]-maxoff:guess[0]+maxoff]))
    status = True
    x = a[pos[0]-1:pos[0]+2,pos[1]-1:pos[1]+2].ravel()
    x0 = [1.0,1.0,x[4],x[4]-x[3]]
    xopt = fmin (maxfit_func1, x0, args=x, disp=0)
#    x = a[pos[0]-1:pos[0]+2,pos[1]-1:pos[1]+2].ravel()
#    x0 = np.zeros(8)
#    gd = x[5]-x[4]
#    x0 = [x[4],0.0,0.0,0.0,0.0, (x[7]-x[4]), 1.0, 1.0]
##    x1,x2 = -a[pos[0],pos[1]-1]+a[pos[0],pos[1]],\
##           -a[pos[0],pos[1]+1]+a[pos[0],pos[1]]
##    y1,y2 = -a[pos[0]-1,pos[1]]+a[pos[0],pos[1]],\
##            -a[pos[0]+1,pos[1]]+a[pos[0],pos[1]]
##    xoff = -0.5+x1/(x1+x2)
##    yoff = -0.5+y1/(y1+y2)
##    out = np.array([pos[1]+xoff,pos[0]+yoff])
#    xopt = fmin (maxfit_func, x0, args=x, disp=0)
#    out = np.array([pos[1]+xopt[6]-1.0, pos[0]+xopt[7]-1.0])
#    xdiff = np.abs (xopt[6]-1.0)
#    ydiff = np.abs (xopt[7]-1.0)
#    if xdiff > 1.0 or ydiff > 1.0:
#        if noisy:
#            print ('Warning: solution for maximum failed')
#        out[0] = pos[1]
#        out[1] = pos[0]
#        status = False
    return [xopt[0]+pos[1]-1,xopt[1]+pos[0]-1],status


def maxfit_func(x0, *x):
    goodness = 0.0
    expect=np.zeros(9)
    for j in range (0,3):
        for i in range (0,3):
            ix = float(i)-x0[6]
            iy = float(j)-x0[7]
            expect[3*j+i] = x0[0]+x0[1]*ix+x0[2]*iy+x0[3]*ix*iy+x0[4]*ix*ix+x0[5]*iy*iy
            goodness = goodness+(expect[3*j+i]-x[3*j+i])*(expect[3*j+i]-x[3*j+i])
    print (x0,goodness)
    return goodness

def mkdecon (posa,posb):
    a = mkgauss ([1024,1024],[posa,posa], 1.0, 20.0)
    b = mkgauss ([1024,1024],[posb,posb], 1.0, 5.0)
    c = convolve (a, b, mode=1)
    return c

def mkpgauss (dist, flux, fwhm, axrat=1.0, angle=0.0):
    fwhm /= 1.66667
    sinth = np.sin(angle*np.pi/180.0)
    costh = np.cos(angle*np.pi/180.0)
    r = np.array([-sinth,costh,-costh,-sinth])
    rt = np.array([-sinth,-costh,costh,-sinth])
    sig = np.array([fwhm,0.0,0.0,fwhm*axrat])
    scr1 = mxmul (sig,r)
    scr2 = mxmul (rt, scr1)
    scr1 = mxinv (scr2)
    ex = scr1[0]*dist[0]+scr1[1]*dist[1]
    ey = scr1[2]*dist[0]+scr1[3]*dist[1]
    return (flux/axrat)*np.exp(-(ex*ex+ey*ey))/(fwhm*fwhm*np.pi)

def mkgauss (naxes,pos,flux,fwhm,axrat=1.0,angle=0.0,ignore=4.0,dodist=False):
# note that total flux = peak flux in a pixel * 1.1331*FWHM**2
# angle is major axis East of North
    a = np.zeros (naxes[0]*naxes[1]).reshape(naxes[1],naxes[0])
    fwhm /= 1.66667
    if axrat==1.0 and angle==0.0:
        for i in range (naxes[1]):
            ydist=float(i)-pos[1]
            for j in range (naxes[0]):
                xdist=float(j)-pos[0]
                if xdist*xdist+ydist*ydist>ignore*ignore*fwhm*fwhm:
                    continue
                if not dodist:
                    a[i,j] = flux*np.exp(-(xdist*xdist+ydist*ydist)/ \
                                    (fwhm*fwhm))/(fwhm*fwhm*np.pi)
                else:
                    a[i,j] = np.hypot(xdist,ydist)
        return a
    sinth = np.sin(angle*np.pi/180.0)
    costh = np.cos(angle*np.pi/180.0)
    r = np.array([-sinth,costh,-costh,-sinth])
    rt = np.array([-sinth,-costh,costh,-sinth])
    sig = np.array([fwhm,0.0,0.0,fwhm*axrat])
    scr1 = mxmul (sig,r)
    scr2 = mxmul (rt, scr1)
    scr1 = mxinv (scr2)
    for i in range(naxes[1]):
        ydist=float(i)-pos[1]
        if abs(ydist)>ignore*fwhm:
            continue
        for j in range (naxes[0]):
            xdist = float(j) - pos[0]
            if abs(xdist)>ignore*fwhm:
                continue
            ex = scr1[0]*xdist+scr1[1]*ydist
            ey = scr1[2]*xdist+scr1[3]*ydist
            if not dodist:
                a[i,j] = (flux/axrat)*np.exp(-(ex*ex+ey*ey))/(fwhm*fwhm*np.pi)
            else:
                a[i,j] = np.hypot(ex,ey)/1.6666667

    return a

def mxmul(a,b):
    output=np.zeros(4)
    output[0]=a[0]*b[0]+a[1]*b[2]
    output[1]=a[0]*b[1]+a[1]*b[3]
    output[2]=a[2]*b[0]+a[3]*b[2]
    output[3]=a[2]*b[1]+a[3]*b[3]
    return output

def mxinv(a):
    det=a[0]*a[3]-a[1]*a[2]
    output=np.array([a[3],-a[1],-a[2],a[0]])/det
    return output


def npolyfit_func (x0, *x):
    dx=np.load('tmp_x.npy')
    dy=np.load('tmp_y.npy')
    derr=np.load('tmp_err.npy')
    chisq=0.0
    for i in range(len(dx)):
        expected=0.0
        if np.isinf(dy[i]) or np.isnan(dy[i]):
            continue
        for j in range(1+x[0]):
            expected+=x0[j]*dx[i]**float(j)
        chisq+=(dy[i]-expected)**2.0/(derr[i]**2.0)
    print (x0, chisq/float(len(dx)-len(x0)))
    return chisq/float(len(dx)-len(x0))

# compensates for the baffling lack of a polynomial fitting routine in
# scipy which handles errors per data point. Returns the optimised array
# of polynomial parameters, the chi-sq, and the modelled data.

def npolyfit (x, y, err, degree):
    x0=np.zeros(degree+1)
    x0[0]=y[0]
    np.save('tmp_x',x)
    np.save('tmp_y',y)
    np.save('tmp_err',err)
    xopt, fopt, iter, funcalls, warnflag = \
          fmin (npolyfit_func, x0, [degree], full_output=1)
    os.system('rm tmp_x.npy');os.system('rm tmp_y.npy')
    model=np.copy(y)
    for i in range(len(model)):
        model[i]=0.0
        for j in range(degree+1):
            model[i]+=xopt[j]*x[i]**float(j)
    return xopt, fopt, model

def nstack(a,b):
    if not len(a) and len(b):
        return b
    if not len(b) and len(a):
        return a
    if a.ndim==1 and b.ndim==1:
        if len(a)==len(b):
            return np.vstack((a,b))
        else:
            print ('Dimensions inconsistent')
            return []
    if a.ndim==2 and b.ndim==1:
        return np.vstack((a,b))
    if a.ndim==1 and b.ndim==2:
        return np.vstack((b,a))
    if a.shape[-2]!=b.shape[-2] or a.shape[-1]!=b.shape[-1]:
        print ('Dimensions inconsistent')
        return []
    c=np.append(np.ravel(a),np.ravel(b))
    extradim=len(c)/(a.shape[-1]*a.shape[-2])
    c=c.reshape(extradim,a.shape[-2],a.shape[-1])
    return c

def pcurs():
    os.system ('rm -f click.tmp')
    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    if "test_disconnect" in sys.argv:
        print ("disconnecting console coordinate printout...")
        plt.disconnect(binding_id)
    plt.show()
    try:
        return np.array(np.loadtxt('click.tmp'),ndmin=2)
    except:
        return []

def printmx(m):
    
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            print (m[i,j],' ')
        print ('\n')


def on_move(event):
    pass

def on_click(event):
    x, y = event.x, event.y
    if event.button==1:
        if event.inaxes is not None:
            f = open ('click.tmp', 'a')
            f.write ('%f %f\n' % (event.xdata, event.ydata))
            f.close ()

def subim(a,h,blc,trc):   # subset of image a, with header h; blc, trc as aips
    asub = a[blc[1]-1:trc[1],blc[0]-1:trc[0]]
    hsub = h.copy()
    hsub['NAXIS1'] = trc[0]-blc[0]+1
    hsub['NAXIS2'] = trc[1]-blc[1]+1
    hsub['CRPIX1'] = h['CRPIX1'] - blc[0]+1
    hsub['CRPIX2'] = h['CRPIX2'] - blc[1]+1
    return asub, hsub

def zeropad (a,tox=0,toy=0):
    if tox==0:
        tox=1
        while tox < a.shape[1]:
            tox*=2
    if toy==0:
        toy=1
        while toy < a.shape[0]:
            toy*=2

    b = np.array(np.zeros(tox*toy).reshape(toy,tox))
    c = addarray (b,[tox/2,toy/2],a,[a.shape[1]/2,a.shape[0]/2])
    return c

# routine to find the coordinates of an input file from the coordinates
# of an output file. Assumes that the pyfits headers are available as
# hin.npy and hout.npy.

def ast_map (outpos):
    outpos = np.asarray(outpos,dtype='float')
    inpos = np.load('inpos.npy')
    try:
        val = (inpos[1,outpos[1],outpos[0]],inpos[0,outpos[1],outpos[0]])
    except:
        val = (0,0)
    return val

def ast_getmap (inhead, outhead, outs, incoord=[]):
    hout = pickle.load(open(outhead,'rb'))
    hin = pickle.load(open(inhead,'rb'))
    crpix1, crpix2 = hout['CRPIX1']-1.0, hout['CRPIX2']-1.0   #convert to index
    crval1, crval2 = hout['CRVAL1'], hout['CRVAL2']
    cdec = np.cos(np.deg2rad(crval2))
    y,x = np.meshgrid (np.arange(outs[1])-crpix2,np.arange(outs[0])-crpix1)
    wcs = np.zeros((2,outs[0],outs[1]))
    try:
        cdelt1, cdelt2 = hout['CDELT1'], hout['CDELT2']
        wcs = np.array([crval1+x*cdelt1/cdec, crval2+y*cdelt2])
    except:
        cd = np.array([hout['CD1_1'],hout['CD1_2'],hout['CD2_1'],hout['CD2_2']])
        wcs = np.array([crval1+(x*cd[0]+y*cd[1])/cdec, crval2+x*cd[2]+y*cd[3]])
    np.save('wcs',wcs)
#   then convert from WCS to input
    crpix1, crpix2 = hin['CRPIX1']-1.0, hin['CRPIX2']-1.0   #convert to index
    crval1, crval2 = hin['CRVAL1'], hin['CRVAL2']
    if len(incoord)==4:
        crpix1,crpix2,crval1,crval2 = incoord
        crpix1 -= 1.0; crpix2 -= 1.0
    try:    # does the input have DELT? If not try for CD matrix
        cdelt1, cdelt2 = hin['CDELT1'], hin['CDELT2']
        inpos = np.array([crpix1+(wcs[0]-crval1)*cdec/cdelt1,\
                          crpix2+(wcs[1]-crval2)/cdelt2])
    except:
        cd = np.array([hin['CD1_1'],hin['CD1_2'],hin['CD2_1'],hin['CD2_2']])
        cdi = mxinv (cd)
        x,y = (wcs[0]-crval1)*cdec, wcs[1]-crval2
        inpos = np.array([crpix1+cdi[0]*x+cdi[1]*y,\
                          crpix2+cdi[2]*x+cdi[3]*y])
    np.save('inpos',inpos)

def ast_register(infile,outfile,inext=0,outext=0,incoord=[]):
    inpoint = pyfits.open(infile)
    outpoint = pyfits.open(outfile)
    if not inext:
        for i in range(len(inpoint)):
            try:
                test = inpoint[i].header['CRVAL1']
                inext = i
                break
            except:
                pass
    if not outext:
        for i in range(len(outpoint)):
            try:
                test = outpoint[i].header['CRVAL1']
                outext = i
                break
            except:
                pass
    print ('Read in=%s[%d], out=%s[%d]' % (infile,inext,outfile,outext))
    indata = inpoint[inext].data
    while indata.ndim>2:
        indata=indata[0]
    inheader = inpoint[inext].header
    outdata = outpoint[outext].data
    while outdata.ndim>2:
        outdata=outdata[0]
    outheader = outpoint[outext].header
    pickle.dump (inheader, open('hin.npy','wb'))
    pickle.dump (outheader, open('hout.npy','wb'))
    os.system('rm inpos.npy')
    ast_getmap('hin.npy','hout.npy',np.asarray(outdata.shape),incoord=incoord)
    register = ndimage.geometric_transform (indata,ast_map,output_shape=outdata.shape)
    return register
