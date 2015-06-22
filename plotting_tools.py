import numpy as np
import numpy.random as r
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import astropy.io.ascii as ascii
import numpy as np
from numpy.random import choice
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV

def circles(x, y, s, c='b', ax=None, vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence 
    like objects of the same lengths. The size of circles are in data scale.
    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, ) 
        Radius of circle in data scale (ie. in data unit)
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or
        RGBA sequence because that is indistinguishable from an array of
        values to be colormapped.  `c` can be a 2-D array in which the
        rows are RGB or RGBA, however.
    ax : Axes object, optional, default: None
        Parent axes of the plot. It uses gca() if not specified.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.  (Note if you pass a `norm` instance, your
        settings for `vmin` and `vmax` will be ignored.)
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Other parameters
    ----------------
    kwargs : `~matplotlib.collections.Collection` properties
        eg. alpha, edgecolors, facecolors, linewidths, linestyles, norm, cmap
    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    """

    '''
    Credit: StackOverflow user Sub Struct
        (http://stackoverflow.com/questions/9081553/python-scatter-plot-size-and-style-of-the-marker/24567352#24567352)
    '''

    if ax is None:
        ax = plt.gca()    

    if isinstance(c,basestring):
        color = c     # ie. use colors.colorConverter.to_rgba_array(c)
    else:
        color = None  # use cmap, norm after collection is created
    kwargs.update(color=color)

    if 'edgecolor' not in kwargs: kwargs.update(edgecolor = 'k')
    #print kwargs

    if isinstance(x, (int, long, float)):
        patches = [Circle((x, y), s),]
    elif isinstance(s, (int, long, float)):
        patches = [Circle((x_,y_), s) for x_,y_ in zip(x,y)]
    else:
        patches = [Circle((x_,y_), s_) for x_,y_,s_ in zip(x,y,s)]
    collection = PatchCollection(patches, **kwargs)

    if color is None:
        collection.set_array(np.asarray(c))
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection)
    return collection

def spspk_overlay(fibers = None, ctr = [0., 0.], angle = 0., which = 'science', **kwargs):
    '''
    overlay a SparsePak-shaped pattern on a figure (usually an image)
    wraps around `circles` (above)
    You can manipulate the bundle center and the angle of the bundle (wrt vertical)
    by setting ctr and CCW angle (in radians) nonzero.
    '''

    plt.ioff()

    #start out by reading in spspk fiber data, ordered by fiber data row
    #fiber data row is just a 0-indexed list that doesn't include sky fibers
    #so fibers are numbered 0-74 rather than 1-82

    if fibers == None:
        fibers = ascii.read('fiberdata.dat')

    fibers.sort('row')

    if which == 'science':
        fibers = fibers[fibers['row'] != -9999]
    elif which == 'sky':
        fibers = fibers[fibers['row'] == -9999]
    elif which == 'all':
        fibers = fibers
    else:
        raise ValueError('Invalid fiber subset')

    coords = np.column_stack((fibers['ra'], fibers['dec']))
    #print coords

    rot_matrix = np.array( [ [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)] ])
    coords = np.dot(coords, rot_matrix.T)
    fibers['ra'] = coords[:, 0]
    fibers['dec'] = coords[:, 1]

    circles(ctr[0] + fibers['ra'], ctr[1] + fibers['dec'], 2.5, **kwargs)

    field_width = np.max(ctr[0] + fibers['ra']) - np.min(ctr[0] - fibers['ra'])
    field_height = np.max(ctr[1] + fibers['dec']) - np.min(ctr[1] - fibers['dec'])

    #print field_width, field_height

    plt.axis('equal')

def gaussian(x, m = 0., s = 1.):
    return 1. / (s * np.sqrt(2.* np.pi)) * np.exp( -0.5*((x - m)/s)**2. )

def kde_errors(x, bandwidth, objname, e = None, cv = 3, offset = 0.):
    '''
    use non-parametric density estimation (KDE) to find an underlying probability density function
    measurement errors can be added as necessary.
    If none are added, then this reduces to a KDE function with built-in cross-validation
    '''

    xoffset = x + offset

    grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidth}, cv = cv)
    grid.fit(xoffset[:, None])
    print grid.best_params_

    xw = np.ptp(xoffset)
    x_grid = np.linspace(xoffset.min() - 0.25*xw, xoffset.max() + 0.25*xw, 10000)

    plt.close('all')

    plt.figure(figsize = (6, 6))

    if e == None:
        kde = grid.best_estimator_
        pdf = np.exp(kde.score_samples(x_grid[:, None]))
        #print pdf
        plt.plot(x_grid, pdf, linewidth=3, alpha=0.5)
    
    else:
        '''this produces weird results'''
        bw = grid.best_params_['bandwidth']
        e_corr = e
        e_corr[e_corr == 0] = np.min(e[e != 0.])
        bw_new = np.sqrt(e**2. + bw**2.)
        #bw_new = bw*np.ones(len(x))
        pdf = np.zeros(len(x_grid))
        #print zip(x, bw_new)
        for (m, s) in zip(xoffset, bw_new):
            pdf += gaussian(x_grid, m, s)
        plt.plot(x_grid, pdf, linewidth=3, alpha=0.5)

    x_sys = x_grid[np.argmax(pdf)]

    #now sample the pdf to estimate the stdev of the distribution about the maximum likelihood
    pdf_sample = choice(a = x_grid, p = pdf/pdf.sum(), replace = True, size = 100000)
    dx_sys = np.sqrt(np.mean((pdf_sample - x_sys)**2.))

    dx_sys = xw/(2.*np.sqrt(len(xoffset)))

    plt.scatter(xoffset, -.05*np.max(pdf)*np.ones(len(xoffset)), marker = 'x', 
        color = 'k', alpha = 0.5)

    #print 'Systemic velocity:', V_sys
    plt.axvline(x_sys, linestyle = '--', c = 'r')

    plt.title(objname + ' radial velocity', size = 18)
    plt.xlim([x_grid.min(), x_grid.max()])

    plt.text(x = x_sys + .01*xw, y = 0.55 * np.max(pdf), 
        s = '$V = ${:5.0f}'.format(x_sys) + ' +/-{:5.0f} km/s'.format(dx_sys), 
        rotation = 'vertical')

    plt.xlabel('Radial Velocity [km/s]', size = 18)
    plt.ylabel('Probability', size = 18)
    plt.tight_layout()
    plt.show()

    return x_sys, dx_sys

def rejection_sample_2d(x, y, z, nper = 100):
    '''
    sample `nper` times from each bin of 2D PDF `z`
    '''

    #normalize
    z /= z.sum()
    #print z.shape
    mc = r.random(z.shape + (nper,))
    selected = mc < z[:,:,np.newaxis]
    number = selected.sum(axis = -1)
    xx, yy = np.meshgrid(x, y)
    coords = np.dstack((xx, yy))

    #print np.repeat(coords.reshape(len(number), 2), number.flatten()[:,np.newaxis])

    #sample = np.column_stack(( np.repeat(xx.flatten(), number.flatten()), np.repeat(xx.flatten(), number.flatten()) ))
    #both of these are the same
    sample = np.repeat(coords.reshape(-1, 2), number.ravel(), axis = 0)
    return sample

def find_linefit_CI(mb, best, xr, a, plot = False):
    '''
    find the credible interval of a MC line fit

    Arguments:
     - mb: array(-like) of fit instances (col for m, col for b)
     - best: best-fit (in form [mbest, bbest])
     - xr: x-range of fit (in form [xllim, xulim])
     - a: "alpha" (e.g., .05 for 95%)
     - plot: bool
    '''
    x = np.linspace(xr[0], xr[1], 100)

    # first create an array with each row being an instance of the fit
    nfits = np.array([m*x + b for m, b in mb])
    #now compute the x-point-wise
    CI_n = np.percentile(nfits, q = [100.*a/2., 100.*(1 - a/2.)], axis = 0)

    if plot == True:
        plt.fill_between(x, y1 = CI_n[0], y2 = CI_n[1], 
            color = 'grey', zorder = 0, alpha = .5)
        plt.plot(x, best[0]*x + best[1], c = 'r', lw = 2, zorder = 1)
        
    return x, CI_n