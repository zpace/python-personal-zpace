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
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    import pylab as plt
    #import matplotlib.colors as colors

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

def spspk_overlay(ctr = [0., 0.], angle = 0., **kwargs):
    '''
    overlay a SparsePak-shaped pattern on a figure (usually an image)

    wraps around `circles` (above)

    You can manipulate the bundle center and the angle of the bundle (wrt vertical)
    by setting ctr and CCW angle (in radians) nonzero.
    '''

    import matplotlib.pyplot as plt
    import astropy.io.ascii as ascii
    import numpy as np

    #start out by reading in spspk fiber data, ordered by fiber data row
    #fiber data row is just a 0-indexed list that doesn't include sky fibers
    #so fibers are numbered 0-74 rather than 1-82
    
    fibers = ascii.read('fiberdata.dat')
    fibers.sort('row')

    coords = np.column_stack((fibers['ra'], fibers['dec']))
    print coords

    rot_matrix = np.array( [ [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)] ])
    coords = np.dot(coords, rot_matrix.T)
    fibers['ra'] = coords[:, 0]
    fibers['dec'] = coords[:, 1]

    circles(ctr[0] + fibers['ra'], ctr[1] + fibers['dec'], 2.5, **kwargs)
    plt.xlim([-40., 40])
    plt.ylim([-40., 40])