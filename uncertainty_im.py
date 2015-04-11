'''
This makes visualizing errors in 2D data (like spatially-resolved 
	emission-line strengths and other IFU data) easier to do.

The new class `uncertainty_im()` just animates imshow such that values with 
	more uncertainty flicker randomly
'''

'''
Written by Zach Pace, U Wisconsin Dept. of Astronomy
zpace@astro.wisc.edu
Apr 2015
'''

class uncertainty_im(object):
	'''
	a fancier version of imshow()

	animates colors in imshow based on the given error of a specific data point
	
	Think of it like a flipbook image

	Arguments:
		- data: (n x m) array of measured values
		- err: (n x m) array of uncertainties (direct correspondence to 
			position in `data` array)
		- mask: (n x m) array of bools; True in a given position sets 
			that position as 'bad'
		- n: integer, essentially the number of pages in the flipbook
		- **kwargs: passed to imshow, like `cmap` and bad data values
	'''
	def __init__(self, data, err, mask, n = 100, title = None, save = False, **kwargs):

		import numpy as np
		import matplotlib.pyplot as plt
		from matplotlib import animation, cm
		import copy

		plt.ioff()

		self.figsize = kwargs.get('figsize', (6, 4))
		self.barlabel = kwargs.pop('barlabel', '')

		# First set up the figure, the axis, and the plot element we want to animate
		fig = plt.figure(figsize = self.figsize)

		err_m = err[np.newaxis, :] * np.random.randn(n, data.shape[0], data.shape[1])
		display_data = data[np.newaxis, :] + err_m

		#set up the cmap
		self.cmap = copy.copy(cm.cubehelix)
		self.cmap.set_bad('#545454', alpha = 1.)

		v_min = np.min(display_data)
		v_max = np.max(display_data)

		im = plt.imshow(np.ma.array(display_data[0], mask = mask), 
			interpolation = 'none', vmin = v_min, vmax = v_max, cmap = self.cmap, 
			origin = 'lower', **kwargs)

		plt.colorbar(im, shrink = 0.8, label = self.barlabel)
		if title != None: plt.title(title, size = 16)

		# initialization function: plot the background of each frame
		def init():
			im.set_data(np.ma.array(display_data[0], mask = mask))
			return [im]

		# animation function.  This is called sequentially
		def animate(i):
			im.set_array(np.ma.array(display_data[i], mask = mask))
			return [im]

		anim = animation.FuncAnimation(fig, animate, init_func = init, frames = n,
			interval = 200, blit = True)
		plt.show()
		if save == True: anim.save('animation.gif', writer = 'imagemagick', fps = 5)

def example():
	import numpy as np
	import matplotlib.mlab as mlab
	import matplotlib.pyplot as plt

	bounds = [-3., 3., -2., 2.]

	delta = 0.05
	x = np.arange(bounds[0], bounds[1], delta)
	y = np.arange(bounds[2], bounds[3], delta)
	X, Y = np.meshgrid(x, y)
	Z = 10.+ 10.*mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
	poisson = 0.05*np.sqrt(np.abs(Z))

	mask = np.zeros(Z.shape)
	mask[-25:, -25:] = True

	uncertainty_plot(Z, poisson, mask, extent = bounds, save = True, barlabel = 'label', title = 'Title')