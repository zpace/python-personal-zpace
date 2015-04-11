class uncertainty_plot(object):
	'''
	a fancier version of imshow()

	animates colors in imshow based on the given error of a specific data point
	
	Think of it like a flipbook image

	Arguments:
		- data: (n x m) array of measured values
		- err: (n x m) array of unceertainties (direct correspondence to 
			position in `data` array)
		- n: integer, essentially the number of pages in the flipbook
		- **kwargs: passed to imshow, like `cmap` and bad data values
	'''
	def __init__(self, data, err, n = 100, title = None, **kwargs):

		import numpy as np
		import matplotlib.pyplot as plt
		from matplotlib import animation

		plt.ioff()

		self.figsize = kwargs.get('figsize', (6, 4))
		self.barlabel = kwargs.pop('barlabel', '')

		# First set up the figure, the axis, and the plot element we want to animate
		fig = plt.figure(figsize = self.figsize)

		err_m = err[np.newaxis, :] * np.random.randn(n, data.shape[0], data.shape[1])
		display_data = np.tile(data, (n, 1, 1)) + err_m

		v_min = np.min(display_data)
		v_max = np.max(display_data)
		im = plt.imshow(data + np.random.randn(data.shape[0], data.shape[1]), interpolation = 'none', vmin = v_min, vmax = v_max, cmap = 'cubehelix', **kwargs)
		plt.colorbar(im, shrink = 0.8, label = self.barlabel)
		if title != None: plt.title(title, size = 16)

		# initialization function: plot the background of each frame
		def init():
			im.set_data(display_data[0])
			return [im]

		# animation function.  This is called sequentially
		def animate(i):
			im.set_array(display_data[i])
			return [im]

		anim = animation.FuncAnimation(fig, animate, init_func = init, frames = n,
			interval = 200, blit = True)

		plt.show()

def example():
	import numpy as np
	import matplotlib.mlab as mlab

	bounds = [-3., 3., -2., 2.]

	delta = 0.025
	x = np.arange(bounds[0], bounds[1], delta)
	y = np.arange(bounds[2], bounds[3], delta)
	X, Y = np.meshgrid(x, y)
	Z = 10.+ 10.*mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
	poisson = 0.05*np.sqrt(np.abs(Z))

	#uncertainty_plot(np.ones(4).reshape((2,2)), 0.1*np.ones(4).reshape((2,2)))
	uncertainty_plot(Z, poisson, extent = bounds, barlabel = 'label', title = 'Title')