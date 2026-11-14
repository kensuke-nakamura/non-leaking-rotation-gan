import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torchvision.transforms as transforms
import numpy as np
import sys
from PIL import Image

import matplotlib
matplotlib.use('Agg')


def normalize_tensor(data, target_min=0, target_max=1):
	data_reshape = data.view(data.shape[0],-1)
	data_min     = torch.min(data_reshape, dim=1).values.unsqueeze(1)
	data_max     = torch.max(data_reshape, dim=1).values.unsqueeze(1)
	data_reshape = (data_reshape - data_min) / (data_max - data_min)
	data_reshape = (target_max - target_min) * data_reshape + target_min 
	return data_reshape.view(data.shape)

'''
def normalize_tensor(data, target_min=0, target_max=1):
	data_reshape = data.flatten()
	data_min     = torch.min(data_reshape)
	data_max     = torch.max(data_reshape)
	data_reshape = (data_reshape - data_min) / (data_max - data_min)
	data_reshape = (target_max - target_min) * data_reshape + target_min 
	return data_reshape.view(data.shape)
'''


def normalize_tensor_2d(data, target_min=0, target_max=1):
	print('warning: normalize_tensor_2d() was integraed to normalize_tensor().')
	return normalize_tensor(data, target_min, target_max)


#
# Example:
#
# python3 visualize.py
#
def plot_curve_errorbar(values, errs, labels, 
colors=['red','blue','green','y','magenta'], 
alphas=[1,1,1,1,1],
linewidths=[1,1,1,1,1],
linestyles=None, #['-','-','-','-','-']
x_intervals=None, 
xlabel=None, ylabel=None, 
ylim=None, 
fontsize_xylabel=24,
fontsize_ticks=14,
fontsize_legend=14,
horizontal_line=None,
legend=True, grid=False, filename=None):
	plt.cla()
	fig,ax = plt.subplots()

	# When plot data are given as list OR numpy array with more than 2 data
	if isinstance(values,list) or (isinstance(values, np.ndarray) and len(values.shape)>1): 
		for i in range(len(values)):
			y     = values[i]
			if x_intervals is not None: x_interval = x_intervals[i]
			else:                       x_interval = 1
			if linestyles  is not None: linestyle = linestyles[i]
			else:                       linestyle = '-'
			x     = x_interval * np.arange(len(y))
			label = labels[i]
			color = colors[i]
			alpha = alphas[i]
			linewidth = linewidths[i]
			ax.plot(x, y, label=label, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
			if errs is not None:
				err   = errs[i]
				ax.fill_between(x, y-err, y+err, color=color, alpha=0.30)
	else: 
			if x_intervals is not None: x_interval = x_intervals[0]
			else:                       x_interval = 1
			y     = values
			x     = x_interval * np.arange(len(y))
			label = labels
			color = colors[0]
			alpha = alphas[0]
			linewidth = linewidths[0]
			if linestyles  is not None: linestyle = linestyles[0]
			else:                       linestyle = '-'
			ax.plot(x, y, label=label, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
			if errs is not None:
				err   = errs
				ax.fill_between(x, y-err, y+err, color=color, alpha=0.30)

	ax.tick_params(axis='both', labelsize=fontsize_ticks)
	if horizontal_line is not None: ax.axhline(y=horizontal_line,color='black',linestyle='--',zorder=-99) 
	if xlabel is not None:   ax.set_xlabel( xlabel, fontsize=fontsize_xylabel )
	if ylabel is not None:   ax.set_ylabel( ylabel, fontsize=fontsize_xylabel )
	if ylim is not None:     ax.set_ylim( ylim[0],ylim[1] )
	if grid:                 ax.grid()
	if legend:               ax.legend(fontsize=fontsize_legend)
	if filename is not None: plt.savefig(filename, bbox_inches='tight')
	plt.gcf().clear() 
	plt.close()





def plot_image_grid_now(handle_figure, image_tensor, filename=None, normalize=True):

	plt.figure(handle_figure.number)

	num_image = image_tensor.shape[0]
	nrows	  = np.sqrt(num_image)
	nrows	  = int(np.ceil(nrows))
	
	if normalize:  image_data = normalize_tensor(image_tensor, 0, 1)
	
	image_data = image_data.detach().cpu()
	image_grid = make_grid(image_data, nrow=nrows)
	image_grid = image_grid.permute(1, 2, 0)

	plt.imshow(image_grid, vmin=0, vmax=1)

	plt.pause(0.001)

	if filename is not None:
		handle_figure.savefig(filename)
		pass



def resize_examples(examples, size=64):
	transform = transforms.Compose([
	    transforms.ToPILImage(),
	    transforms.Resize(size=size,interpolation=transforms.InterpolationMode.NEAREST),
	    transforms.ToTensor()
	])
	return [transform(x) for x in examples]


def draw_image_grid(image_data, filename=None, nrows=None, upsize=True, normalize=None, scale_each=None, pad_value=None):  # pad_value: color of grid lines, 0 black, 1 white.

	image_data = image_data.detach().cpu()

	num_image, channel, height, width  = image_data.shape

	if normalize is None:
		if channel == 1: normalize = True
		else:            normalize = True   # color images

	if scale_each is None:
		if channel == 1: scale_each = False
		else:            scale_each = True   # color images

	if pad_value is None:
		if channel == 1: pad_value = 1  # white grid line
		else:            pad_value = 0  # black grid line

	if nrows is None or nrows * nrows > num_image:
		nrows = int(np.ceil( np.sqrt(num_image) ))  

	if normalize:  image_data = normalize_tensor(image_data, 0, 1)

	#print(image_data.shape)

	if upsize and width <= 64:
		image_data = resize_examples(image_data, size=64)
	

	image_grid = make_grid(image_data, nrow=nrows, pad_value=pad_value, scale_each=scale_each)
	# https://pytorch.org/vision/stable/_modules/torchvision/utils.html#save_image
	ndarr = image_grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
	im = Image.fromarray(ndarr)
	im.save(filename)




def plot_curve_D_outputs(epoch,D_outputs_real,D_outputs_fake,FID,ylim=[-3,3],filename=None):

	# log        = np.load(file_input, allow_pickle=True)
	#
	# epoch          = log['epoch']
	# D_outputs_real = log['D_outputs_real']
	# D_outputs_fake = log['D_outputs_fake']
	# FID            = log['FID']
	#
	# plot_curve_D_outputs(epoch,D_outputs_real,D_outputs_fake,FID,filename='./test.png'):
	

	# configures
	plot_min_FID = True
	legend = True
	xlim   = None
	
	xlabel = 'epoch'
	ylabel = 'discriminator output, $D(x)$'
	colors = {'real':'tab:blue', 'fake':'tab:orange'}

	D_outputs_real = np.transpose( D_outputs_real ) # [epoch][9] -> [9][epoch]
	D_outputs_fake = np.transpose( D_outputs_fake )
	D_outputs      = {'real':D_outputs_real, 'fake':D_outputs_fake}
	# D_outputs[label][0]: 10%-th values
	#     :
	# D_outputs[label][4]: 50%-th values, median
	#     :
	# D_outputs[label][8]: 90%-th values

	if len(D_outputs_real) != 9:
		print('error: plot_D_outputs(), num of percentiles = {} should be 9 (10%,20%,...,90%)'.format( len(D_outputs_real) ))

	plt.cla()
	ax = plt.subplot()
	for label in ['real','fake']:
		color = colors[label]
		ys    = D_outputs[label]
		
		ax.fill_between(epoch,y1=ys[3],y2=ys[5],alpha=0.20,facecolor=color)
		ax.fill_between(epoch,y1=ys[2],y2=ys[6],alpha=0.15,facecolor=color)
		ax.fill_between(epoch,y1=ys[1],y2=ys[7],alpha=0.10,facecolor=color)
		ax.fill_between(epoch,y1=ys[0],y2=ys[8],alpha=0.05,facecolor=color)

		ax.plot(epoch,ys[0],alpha=0.30,color=color,linewidth=1)
		ax.plot(epoch,ys[1],alpha=0.30,color=color,linewidth=1)
		ax.plot(epoch,ys[2],alpha=0.30,color=color,linewidth=1)
		ax.plot(epoch,ys[3],alpha=0.30,color=color,linewidth=1)
		ax.plot(epoch,ys[4],color=color,linewidth=2,label=label)
		ax.plot(epoch,ys[5],alpha=0.30,color=color,linewidth=1)
		ax.plot(epoch,ys[6],alpha=0.30,color=color,linewidth=1)
		ax.plot(epoch,ys[7],alpha=0.30,color=color,linewidth=1)
		ax.plot(epoch,ys[8],alpha=0.30,color=color,linewidth=1)

	# y=0
	ax.axhline(y=0,color='black',alpha=0.50,linewidth=0.5,zorder=-99) 
	
	# epoch with the best FID
	best_epoch   = epoch[ np.argmin(FID) ]
	ax.axvline(x=best_epoch, linewidth=1, zorder=0, color='red', linestyle='--',label='Best FID')

	if xlabel is not None:   ax.set_xlabel( xlabel, fontsize=14 )
	if ylabel is not None:   ax.set_ylabel( ylabel, fontsize=14 )
	if xlim is not None:     ax.set_xlim( xlim[0],xlim[1] )
	if ylim is not None:     ax.set_ylim( ylim[0],ylim[1] )
	if legend:               ax.legend()
	if filename is not None: plt.savefig(filename, bbox_inches='tight')
	plt.gcf().clear() 
	plt.close()



if __name__=='__main__':
	
	#
	# Demonstration of plot_curve_errobar()
	#
	x  = np.arange(10)
	y1 = x**2
	e1 = np.random.randn(*y1.shape)
	y2 = 2*x + 1
	e2 = np.random.randn(*y2.shape)

	plot_curve_errorbar(y1, e1, labels='real', ylim=[0,100], filename='test_plot_curve_errorbar1.png')
	plot_curve_errorbar([y1,y2], [e1,e2], labels=['real','fake'], ylim=[0,100], filename='test_plot_curve_errorbar2.png')


	#
	# Demonstration of plot_batch()
	#
	batch = torch.rand((64,3,32,32)) # number of images, channel, height, and width of image

	# we plot 4 x 4 images
	#plot_batch(batch=batch, filename='./test_plot_batch.png')
	draw_image_grid(image_data=batch, filename='./test_plot_batch.png')


