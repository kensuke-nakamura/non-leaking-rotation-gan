
'''
#
# Example of use
#

import torch
import numpy as np
import ada_transform as ada

# Perform Ada Heuristic to determine the global P
I = 1000
for i in range(I):
	rt = np.random.rand(64)  # Sign of discriminator output with training data # 64: batch-size
	ada.update_global_P(rt)  # the global P will be updated using 4 batches with some constant step 
	print( '{:.8f}'.format( ada.global_P) )

# Test data transform with P=1 
ada.global_P=1
data             = torch.rand([128,3,64,64])
data_transformed = ada.Theta(data, ada.global_P) 

print(data[0])
print(data_transformed[0])
'''


import torch
from   torch.autograd import grad
from   torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
import numpy as np



from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torchvision.transforms as transforms
import numpy as np
import sys
from PIL import Image

import matplotlib
matplotlib.use('Agg')



# adaptive probability of the nvidia algorithm (global parameter)
global_P = 0.0
#
# Excute ADA heuristic (Lines 312-315 in training_loop.py of the official styleGAN2-Ada
#
value_global_P=[]
rt_list=[]
def update_global_P(rt, ada_interval=256, ada_target=0.6, speed=0.0000128):
	# rt:         : discriminator output with real data 
	# ada_interval: num. of real examples to update P
	# ada_target  : the target value of rt
	# speed       : an approximation of "(batch_size * ada_interval) / (ada_kimg * 1000)" in the official code
	global global_P
	global rt_list

	global value_global_P # monitoring

	rt_list.extend(rt)

	if len(rt_list) > ada_interval:
 
		# compute the adjust step
		ada_rt     = np.sign( np.array(rt_list) ).mean()
		step_size  = speed #ada_interval * speed 
		adjust     = np.sign( ada_rt - ada_target ) * step_size	
		# update P
		global_P = np.clip(global_P + adjust, 0, 1) # global variable
		# initialize
		rt_list = []

		#
		value_global_P.append( global_P )






# stylegan/torch_utils/misc.py
#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.
_constant_cache = dict()
def constant(value, shape=None, dtype=None, device=None, memory_format=None):
	value = np.asarray(value)
	if shape is not None:
		shape = tuple(shape)
	if dtype is None:
		dtype = torch.get_default_dtype()
	if device is None:
		device = torch.device('cpu')
	if memory_format is None:
		memory_format = torch.contiguous_format

	key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
	tensor = _constant_cache.get(key, None)
	if tensor is None:
		tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
		if shape is not None:
			tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
		tensor = tensor.contiguous(memory_format=memory_format)
		_constant_cache[key] = tensor
	return tensor
# stylegan/training/augment.py
def matrix(*rows, device=None):
	assert all(len(row) == len(rows[0]) for row in rows)
	elems = [x for row in rows for x in row]
	ref = [x for x in elems if isinstance(x, torch.Tensor)]
	if len(ref) == 0:
		#return misc.constant(np.asarray(rows), device=device)
		return constant(np.asarray(rows), device=device)
	assert device is None or device == ref[0].device
	#elems = [x if isinstance(x, torch.Tensor) else misc.constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
	elems = [x if isinstance(x, torch.Tensor) else constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
	return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))
def rotate3d(v, theta, **kwargs):
	vx = v[..., 0]; vy = v[..., 1]; vz = v[..., 2]
	s = torch.sin(theta); c = torch.cos(theta); cc = 1 - c
	return matrix(
		[vx*vx*cc+c,	vx*vy*cc-vz*s, vx*vz*cc+vy*s, 0],
		[vy*vx*cc+vz*s, vy*vy*cc+c,	vy*vz*cc-vx*s, 0],
		[vz*vx*cc-vy*s, vz*vy*cc+vx*s, vz*vz*cc+c,	0],
		[0,			 0,			 0,			 1],
		**kwargs)
#
#
#
def Phi(data, P=None):
		
	global global_P

	if P is None: P = global_P

	#P = 1 # leaking test

	# Given global_P_nvidia_ada (P), apply rotation-4 and hue-4 with P
	# rot-4
	option_rotate = torch.randint(low=0,high=4,size=(len(data),)) # low is inclusive and high is *exclusive*
	option_rotate[np.random.random(size=option_rotate.shape) > P] = 0 # use the orignal data with probability (transform[name])
	transformed = [torch.rot90(data[i].unsqueeze(0), option_rotate[i], [2,3]) for i in range(len(data))]
	data        = torch.cat(transformed, dim=0)
	batch_size, num_channels, height, width = data.shape
	#
	# hue-4
	v     = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=data.device) # Luma axis.
	theta = np.random.choice([-0.5,0,0.5,1], size=len(data))  # -0.5:purple, 0:original, 0.5:green, 1.0:blue faces (1.5: purple)
	theta[np.random.random(size=theta.shape) > P] = 0.0 # use the orignal data with probability (transform[name])
	theta = torch.from_numpy(theta).type(torch.FloatTensor).to(data.device)
	theta = theta * np.pi
	C     = rotate3d(v, theta)
	#
	data = data.reshape([batch_size, num_channels, height * width])
	data = C[:, :3, :3] @ data + C[:, :3, 3:]
	data = data.reshape([batch_size, num_channels, height, width])

	return data







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


def plot_global_P(filename):
	plot_curve_errorbar([np.array(value_global_P)], colors=['blue'], errs=None, labels=['Ada P'],  filename=filename)



'''
if __name__ == '__main__':

	I = 100
	for i in range(I):
		rt = np.random.rand(64) 
		update_global_P(rt)
		print( '{:.8f}'.format(global_P) )
'''
	
