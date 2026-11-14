

import torch
from   torch.autograd import grad
from   torch.autograd import Variable
import torch.nn as nn
#import torchvision
#import torchvision.datasets as dsets
#import torchvision.transforms as transforms
#import utils
import numpy as np


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
def Phi(data):
	# rotation w/o original (low=0)
	option_rotate = torch.randint(low=1,high=4,size=(len(data),)) # low is inclusive and high is *exclusive*
	#option_rotate[np.random.random(size=option_rotate.shape) > transform[name].numpy()] = 0 # use the orignal data with probability (transform[name])
	transformed = [torch.rot90(data[i].unsqueeze(0), option_rotate[i], [2,3]) for i in range(len(data))]
	data        = torch.cat(transformed, dim=0)

	# hue
	batch_size, num_channels, height, width = data.shape
	#
	v     = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=data.device) # Luma axis.
	theta = np.random.choice([-0.5,0.5,1], size=len(data))  # -0.5:purple, 0:original, 0.5:green, 1.0:blue faces (1.5: purple)
	#theta[np.random.random(size=theta.shape) > transform[name].numpy()] = 0.0 # use the orignal data with probability (transform[name])
	theta = torch.from_numpy(theta).type(torch.FloatTensor).to(data.device)
	theta = theta * np.pi
	C     = rotate3d(v, theta)
	#
	data = data.reshape([batch_size, num_channels, height * width])
	data = C[:, :3, :3] @ data + C[:, :3, 3:]
	data = data.reshape([batch_size, num_channels, height, width])
	return data


if __name__ == '__main__':

	# Example
	
	# given batch of real data (or fakes)
	batch = torch.FloatTensor(size=(128,3,64,64)) # batch size, num. channel, image height, image width

	# apply our transform
	transformed = Phi(batch)

	# enc
