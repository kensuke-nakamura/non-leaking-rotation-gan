# ------------------------------------------------------------------------------------------------
#
# Naive GAN
#
# ------------------------------------------------------------------------------------------------


import torch
from torch import nn
import numpy as np



class GAN_Generator(nn.Module):
	def __init__(self, length_latent=10, image_shape=(1,8,8), activation=True):
		super(GAN_Generator, self).__init__()

		self.image_shape = image_shape


		def block(in_feat, out_feat, normalize=True):
			layers = [nn.Linear(in_feat, out_feat)]
			if normalize:
				layers.append(nn.BatchNorm1d(out_feat, 0.8))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		'''
		dim_feature = int(np.prod(image_shape) / 16)
		self.model = nn.Sequential(
			*block(   length_latent,   dim_feature, normalize=False),
			*block(     dim_feature, 2*dim_feature),
			*block(   2*dim_feature, 4*dim_feature),
			*block(   4*dim_feature, 8*dim_feature),
			nn.Linear(8*dim_feature, int(np.prod(self.image_shape))),
		)
		'''
		self.model = nn.Sequential(
			*block(   length_latent,   128, normalize=False),
			*block(     128, 256),
			*block(   256, 512),
			*block(   512, 1024),
			nn.Linear(1024, int(np.prod(self.image_shape))),
			#nn.Tanh()
		)
		

		if activation: self.model.add_module('activation', nn.Tanh() )

	def forward(self, z):
		z = nn.Flatten()(z)
		y = self.model(z)
		y = y.view(y.shape[0], *(self.image_shape))
		return y


class GAN_Discriminator(nn.Module):
	def __init__(self, image_shape = (1,8,8), activation=False):
		super(GAN_Discriminator, self).__init__()
		
		'''
		dim_feature = int(np.prod(image_shape) / 4)
		self.model = nn.Sequential(
			nn.Linear(int(np.prod(image_shape)), 2*dim_feature),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(2*dim_feature, dim_feature),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(dim_feature, 1),
		)
		'''
		self.model = nn.Sequential(
			nn.Linear(int(np.prod(image_shape)), 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			#nn.Sigmoid(),
		)

		if activation: self.model.add_module('activation', nn.Sigmoid() )

	def forward(self, x):
		x = nn.Flatten()(x)
		y = self.model(x)
		return y.squeeze(-1)



class GAN_Discriminator_Condition(nn.Module):
	def __init__(self, dim_condition=0, image_shape = (1,8,8), activation=False):
		super(GAN_Discriminator_Condition, self).__init__()

		'''
		dim_feature = int(np.prod(image_shape) / 4)
		self.model = nn.Sequential(
			nn.Linear(int(np.prod(image_shape)) + dim_condition, 2*dim_feature),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(2*dim_feature, dim_feature),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(dim_feature, 1),
		)
		'''

		self.model = nn.Sequential(
			nn.Linear(int(np.prod(image_shape)) + dim_condition, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			#nn.Sigmoid(),
		)

		if activation: self.model.add_module('activation', nn.Sigmoid() )

	def forward(self, x, condition=None):
		x = nn.Flatten()(x)
		
		if condition is not None: 
			x = torch.cat((x, condition), dim=1)

		y = self.model(x)
		return y.squeeze(-1)



if __name__=='__main__':

	image_h  = 8
	image_w  = 8
	nchannel = 1
	batch_size    = 128
	length_latent = 10

	image_shape = (nchannel, image_h, image_w)

	generator     = GAN_Generator(length_latent=length_latent, image_shape=image_shape)
	discriminator = GAN_Discriminator(image_shape=image_shape)

	latent     = torch.randn(batch_size, length_latent, 1, 1)
	fake_batch = generator(latent)
	prediction = discriminator(fake_batch)

	print( latent.shape )
	print( fake_batch.shape )
	print( prediction.shape )


	dim_condition = 4
	condition     = torch.zeros( batch_size, dim_condition )
	discriminator = GAN_Discriminator_Condition(dim_condition=dim_condition, image_shape=image_shape)
	prediction = discriminator(fake_batch, condition)
	print( prediction.shape )

	


	dim_condition = 0
	condition     = torch.zeros( batch_size, dim_condition )
	discriminator = GAN_Discriminator_Condition(dim_condition=dim_condition, image_shape=image_shape)
	prediction = discriminator(fake_batch)
	print( prediction.shape )





