# ------------------------------------------------------------------------------------------------
#
# DCGAN in PyTorch tutorial:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#
# initialize_weight() added.
#
# spectral normalization added (DCGAN)
#
# ------------------------------------------------------------------------------------------------


import torch
from torch import nn
import numpy as np



from torch.nn.utils import spectral_norm


'''
# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.normal_(m.weight, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		torch.nn.init.normal_(m.weight, 1.0, 0.02)
		torch.nn.init.zeros_(m.bias)
'''




class DCGAN2_Discriminator32_Estimate_Condition(nn.Module):
	def __init__(self, channel=3, dim_condition=[1], activation=False, bottleneck=512):
		super(DCGAN2_Discriminator32_Estimate_Condition, self).__init__()
		ndf = 32
		threshold_ReLU = 0.2		

		self.activation = activation
		self.dim_condition = dim_condition
	
		self.network = nn.Sequential(
			# input is (channe) x 32 x 32
			nn.Conv2d(channel, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# 
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# 
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# 
			#nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
			nn.Conv2d(ndf * 4, bottleneck, 4, 1, 0, bias=False),
			#nn.Sigmoid()
		)


		self.classifier = torch.nn.Sequential(   # modified
			nn.Linear(bottleneck,  4 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#
			nn.Linear(4 * ndf, 2 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#
			nn.Linear(2 * ndf, 1 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#			
			nn.Linear(1 * ndf, 1 + dim_condition,  bias=True),
			)
		
		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()

		#if activation: self.classifier.add_module('activation', nn.Sigmoid())


	def forward(self, x):
		x = self.network(x)
		x = nn.Flatten()(x)	
		y = self.classifier(x).squeeze(-1).squeeze(-1)

		# Split the output like [128,11] -> y=[128,1], c=[128,10] 
		# where y is the real/fake estimation and c is the estimated condition
		y,c = torch.split(y, [1,self.dim_condition], dim=1) 

		# The activation will be turn off when using 
		# BCE with logits function 
		if self.activation: 
			y = self.sigmoid( y )
			c = self.softmax( c )

		return y,c




class DCGAN2_Discriminator32_Classifier_Condition(nn.Module):
	def __init__(self, channel=3, dim_condition=[1], activation=False, bottleneck=512):  # modified
		super(DCGAN2_Discriminator32_Classifier_Condition, self).__init__()
		ndf = 32
		threshold_ReLU = 0.2		

		self.network = nn.Sequential(
			# input is (channe) x 32 x 32
			nn.Conv2d(channel, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# 16 x 16
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# 8 x 8
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# 4 x 4
			# 
			nn.Conv2d(ndf * 4, bottleneck, 4, 1, 0, bias=False),
			#nn.Sigmoid()
		)

		#bottleneck = ndf * 8 * 2 * 2
		self.classifier = torch.nn.Sequential(   # modified
			nn.Linear(bottleneck + dim_condition,  4 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#
			nn.Linear(4 * ndf, 2 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#
			nn.Linear(2 * ndf, 1 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#			
			nn.Linear(1 * ndf, 1,       bias=True),
			)

		if activation: self.classifier.add_module('activation', nn.Sigmoid())



	def forward(self, x, condition):
		x = self.network(x)
		x = nn.Flatten()(x)
		#print(x.shape, condition.shape)
		#exit(-1)
		x = torch.cat((x, condition), dim=1)
		y = self.classifier(x)

		return y.squeeze(-1).squeeze(-1)



class DCGAN2_Discriminator32(nn.Module):
	def __init__(self, channel=3, activation=False, bottleneck=512):
		super(DCGAN2_Discriminator32, self).__init__()
		ndf = 32
		threshold_ReLU = 0.2		
		self.network = nn.Sequential(
			# input is (channe) x 32 x 32
			nn.Conv2d(channel, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# 
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# 
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# 
			#nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
			nn.Conv2d(ndf * 4, bottleneck, 4, 1, 0, bias=False),
			#nn.Sigmoid()
		)


		self.classifier = torch.nn.Sequential(   # modified
			nn.Linear(bottleneck,  4 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#
			nn.Linear(4 * ndf, 2 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#
			nn.Linear(2 * ndf, 1 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#			
			nn.Linear(1 * ndf, 1,       bias=True),
			)

		if activation: self.network.add_module('activation', nn.Sigmoid() )

	def forward(self, x):
		x = self.network(x)
		x = nn.Flatten()(x)
		y = self.classifier(x)
		return y.squeeze(-1).squeeze(-1)





class DCGAN2_Generator32(nn.Module):
	def __init__(self, length_latent=100, channel=3):
		super(DCGAN2_Generator32, self).__init__()
		nz  = length_latent
		ngf = 32
		self.network = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(	 nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			#nn.ConvTranspose2d(ngf * 2,	 ngf, 4, 2, 1, bias=False),
			nn.ConvTranspose2d(ngf * 2,	 channel, 4, 2, 1, bias=False),
			#nn.BatchNorm2d(ngf),
			#nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			#nn.ConvTranspose2d(	ngf,	  channel, 4, 2, 1, bias=False),
			nn.Tanh()
			# state size. (channel) x 64 x 64
		)

	def initialize_weight(self):	
		#print('initialize model parameters')
		for m in self.network.modules():
			classname = m.__class__.__name__
			if classname.find('Conv') != -1:
				nn.init.normal_(m.weight, 0.0, 0.02)
			elif classname.find('BatchNorm') != -1:
				nn.init.normal_(m.weight, 1.0, 0.02)
				nn.init.zeros_(m.bias)
			'''
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
					pass
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)`
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
					pass
			'''

	def forward(self, x):
		y = self.network(x)
		return y












class DCGAN2_Generator(nn.Module):
	def __init__(self, length_latent=100, channel=3):
		super(DCGAN2_Generator, self).__init__()
		nz  = length_latent
		ngf = 64
		self.network = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(	 nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(ngf * 2,	 ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d(	ngf,	  channel, 4, 2, 1, bias=False),
			nn.Tanh()
			# state size. (channel) x 64 x 64
		)

	def initialize_weight(self):	
		#print('initialize model parameters')
		for m in self.network.modules():
			classname = m.__class__.__name__
			if classname.find('Conv') != -1:
				nn.init.normal_(m.weight, 0.0, 0.02)
			elif classname.find('BatchNorm') != -1:
				nn.init.normal_(m.weight, 1.0, 0.02)
				nn.init.zeros_(m.bias)
			'''
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
					pass
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)`
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
					pass
			'''

	def forward(self, x):
		y = self.network(x)
		return y




class DCGAN2_Discriminator(nn.Module):
	def __init__(self, channel=3, activation=False, bottleneck=512):
		super(DCGAN2_Discriminator, self).__init__()
		ndf = 64
		threshold_ReLU = 0.2		
		self.network = nn.Sequential(
			# input is (channe) x 64 x 64
			nn.Conv2d(channel, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			#nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Conv2d(ndf * 8, bottleneck, 4, 1, 0, bias=False),
			#nn.Sigmoid()
		)


		self.classifier = torch.nn.Sequential(  
			nn.Linear(bottleneck,  4 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#
			nn.Linear(4 * ndf, 2 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#
			nn.Linear(2 * ndf, 1 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#			
			nn.Linear(1 * ndf, 1,       bias=True),
			)

		if activation: self.network.add_module('activation', nn.Sigmoid() )

	def forward(self, x):
		x = self.network(x)
		x = nn.Flatten()(x)
		y = self.classifier(x)
		return y.squeeze(-1).squeeze(-1)
		#return y.view(-1, 1).squeeze(1)




class DCGAN2_Discriminator_Classifier_Condition(nn.Module):
	def __init__(self, channel=3, dim_condition=[1], activation=False, bottleneck=512):
		super(DCGAN2_Discriminator_Classifier_Condition, self).__init__()
		ndf = 64
		threshold_ReLU = 0.2		

		self.network = nn.Sequential(
			# input is (channe) x 64 x 64
			nn.Conv2d(channel, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(ndf * 8, bottleneck, 4, 1, 0, bias=False),
			#nn.Sigmoid()
		)


		self.classifier = torch.nn.Sequential(   
			nn.Linear(bottleneck + dim_condition,  4 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#
			nn.Linear(4 * ndf, 2 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#
			nn.Linear(2 * ndf, 1 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#			
			nn.Linear(1 * ndf, 1,       bias=True),
			)

		if activation: self.classifier.add_module('activation', nn.Sigmoid())



	def forward(self, x, condition):
		x = self.network(x)
		x = nn.Flatten()(x)
		x = torch.cat((x, condition), dim=1)		
		y = self.classifier(x)
		return y.squeeze(-1).squeeze(-1)
		#return y.view(-1, 1).squeeze(1)



# https://arxiv.org/abs/1605.05396
# Generative Adversarial Text to Image Synthesis
class DCGAN2_Discriminator_Conv2_Condition(nn.Module):
	def __init__(self, length_latent=100, channel=3, dim_condition=[1], activation=False, bottleneck=512):
		super(DCGAN2_Discriminator_Conv2_Condition, self).__init__()
		ndf = 64

		# covert the condition code into a data with single channel
		self.mapping = nn.Sequential(
			nn.Linear(dim_condition,  8*8, bias=False)
			)

		self.network1 = nn.Sequential(
			# input is (channe) x 64 x 64
			nn.Conv2d(channel, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.network2 = nn.Sequential(
			# state size. (ndf*4) x 8 x 8
			#nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.Conv2d(ndf * 4 + 1, ndf * 8, 4, 2, 1, bias=False),  # +1: condition channel
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			#nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Conv2d(ndf * 8, bottleneck, 4, 1, 0, bias=False),
			#nn.Sigmoid()
		)



		self.classifier = torch.nn.Sequential(   
			nn.Linear(bottleneck,  4 * ndf, bias=True),
			nn.LeakyReLU(0.2, inplace=True),
			#
			nn.Linear(4 * ndf, 2 * ndf, bias=True),
			nn.LeakyReLU(0.2, inplace=True),
			#
			nn.Linear(2 * ndf, 1 * ndf, bias=True),
			nn.LeakyReLU(0.2, inplace=True),
			#			
			nn.Linear(1 * ndf, 1,       bias=True),
			)



		if activation: self.classifier.add_module('activation', nn.Sigmoid())

	def forward(self, x, condition):


		x = self.network1(x)

		# covert the condition vector to a single channel data
		c = self.mapping(condition)
		c = c.view(x.shape[0], 1, 8, 8)
		# concatenate the condition data to the image feature
		x = torch.cat([x,c],dim=1)

		x = self.network2(x)

		#return x.squeeze(-1).squeeze(-1)
		return x.view(-1, 1).squeeze(1)






'''
class DCGAN2_Discriminator_Classifier_Condition(nn.Module):
	def __init__(self, channel=3, dim_condition=[1], activation=False, bottleneck=512):
		super(DCGAN2_Discriminator_Classifier_Condition, self).__init__()
		ndf = 64
		threshold_ReLU = 0.2		

		self.network = nn.Sequential(
			# input is (channe) x 64 x 64
			nn.Conv2d(channel, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			#nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			#nn.Conv2d(ndf * 8, length_latent, 4, 1, 0, bias=False),
			nn.Conv2d(ndf * 8, bottleneck, 4, 1, 0, bias=False),
			#nn.Sigmoid()
		)


		self.classifier = torch.nn.Sequential(   
			nn.Linear(bottleneck + dim_condition,  4 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#
			nn.Linear(4 * ndf, 2 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#
			nn.Linear(2 * ndf, 1 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#			
			nn.Linear(1 * ndf, 1,       bias=True),
			)

		if activation: self.classifier.add_module('activation', nn.Sigmoid())



	def forward(self, x, condition):
		x = self.network(x)
		x = nn.Flatten()(x)
		x = torch.cat((x, condition), dim=1)		
		y = self.classifier(x)
		return y.squeeze(-1).squeeze(-1)
		#return y.view(-1, 1).squeeze(1)

'''

'''
class DCGAN2_Discriminator_Input_Condition(nn.Module):
	def __init__(self, length_latent=100, channel=3, dim_condition=[1], activation=False):
		super(DCGAN2_Discriminator_Input_Condition, self).__init__()
		ndf = 64

		# covert the condition code into a data with single channel
		self.mapping = nn.Sequential(
			nn.Linear(dim_condition,  1*64*64, bias=False)
			)

		self.network = nn.Sequential(
			# input is (channe) x 64 x 64
			nn.Conv2d(channel+1, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			#nn.Sigmoid()
		)

		if activation: self.classifier.add_module('activation', nn.Sigmoid())

	def forward(self, x, condition):

		# covert the condition vector to a single channel image 
		c = self.mapping(condition)
		c = c.view(x.shape[0], 1, x.shape[2], x.shape[3])

		# concatenate the condition image with the input image
		x = torch.cat([x,c],dim=1)

		y = self.network(x)

		return y.squeeze(-1).squeeze(-1)
		#return y.view(-1, 1).squeeze(1)



class DCGAN2_Discriminator_Conv2_Condition(nn.Module):
	def __init__(self, length_latent=100, channel=3, dim_condition=[1], activation=False):
		super(DCGAN2_Discriminator_Conv2_Condition, self).__init__()
		ndf = 64

		# covert the condition code into a data with single channel
		self.mapping = nn.Sequential(
			nn.Linear(dim_condition,  8*8, bias=False)
			)

		self.network1 = nn.Sequential(
			# input is (channe) x 64 x 64
			nn.Conv2d(channel, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
		)


		self.network2 = nn.Sequential(
			# state size. (ndf*4) x 8 x 8
			#nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.Conv2d(ndf * 4 + 1, ndf * 8, 4, 2, 1, bias=False),  # +1: condition channel
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			#nn.Sigmoid()
		)


		if activation: self.classifier.add_module('activation', nn.Sigmoid())

	def forward(self, x, condition):



		x = self.network1(x)
		#print(x.shape)

		# covert the condition vector to a single channel data
		c = self.mapping(condition)
		c = c.view(x.shape[0], 1, 8, 8)
		# concatenate the condition data to the image feature
		x = torch.cat([x,c],dim=1)

		x = self.network2(x)

		#return x.squeeze(-1).squeeze(-1)
		return x.view(-1, 1).squeeze(1)






class DCGAN2_Discriminator_Conv1_Condition(nn.Module):
	def __init__(self, length_latent=100, channel=3, dim_condition=[1], activation=False):
		super(DCGAN2_Discriminator_Conv1_Condition, self).__init__()
		ndf = 64

		# covert the condition code into a data with single channel
		self.mapping = nn.Sequential(
			nn.Linear(dim_condition,  4*4, bias=False)
			)

		self.network1 = nn.Sequential(
			# input is (channe) x 64 x 64
			nn.Conv2d(channel, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			#nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # +1: condition channel
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
		)


		self.network2 = nn.Sequential(
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(ndf * 8 + 1, 1, 4, 1, 0, bias=False),
			#nn.Sigmoid()
		)


		if activation: self.classifier.add_module('activation', nn.Sigmoid())

	def forward(self, x, condition):



		x = self.network1(x)
		#print(x.shape)

		# covert the condition vector to a single channel data
		c = self.mapping(condition)
		c = c.view(x.shape[0], 1, 4, 4)
		# concatenate the condition data to the image feature
		x = torch.cat([x,c],dim=1)

		x = self.network2(x)

		#return x.squeeze(-1).squeeze(-1)
		return x.view(-1, 1).squeeze(1)
'''





class DCGAN2_Discriminator_Estimate_Condition(nn.Module):
	def __init__(self, channel=3, dim_condition=[1], activation=False, bottleneck=512):
		super(DCGAN2_Discriminator_Estimate_Condition, self).__init__()
		ndf = 64
		threshold_ReLU = 0.2		

		self.activation = activation
		self.dim_condition = dim_condition

		self.network = nn.Sequential(
			# input is (channe) x 64 x 64
			nn.Conv2d(channel, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(ndf * 8, bottleneck, 4, 1, 0, bias=False),
			#nn.Sigmoid()
		)


		self.classifier = torch.nn.Sequential(   
			nn.Linear(bottleneck,  2 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#
			nn.Linear(2 * ndf, 1 * ndf, bias=True),
			nn.LeakyReLU(threshold_ReLU, inplace=True),
			#			
			nn.Linear(1 * ndf, 1 + dim_condition,       bias=True),
			)

		
		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()

		#if activation: self.classifier.add_module('activation', nn.Sigmoid())


	def forward(self, x):
		x = self.network(x)
		x = nn.Flatten()(x)	
		y = self.classifier(x).squeeze(-1).squeeze(-1)

		# Split the output like [128,11] -> y=[128,1], c=[128,10] 
		# where y is the real/fake estimation and c is the estimated condition
		y,c = torch.split(y, [1,self.dim_condition], dim=1) 

		# The activation will be turn off when using 
		# BCE with logits function 
		if self.activation: 
			y = self.sigmoid( y )
			c = self.softmax( c )

		return y,c




if __name__=='__main__':


	net = DCGAN2_Discriminator(channel=3, activation=False,bottleneck=512)
	print(net) 

	B = 128
	nc = 3
	h = 64
	w = 64
	x = torch.randn(B,nc,h,w)
	y = net(x)
	print( y.shape )

	
	dim_condition=10
	net = DCGAN2_Discriminator_Classifier_Condition(channel=3, dim_condition=dim_condition, activation=False,bottleneck=512)
	print(net) 

	B = 128
	nc = 3
	h = 64
	w = 64
	x = torch.randn(B,nc,h,w)
	condition = torch.zeros(B,dim_condition)
	y = net(x,condition)
	print( y.shape )



	# ----------------------------------------------------------------------
	# Debug mode using BCE loss 
	# ----------------------------------------------------------------------
	dim_condition=5
	net = DCGAN2_Discriminator_Estimate_Condition(channel=3, dim_condition=dim_condition, activation=True,bottleneck=512) # activation will be False 
	print(net) 
	#
	B = 10
	nc = 3
	h = 64
	w = 64
	x = torch.randn(B,nc,h,w)
	#
	y,c = net(x)
	print( y.shape )
	print( c.shape )
	#
	# a true condition
	condition = torch.zeros(B,dim_condition)
	condition[:,0] = 1
	print( condition )
	#
	F = torch.nn.functional
	#
	#
	loss_class = torch.sum( F.binary_cross_entropy(input=c, target=condition) )
	print( loss_class.item() )

	c = (condition + c) / 2.
	loss_class = torch.sum( F.binary_cross_entropy(input=c, target=condition) )
	print( loss_class.item() )


	# ----------------------------------------------------------------------
	# Practical set-up
	# ----------------------------------------------------------------------
	dim_condition=5
	net = DCGAN2_Discriminator_Estimate_Condition(channel=3, dim_condition=dim_condition, activation=False,bottleneck=512)
	print(net) 
	#
	B = 10
	nc = 3
	h = 64
	w = 64
	x = torch.randn(B,nc,h,w)
	#
	y,c = net(x)
	print( y.shape )
	print( c.shape )
	#
	# a true condition
	condition = torch.zeros(B,dim_condition)
	condition[:,0] = 1
	print( condition )
	#
	F = torch.nn.functional
	loss_class = torch.sum( F.binary_cross_entropy_with_logits(input=c, target=condition) )
	print( loss_class.item() )



