

'''
# How to install
pip3 install prdc
pip3 install install pytorch-fid-wrapper; pip3 install git+https://github.com/evenmn/pytorch-sfid

# Toy example
python3 precision_recall.py
'''

import torch
import numpy
import sys 
import os
import matplotlib.pyplot as plt
import glob
import time
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torchvision

from prdc import compute_prdc

# ======================================================================
# cuda
# ======================================================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

def data2batch_to_compute_prec_rec(batch, batch_size=1):
	_, nc, height, width = batch.shape
	batch = batch.reshape(-1,batch_size,nc,height,width)
	if height != 224 or width != 224:  # resize
		resized_batch=[]
		for i in range(len(batch)):
			data    = batch[i].squeeze(0)
			resized = torch.nn.functional.interpolate(input=data,size=(224, 224))
			resized_batch.append( resized.unsqueeze(0).detach().cpu() )
		batch = torch.cat( resized_batch, dim=0 )
	if nc < 3:  batch = batch.expand(-1,-1,3,-1,-1)  # gray to rgb
	return batch



# batch_data_real:   real data with [N,c,H,W]
# batch_data_fake:   fake data with [N,c,H,W]
# is_2d_image:       [True] for images, and [False] for 1D data like two-moons
# feature_extractor: Network to computes features of images
# k:                 The number of points to compute manifold, The original paper recommends k=[5] while recent papers use k=[3].
def compute_precision_and_recall(batch_data_real, batch_data_fake, is_2d_image=True, feature_extractor=None, k_set=[3], batch_size=5):

	if len(batch_data_real) != len(batch_data_fake):
		print('warning: real data size {} is not equal to fake data size {}'.format(len(batch_data_real), len(batch_data_fake)))
		print('         precision and recall assumes the same number of reals and fakes')

	# ------------------------------------------------------------------------------------------------------
	# Step1: Given real, fake images, with a pretrained network, we compute features of the data
	# https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/5ad4629b07f3f3a51184c39d3dbe9085a60e264c/improved_precision_recall.py
	# ------------------------------------------------------------------------------------------------------
	if is_2d_image:
		if feature_extractor == None:
			print('loading vgg16 for improved precision and recall...', end='', flush=True)
			#feature_extractor = models.vgg16(pretrained=True).eval().to(device)
			feature_extractor = models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).eval().to(device)

		print('image data is converted to (3,224,224) to compute precision and recall')
		resized_batch_real = data2batch_to_compute_prec_rec(batch_data_real, batch_size=batch_size)	
		resized_batch_fake = data2batch_to_compute_prec_rec(batch_data_fake, batch_size=batch_size)

		print('converting resized data to feature...')
		# real
		real_features = []
		for i in range(len(resized_batch_real)):
			batch    = resized_batch_real[i].squeeze(0)
			features = feature_extractor.features(batch.to(device)).view(-1, 7 * 7 * 512) # VGG11
			features = feature_extractor.classifier[:4](features)
			real_features.append( features.detach().cpu() )
		real_features = torch.cat( real_features, dim=0 ).numpy()	
		# fake
		fake_features = []
		for i in range(len(resized_batch_fake)):
			batch    = resized_batch_fake[i].squeeze(0)
			features = feature_extractor.features(batch.to(device)).view(-1, 7 * 7 * 512) # VGG11
			features = feature_extractor.classifier[:4](features)
			fake_features.append( features.detach().cpu() )
		fake_features = torch.cat( fake_features, dim=0 ).numpy()
		print('done')
	else:
		# 1d data
		real_features = batch_data_real.reshape(batch_data_real.shape[0],-1).detach().cpu().numpy()
		fake_features = batch_data_fake.reshape(batch_data_fake.shape[0],-1).detach().cpu().numpy()


	# ------------------------------------------------------------------------------------------------------
	# Step2: Given features, we compute precision, recall, and others based on 
	# Reliable Fidelity and Diversity Metrics for Generative Models (ICML 2020)
	# https://github.com/clovaai/generative-evaluation-prdc
	# ------------------------------------------------------------------------------------------------------
	'''
	evaluation_results = ""
	for nearest_k in k_set:
		metrics = compute_prdc(real_features=real_features, fake_features=fake_features, nearest_k=nearest_k)	
		result  = 'precision_recall_density_coverage_k{}\t'.format(nearest_k)
		result  = result + '{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t'.format(metrics['precision'], metrics['recall'], metrics['density'], metrics['coverage'])
		evaluation_results += result
	return evaluation_results
	'''
	evaluation_results=[]
	for nearest_k in k_set:
		metrics = compute_prdc(real_features=real_features, fake_features=fake_features, nearest_k=nearest_k)	
		evaluation_results.append( metrics )

	# evaluation_results[0]['precision']
	# evaluation_results[0]['recall']
	# evaluation_results[0]['density']
	# evaluation_results[0]['coverage']
	return evaluation_results 
	

if __name__ == '__main__':

	# ---------------------------------------------------------------------------------------------------------------
	# demonstration: cifar-10 like images
	batch_data_real =  torch.randn([10000,3,28,28])
	batch_data_fake =  torch.randn([10000,3,28,28])

	evaluation_results = compute_precision_and_recall(batch_data_real, batch_data_fake, k_set=[3], is_2d_image=True) 

	print('recall',    evaluation_results[0]['recall'])       # [0]: result using 1st(0-th) k in k_set
	print('precision', evaluation_results[0]['precision']) 
	print('density',   evaluation_results[0]['density']) 
	print('coverage',  evaluation_results[0]['coverage']) 


	# ---------------------------------------------------------------------------------------------------------------
	# demonstration: synthetic data
	batch_data_real =  torch.randn([10000,3,1,1])
	batch_data_fake =  torch.randn([10000,3,1,1])

	evaluation_results = compute_precision_and_recall(batch_data_real, batch_data_fake, k_set=[3], is_2d_image=False)

	print('recall',    evaluation_results[0]['recall'])       # [0]: result using 1st(0-th) k in k_set
	print('precision', evaluation_results[0]['precision']) 
	print('density',   evaluation_results[0]['density']) 
	print('coverage',  evaluation_results[0]['coverage']) 

