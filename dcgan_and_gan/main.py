# ------------------------------------------------------------------------------------------
#
# (1) simple example
# python3 main.py ./option/default/ini
#
#
# (2) experiment
# # display options
# python3 python3 edit_ini_file.py ./option/default/ini
#
# # edit options and run 
# cuda=0; ini=`python3 edit_ini_file.py ./option/default/ini --Dataset.fake__length_latent 100 `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &
# cuda=1; ini=`python3 edit_ini_file.py ./option/default/ini --Dataset.fake__length_latent 10  `; CUDA_VISIBLE_DEVICES=$cuda python3 main.py $ini &		
# wait
#
# ------------------------------------------------------------------------------------------


import os
import fcntl

root_dataset         = os.path.join(os.environ.get('DCGAN_DATASET_ROOT', '/hdd1/dataset'), '')
dir_temporal_real    = os.path.join(os.environ.get('DCGAN_TEMPORAL_REAL_DIR', '/hdd1/dataset/temporal_data'), '')


use_temporal_real    = True   # save the on_memory_dataset_real in a local folder: (True)
compute_FID          = True   # 
is_2d_image          = True   # automatically False when using 1d datasets

# vasualization (optional)
show_fake_process    = True   # optional
show_real_images     = True   # example of (transformed) real images
num_images_in_grid   = 25     # grid of fake images
num_image_per_trans  = 5      # random transform is generated every {5} images when using {show_real_images}, 
			      # if equal to num_images_in_grid (25), different (random) transform will be applied to each data


import torch
from torch import nn
from torchvision.datasets import * # Training dataset
from torchvision import models, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.autograd as autograd
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import random
import string
from torch.nn.parameter import Parameter
from torchvision.utils import save_image as save_image
from torch.utils.data.sampler import SubsetRandomSampler as SubsetRandomSampler
from torch.utils.data.sampler import RandomSampler	 as RandomSampler

import configparser
import sys
import os
import datetime 
from tqdm import tqdm
import copy
import socket
import time

time_elapsed = time.time()

dir_utility = '../../utility/'
sys.path.append(dir_utility)


from utility_visualize.visualize import *
from scheduler_learning_rate     import *

# will be moved to utitily
from utility_evaluation.fidis import fid_score as fid 
from utility_evaluation.fidis import inception_score as inception_score
from utility_network          import DCGAN2_Generator,   DCGAN2_Discriminator,   DCGAN2_Discriminator_Classifier_Condition
from utility_network          import DCGAN2_Discriminator_Estimate_Condition
from utility_network          import DCGAN2_Generator32, DCGAN2_Discriminator32, DCGAN2_Discriminator32_Classifier_Condition, DCGAN2_Discriminator_Conv2_Condition
from utility_network          import DCGAN2_Discriminator32_Estimate_Condition
from utility_network          import GAN_Generator, GAN_Discriminator, GAN_Discriminator_Condition
from option                   import abstract_of_option

# ======================================================================
# directory
# ======================================================================

# output directory
dir_result         = './result/'
path_figure        = dir_result + 'figure/'
path_real          = dir_result + 'figure/real/'
path_log           = dir_result + 'log/'
path_plot          = dir_result + 'plot/'
path_model         = dir_result + 'model/'
path_option        = dir_result + 'option/'
path_hadamard      = dir_result + 'hadamard/'

os.makedirs(dir_result,        exist_ok=True)
os.makedirs(path_figure,       exist_ok=True)
os.makedirs(path_real,         exist_ok=True)
os.makedirs(path_log,          exist_ok=True)
os.makedirs(path_plot,         exist_ok=True)
os.makedirs(path_model,        exist_ok=True)
os.makedirs(path_option,       exist_ok=True)
os.makedirs(path_hadamard,     exist_ok=True)
os.makedirs(dir_temporal_real, exist_ok=True)


# ======================================================================
# usage:
#   python3 main.py OPTION_FILE CUDA_DEVICE_NUMBER
#   python3 main.py option_file.ini 
# ======================================================================
if len(sys.argv) == 1:
	file_option         = './option/default.ini'
elif len(sys.argv) == 2:
	file_option         = sys.argv[1]
elif len(sys.argv) == 3:
	print('The argument of CUDA_DEVICE_NUMBER is no longer supported.')
	print('Type like:')
	print('CUDA_VISIBLE_DEVICES=0 python3 main.py ./option/default')
	exit(-1)
else:
	pass


# ======================================================================
# options
# ======================================================================

# Given file_option, read options
if not os.path.exists(file_option):
	print('The option file does not exist:', file_option)
	exit(-1)

from option import read_config 

config_dataset_real,             \
config_dataset_fake,             \
config_transform,                \
config_network,                  \
config_optimize,                 \
config_optimize_generator,       \
config_optimize_discriminator,   \
config                           = read_config(file_option)


# ======================================================================
# cuda
# ======================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device :', device)


# ======================================================================
# filename
# ======================================================================

hostname    = socket.gethostname()
option_name = os.path.splitext( os.path.basename(file_option) )[0]

now         = datetime.datetime.now()
date_stamp  = now.strftime('%Y_%m_%d') 
time_stamp  = now.strftime('%H_%M_%S') 

# identical filename like:  MNIST,2021_10_19,15_00_28
#file_id     = config_dataset_real['name'] + ',' + date_stamp + ',' + time_stamp
#file_id     = date_stamp + ',' + time_stamp + ',' + option_name
file_id     = option_name


print(file_id)
print(abstract_of_option(config))

file_figure_loss                     = path_figure + file_id + ',loss.png'
file_figure_prediction               = path_figure + file_id + ',prediction.png'
file_image_real                      = path_real   + file_id + ',real.png'
file_image_transform                 = path_real   + file_id + ',real_transform.png'
file_image_train                     = path_figure + file_id + ',train.png'
file_image_test                      = path_figure + file_id + ',test.png'
file_image_best                      = path_figure + file_id + ',test,best.png'  # test images with the smallest FID
file_generator_final                 = path_model  + file_id + ',generator,final.pt'
file_generator_best                  = path_model  + file_id + ',generator,best.pt'
file_log                             = path_log    + file_id + '.log'
file_option                          = path_option + file_id + '.ini'

file_plot_loss                       = path_plot   + file_id + ',loss.npz'
file_plot_prediction                 = path_plot   + file_id + ',prediction.npz'
file_plot_FID                        = path_plot   + file_id + ',FID.npz'


file_plot_overfit_r                  = path_plot   + file_id + ',overfit_r.npz'
file_plot_overfit_r_epoch            = path_plot   + file_id + ',overfit_r.npz'

file_figure_test_L1                  = path_figure + file_id + ',test_update,L1.png'
file_figure_FID                      = path_figure + file_id + ',FID.png'

file_figure_overfit                  = path_figure + file_id + ',overfit_r.png'
file_figure_overfit_rv               = path_figure + file_id + ',overfit_rv.png'
file_figure_overfit_rt               = path_figure + file_id + ',overfit_rt.png'
file_figure_overfit_epoch            = path_figure + file_id + ',overfit_ep_r.png'
file_figure_overfit_epoch_rv         = path_figure + file_id + ',overfit_ep_rv.png'
file_figure_overfit_epoch_rt         = path_figure + file_id + ',overfit_ep_rt.png'

file_hadamard_estimate               = path_hadamard + file_id + '.npz'
file_fake_process                    = path_figure   + file_id + ',fake_process.png'



# ======================================================================
# size of the data
# ======================================================================
size_real       = {'channel' : config_dataset_real['channel'], 'height' : config_dataset_real['height'], 'width' : config_dataset_real['width']}
size_fake       = {'channel' : config_dataset_fake['channel'], 'height' : config_dataset_fake['height'], 'width' : config_dataset_fake['width']}
length_latent   = config_dataset_fake['length_latent']


# ======================================================================
# transformations
# ======================================================================
registered_key   = [
'hue-2-3','hue-3-3',
'rot-a3','rot-a4',
'hue-a3','hue-a4',
'hue-4-5',
'hue-5-6',
'hue-7-8',
'Ada']








# adaptive probability of the nvidia algorithm (global parameter)
global_P_nvidia_ada = 0.0

# Excute ADA heuristic (Lines 312-315 in training_loop.py of the official styleGAN2-Ada
# https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py
rt_list=[]
def update_global_P_nvidia_ada(rt, ada_interval=256, ada_target=0.6, speed=0.0000128):
	# ada_interval: num. of real examples to update P
	# ada_target  : the target value of rt
	# speed       : an approximation of "(batch_size * ada_interval) / (ada_kimg * 1000)" in the official code

	global global_P_nvidia_ada
	global rt_list

	rt_list.extend(rt)

	if len(rt_list) > ada_interval:
 
		# compute the adjust step
		ada_rt     = np.sign( np.array(rt_list) ).mean()
		step_size  = speed #ada_interval * speed 
		adjust     = np.sign( ada_rt - ada_target ) * step_size	

		# update P
		global_P_nvidia_ada = np.clip(global_P_nvidia_ada + adjust, 0, 1) # global variable

		# initialize
		rt_list = []


def static_global_P_nvidia_ada(epoch, number_epoch, maxP=0.25, warmup=0.02): 
	global global_P_nvidia_ada

	progress   = float(epoch / number_epoch)
	top_warmup = maxP / 2.0

	if progress <= warmup: # start from 0 and rise to P quickly in the early epoch
		t = progress / warmup 
		P = top_warmup * t
	elif progress <= 2 * warmup:
		t = ( progress - warmup ) / warmup
		P = top_warmup * (1 - t)
	else:
		t = ( progress - 2 * warmup ) / ( 1. - 2 * warmup )			
		P = maxP * t

	global_P_nvidia_ada = P # global variable













		


# Set option_transform and use_tranform
option_transform = {}
for key in registered_key:
	option_transform[key] = True if key in config_transform['option'] else False
if True in option_transform.values():   use_transform = True   # activated when one or more transform are used.
else:                                   use_transform = False

# Exception handlings
if use_transform and 'none' in config_transform['option']:
		print('error: transform option \'none\' cannot be combined with other options, ', option_transform)
		exit(-1)
for key in config_transform['option'] :
	if key not in registered_key and key != 'none':
		print('error: invalid transform option: ', config_transform['option'])
		exit(-1)

# Set transform_probability
transform_probability = config_transform['probability']
if use_transform and transform_probability == 0:
	print(' ** warning **: [Transform] probability = {} is zero with tranform(s):'.format(transform_probability), option_transform)
	time.sleep(5)

# Condition for discriminator
use_condition = config_transform['use_condition']

def generate_transform(option_transform):

	# We make a list of transforms with the code length of (dim_transform).
	# Each of transform code is a one-hot vector like [0,1,0,0].

	dim_transform   = 0   # length of condition vector
	transform       = {}

	if option_transform['rot-a3']:
		name            = 'rot-a3'
		code_length     = 1
		dim_transform   += code_length
		transform[name] = torch.zeros(code_length).type(torch.FloatTensor)


	if option_transform['rot-a4']:
		name            = 'rot-a4'
		code_length     = 1
		dim_transform   += code_length
		transform[name] = torch.zeros(code_length).type(torch.FloatTensor)


	if option_transform['Ada']:
		name            = 'Ada'
		code_length     = 1
		dim_transform   += code_length
		transform[name] = torch.zeros(code_length).type(torch.FloatTensor)



	if option_transform['hue-a3']: # hue in the nvidia stylegan code
		name            = 'hue-a3'
		code_length     = 1 
		dim_transform   += code_length
		transform[name] = torch.zeros(code_length).type(torch.FloatTensor)




	if option_transform['hue-2-3']: # hue in the nvidia stylegan code
		name            = 'hue-2-3'
		code_length     = 1 
		dim_transform   += code_length
		transform[name] = torch.zeros(code_length).type(torch.FloatTensor)

	if option_transform['hue-3-3']: # hue in the nvidia stylegan code
		name            = 'hue-3-3'
		code_length     = 1 
		dim_transform   += code_length
		transform[name] = torch.zeros(code_length).type(torch.FloatTensor)


	if option_transform['hue-4-5']: # hue in the nvidia stylegan code
		name            = 'hue-4-5'
		code_length     = 1 
		dim_transform   += code_length
		transform[name] = torch.zeros(code_length).type(torch.FloatTensor)

	if option_transform['hue-5-6']: # hue in the nvidia stylegan code
		name            = 'hue-5-6'
		code_length     = 1 
		dim_transform   += code_length
		transform[name] = torch.zeros(code_length).type(torch.FloatTensor)

	if option_transform['hue-7-8']: # hue in the nvidia stylegan code
		name            = 'hue-7-8'
		code_length     = 1 
		dim_transform   += code_length
		transform[name] = torch.zeros(code_length).type(torch.FloatTensor)


	if option_transform['hue-a4']: # hue in the nvidia stylegan code
		name            = 'hue-a4'
		code_length     = 1 
		dim_transform   += code_length
		transform[name] = torch.zeros(code_length).type(torch.FloatTensor)



	# ---------------------------------------------------------------------------
	#
	# STEP 2:	
	# Activate the transforms with probability (P).
	#
	# ---------------------------------------------------------------------------

	list_transform = list( transform )
	for name in list_transform: # activate all transforms			
		dim                         = len(transform[name])
		data_index                  = torch.randint(dim, (1,))
		transform[name][data_index] = transform_probability

	return (transform, dim_transform)









# ======================================================================
# transformations
# ======================================================================
def convert_transform_to_feature(transform):
	list_transform = []
	for (name, value) in transform.items():
		list_transform.append(value)
	feature_transform = torch.cat(list_transform)

	return feature_transform

# ======================================================================
(transform, dim_transform) = generate_transform(option_transform)


# ======================================================================
# neural network model
# ======================================================================
dim_feature              = 32  
# dimension factor of convolutional architecture
#   For example, given image with (H,W,C)=(64,64,3) and dim_feature=32,
#   the discriminator follows (64,64,3) -> (32,32,'32') -> (16,16,'64') -> ...

dim_generator_input      = [length_latent, 1, 1]
dim_generator_output     = [size_fake['channel'], size_fake['height'], size_fake['width']]
dim_discriminator_input  = [size_real['channel'], size_real['height'], size_real['width']]
dim_discriminator_output = [1, 1, 1]
dim_condition            = dim_transform 

# ======================================================================
# dataset
# ======================================================================
#path_directory  = config_dataset_real['directory']

name_dataset    = config_dataset_real['name'].lower() 
path_dataset    = root_dataset + name_dataset


# get real data with the specific labels
def split_digit(dataset, labels):
	labels = [int(label) for label in labels]  # string to interger '3' -> 3
	indices = [True if target in labels else False for target in dataset.targets]
	dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices]
	return dataset

if 'MNIST'.lower() in name_dataset:
	#print('{} is called.  A subset of the digits can be used by \'MNIST-1-3-6\' for example.'.format(name_dataset))
	transform_list = []
	transform_list.append( transforms.ToTensor() )
	transform_list.append( transforms.Resize((size_real['height'], size_real['width'])) )
	transform_compose   = transforms.Compose(transform_list)
	dataset_real        = MNIST(root_dataset, download=False, train=True, transform=transform_compose)
	# specific labels
	subset_labels = name_dataset.split('-')[1:]     
	num_images_in_grid  = 25
	num_image_per_trans = 5
	if len(subset_labels) > 0: dataset_real = split_digit(dataset_real, subset_labels)


elif 'cifar10'.lower() in name_dataset:
	transform = transforms.Compose([
		transforms.Resize(size_real['width']),
		transforms.CenterCrop(size_real['width']),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
	])
	dataset_real = CIFAR10(path_dataset, train=True,  download=True, transform=transform)
	compute_FID  = False


elif 'CelebA'.lower() in name_dataset:

	if size_real['channel'] == 1: # gray image
		transform_list = []
		transform_list.append( transforms.Resize(size_real['width']) )
		transform_list.append( transforms.CenterCrop(size_real['height']) )
		transform_list.append( transforms.ToTensor() )
		transform_list.append( transforms.Grayscale(num_output_channels=1) )
		transform_list.append( transforms.Normalize((0.5), (0.5)) )
		transform_compose   = transforms.Compose(transform_list)
	else:	
		# Pytorch Tutorial Setting
		transform_list = []
		transform_list.append( transforms.Resize(size_real['width']) )
		transform_list.append( transforms.CenterCrop(size_real['height']) )
		transform_list.append( transforms.ToTensor() )
		transform_list.append( transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) )
		'''
		transform_list = []
		transform_list.append( transforms.ToTensor() )
		transform_list.append( transforms.CenterCrop((size_real['width'],size_real['height'])) )
		'''
		transform_compose   = transforms.Compose(transform_list)
	#dataset_real        = CelebA(path_dataset, download=False, transform=transform_compose, split='train')
	dataset_real        = ImageFolder(path_dataset, transform=transform_compose)

elif name_dataset == '4digits'.lower():
	transform_list = []
	transform_list.append( transforms.ToTensor() )
	transform_list.append( transforms.Grayscale(num_output_channels=1) )
	transform_list.append( transforms.Resize((size_real['height'], size_real['width'])) )
	transform_compose   = transforms.Compose(transform_list)
	dataset_real        = ImageFolder(path_dataset, transform=transform_compose)

	#compute_FID         = False
	num_images_in_grid  = 4
	num_image_per_trans = 4

elif name_dataset == 'oxflower':
	transform_list = []
	transform_list.append( transforms.ToTensor() )
	transform_list.append( transforms.Resize(size_real['height']) )
	transform_list.append( transforms.CenterCrop(size_real['height']) )
	transform_list.append( transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) )
	transform_compose   = transforms.Compose(transform_list)
	dataset_real        = ImageFolder(path_dataset, transform=transform_compose)
	dataloader_real     = DataLoader(dataset=dataset_real, batch_size=1, drop_last=True, num_workers=2, pin_memory=True)
	#for inputs, targets in iter(dataloader_real):
	#	print(inputs.shape)
	#	print(targets)
	#print(dir(dataset_real))
	#exit(-1)

elif 'food-101' in name_dataset:
	# We introduce a challenging data set of 101 food categories, with 101'000 images. 
	# For each class, 250 manually reviewed test images are provided as well as 750 training images. 
	# On purpose, the training images were not cleaned, and thus still contain some amount of noise. 
	# This comes mostly in the form of intense colors and sometimes wrong labels. 
	# All images were rescaled to have a maximum side length of 512 pixels.
	transform_list = []
	transform_list.append( transforms.ToTensor() )
	transform_list.append( transforms.Resize(size_real['height']) )
	transform_list.append( transforms.CenterCrop(size_real['height']) )
	transform_list.append( transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) )
	transform_compose   = transforms.Compose(transform_list)
	dataset_real        = Food101(path_dataset, split='train',  download=True, transform=transform_compose)


elif 'flowers102' in name_dataset:
	transform_list = []
	transform_list.append( transforms.ToTensor() )
	transform_list.append( transforms.Resize(size_real['height']) )
	transform_list.append( transforms.CenterCrop(size_real['height']) )
	transform_list.append( transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) )
	transform_compose   = transforms.Compose(transform_list)
	dataset_real        = Flowers102(path_dataset, split='train',  download=True, transform=transform_compose)

elif 'stanford-cars' in name_dataset:
	# The Cars dataset contains 16,185 images of 196 classes of cars. 
	# The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split
	transform_list = []
	transform_list.append( transforms.ToTensor() )
	transform_list.append( transforms.Resize(size_real['height']) )
	transform_list.append( transforms.CenterCrop(size_real['height']) )
	transform_list.append( transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) )
	transform_compose   = transforms.Compose(transform_list)
	dataset_real        = StanfordCars(path_dataset, split='train',  download=True, transform=transform_compose)


elif 'lsun' in name_dataset: # 'lsun-church'

	transform=transforms.Compose([
		transforms.Resize(size_real['height']),
		transforms.CenterCrop(size_real['height']),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])

	classes=[]
	if 'church'     in name_dataset:  classes.append('church_outdoor_train')
	if 'conference' in name_dataset:  classes.append('conference_room_train')
	if 'restaurant' in name_dataset:  classes.append('restaurant_train')
	if 'tower'      in name_dataset:  classes.append('tower_train')
	if 'bridge'     in name_dataset:  classes.append('tower_bridge')

	if len(classes) == 0:
		print('error lsun dataset should be called with labels as: lsun-church')
		exit(-1)

	dataset_real = LSUN(root_dataset+'lsun', classes=classes, transform=transform)
	print('LSUN dataset with classes: {}'.format(classes))
	print('Total num. of examples: ', len(dataset_real))
	time.sleep(10)
	#exit(-1)

	
else:
	pass




# activate when using dataset_real
# ======================================================================
# subset of the dataset
# ======================================================================

size_data_real  = config_dataset_real['size_data'] # 'all' or number of data

if size_data_real.lower() == 'all' or size_data_real.lower() == '1':
	num_size_data_real   = int( len(dataset_real ) )
	sampler=RandomSampler(range(len(dataset_real)))
else:
	print('1/({}) subset of data are used.'.format(size_data_real))
	num_size_data_real   = int( len(dataset_real ) / float(size_data_real) )
	subset_indices   = torch.randperm(len(dataset_real))[:num_size_data_real]
	sampler=SubsetRandomSampler(subset_indices)

# ======================================================================
# real dataset 
# ======================================================================
dataloader_real         = DataLoader(dataset=dataset_real, batch_size=1, sampler=sampler, drop_last=True, num_workers=2, pin_memory=True)

file_temporal_real      = dir_temporal_real + name_dataset + ',M,' + size_data_real + ',h,' + str(size_real['height']) + ',w,' + str(size_real['width']) + ',' + str(config_dataset_real['normalize'].lower()) + '.pt'

on_memory_dataset_real = []
with open(file_temporal_real + '.lock', 'a+') as cache_lock:
	# Only one of the four GPU processes may create the shared dataset cache.
	fcntl.flock(cache_lock, fcntl.LOCK_EX)
	if use_temporal_real and os.path.exists(file_temporal_real):
		print('Loading on-memory real data from ', file_temporal_real)
		on_memory_dataset_real = torch.load(file_temporal_real)
		print('done')
	else:
		print('Loading real data from data loader:')
		for i, (batch_data_real, batch_label_real) in enumerate(tqdm(dataloader_real)):
			if config_dataset_real['normalize'].lower() == 'scale':
				value_min       = config_dataset_real['normalize_value1']
				value_max       = config_dataset_real['normalize_value2']
				batch_data_real = normalize_tensor(batch_data_real, value_min, value_max)
			else:
				batch_data_real = batch_data_real
			on_memory_dataset_real.append(batch_data_real)
		on_memory_dataset_real = torch.cat(on_memory_dataset_real, dim=0)
		print('done.')

		if use_temporal_real:
			print('Saving real data in a temporal directory --> ', file_temporal_real)
			torch.save(on_memory_dataset_real, file_temporal_real)
			print('done.')
	fcntl.flock(cache_lock, fcntl.LOCK_UN)



print(size_data_real, num_size_data_real, len(dataloader_real))



# y-range in plot
if 'celeba' in name_dataset.lower(): ylim_FID = [0,50]
else:                                ylim_FID = [0,400]



# ======================================================================
# Training, test, validation data
# ======================================================================
num_train_data       = None
num_test_data        = None 
num_valid_data       = 1 * config_optimize['size_batch']  #4 * config_optimize['size_batch'] # batch size * 4 
batch_size_FID       = 20
interval_compute_FID = max(1, int(config_optimize['number_epoch'] / 10.))
best_FID             = 9999999   # a temporal value
best_FID_epoch       = None
final_FID            = None

# -----------------------------------------------------------------------------------------------------
# spliting Real Examples (incomplete)
#
# Given the real set (on_memory_dataset_real),
#   training set (train_data_real)   is a subset of the real set,
#   test set (test_data_real)        is a small subset of the training set that is shuffled every time,
#   validation set (valid_data_real) is the complement set of the training set.
# ------------------------------------------------------------------------------------------------------
# permute the real data

nonpermute_data_real   = on_memory_dataset_real[0:config_optimize['size_batch'] ]

on_memory_dataset_real = on_memory_dataset_real[np.random.permutation(len(on_memory_dataset_real))]

# split the real in training and validation sets.   *note* real set can be reduced later 
train_data_real = on_memory_dataset_real[num_valid_data:]
valid_data_real = on_memory_dataset_real[:num_valid_data]


# finalize the training set
if config_dataset_real['size_data'] != 'all':
	# If config_dataset_real['size_data'] == 10, we use (1/10) of the dataset
	num_train_data       = int( len( train_data_real ) / float(config_dataset_real['size_data']) )
	# crop a subset of data 
	train_data_real = train_data_real[0:num_train_data]

num_train_data = len(train_data_real)


# test data size should be lower than the training data size
# and also should be factor of 5
factor_round      = 5
num_test_data     = np.min([4000, num_train_data])
num_test_data     = int( np.floor(num_test_data / factor_round ) * factor_round )


# sample a test set from the training set, and also prepare a latent for fakes
test_data_real    = train_data_real[np.random.permutation(len(train_data_real))][:num_test_data]
test_batch_latent = torch.randn((num_test_data, length_latent, 1, 1)).to(device)

print('the real examples: ', on_memory_dataset_real.shape)
print('  +tranining set:',  train_data_real.shape)
print('  |   + test set:',       test_data_real.shape)
print('  +validation set:', valid_data_real.shape)


# ======================================================================
# batch size
# ======================================================================
dim_dataset_real = [num_train_data, size_real['channel'], size_real['height'], size_real['width']]
print('number of real data', num_train_data)


# ========================================================================================================
# iteration for the mini-batch of real data
#     
#     number_data_epoch_real       : number of training data in one epoch (= batch size * number of batch)
#     number_data_epoch_real_real  : number of the remaining training-data
#
# ========================================================================================================
size_batch                  = config_optimize['size_batch'] 

# training data
number_batch_real           = np.floor(num_train_data / size_batch).astype(int)
number_data_epoch_real      = number_batch_real * size_batch 
number_data_epoch_real_last = num_train_data - number_data_epoch_real
dim_dataset_real            = [num_train_data, size_real['channel'], size_real['height'], size_real['width']]
dim_batch_data_real         = [size_batch, size_fake['channel'], size_fake['height'], size_fake['width']]
dim_batch_latent            = [size_batch, length_latent, 1, 1]
index_data_epoch_real       = np.arange(0, number_data_epoch_real) 
index_data_epoch_real_last  = np.arange(number_data_epoch_real, num_train_data)


# validation data
number_batch_valid          = int( num_valid_data / float(size_batch) )
number_data_epoch_valid     = number_batch_valid * size_batch
index_data_epoch_valid      = np.arange(0, number_data_epoch_valid)
index_data_epoch_valid_last = np.arange(number_data_epoch_valid, num_valid_data)

# shuffle index of validation real data
index_batch_valid           = np.random.permutation(index_data_epoch_valid)
index_batch_valid_slice     = np.reshape(index_batch_valid, [number_batch_valid, size_batch]) 
index_j_epoch_valid         = 0


# ======================================================================
# neural networks
# ======================================================================
# Generator
#if config_network['generator'] == 'DCGAN_Generator':      generator = DCGAN_Generator(length_latent, size_real['channel']).to(device)
#elif config_network['generator'] == 'DCGAN_Generator32':  generator = DCGAN_Generator32(length_latent, size_real['channel']).to(device)
if config_network['generator'] == 'DCGAN2_Generator':   generator = DCGAN2_Generator(length_latent, size_real['channel']).to(device)
elif config_network['generator'] == 'DCGAN2_Generator32': generator = DCGAN2_Generator32(length_latent, size_real['channel']).to(device)
elif config_network['generator'] == 'DCGAN3_Generator':   generator = DCGAN3_Generator(length_latent, size_real['channel']).to(device)
elif config_network['generator'] == 'DCGAN3_Generator32': generator = DCGAN3_Generator32(length_latent, size_real['channel']).to(device)
#elif config_network['generator'] == 'SNGAN_Generator':    generator = SNGAN_Generator(length_latent, size_real['channel']).to(device) 
#lif config_network['generator'] == 'SNGAN_Generator32':  generator = SNGAN_Generator32(length_latent, size_real['channel']).to(device)  
#elif config_network['generator'] == 'GAN_Generator_wo_Activation':      generator = GAN_Generator(length_latent, dim_generator_output, activation=False).to(device)
#elif config_network['generator'] == 'GAN_Generator':      generator = GAN_Generator(length_latent, dim_generator_output).to(device)
elif config_network['generator'] == 'GAN64_Generator':    generator = GAN64_Generator(length_latent, dim_generator_output).to(device)
#elif config_network['generator'] == 'GAN_reverseG_Generator':    generator = GAN_reverseG_Generator(length_latent, dim_generator_output).to(device)
else:                                                     generator = Network_Generator(dim_generator_input, dim_generator_output, dim_feature, config_network['generator']).to(device)

# Disriminator 


if 'DCGAN2_Discriminator32' in config_network['discriminator']:

	if 'wo_Sigmoid' in  config_network['discriminator']: activation = False
	else:                                                activation = True

	if 'Bottleneck32'     in config_network['discriminator']: bottleneck=32
	elif 'Bottleneck64'   in config_network['discriminator']: bottleneck=64
	elif 'Bottleneck128'  in config_network['discriminator']: bottleneck=128
	elif 'Bottleneck256'  in config_network['discriminator']: bottleneck=256
	elif 'Bottleneck512'  in config_network['discriminator']: bottleneck=512
	elif 'Bottleneck1024' in config_network['discriminator']: bottleneck=1024
	else: bottleneck=512

	if   use_condition == 'none' or use_transform == False: discriminator = DCGAN2_Discriminator32(size_real['channel'],activation=activation,bottleneck=bottleneck).to(device)
	elif use_condition == 'classifier':                     discriminator = DCGAN2_Discriminator32_Classifier_Condition(size_real['channel'], dim_condition=dim_condition, activation=activation,bottleneck=bottleneck).to(device)
	elif 'estimate' in use_condition:                       discriminator = DCGAN2_Discriminator32_Estimate_Condition(size_real['channel'], dim_condition=dim_condition, activation=activation).to(device)
	else:
		print('use_condition = {} is not implemented'.format(use_condition))
		exit(-1)


elif 'DCGAN2_Discriminator' in config_network['discriminator']: # 'DCGAN_Discriminator' or  'DCGAN_Discriminator_wo_Sigmoid'

	if 'wo_Sigmoid' in  config_network['discriminator']: activation = False
	else:                                                activation = True

	if 'Bottleneck32'     in config_network['discriminator']: bottleneck=32
	elif 'Bottleneck64'   in config_network['discriminator']: bottleneck=64
	elif 'Bottleneck128'  in config_network['discriminator']: bottleneck=128
	elif 'Bottleneck256'  in config_network['discriminator']: bottleneck=256
	elif 'Bottleneck512'  in config_network['discriminator']: bottleneck=512
	elif 'Bottleneck1024' in config_network['discriminator']: bottleneck=1024
	else: bottleneck=512


	if   use_condition == 'none' or use_transform == False: discriminator = DCGAN2_Discriminator(size_real['channel'],activation=activation,bottleneck=bottleneck).to(device)
	elif use_condition == 'classifier':                     discriminator = DCGAN2_Discriminator_Classifier_Condition(size_real['channel'], dim_condition=dim_condition, activation=activation,bottleneck=bottleneck).to(device)
	elif use_condition == 'conv2':                          discriminator = DCGAN2_Discriminator_Conv2_Condition(length_latent, size_real['channel'], dim_condition=dim_condition, activation=activation).to(device)
	elif 'estimate' in use_condition:                       discriminator = DCGAN2_Discriminator_Estimate_Condition(size_real['channel'], dim_condition=dim_condition, activation=activation).to(device)

	else:
		print('use_condition = {} is not implemented'.format(use_condition))
		exit(-1)



elif 'DCGAN3_Discriminator32' in config_network['discriminator']:

	if 'wo_Sigmoid' in  config_network['discriminator']: activation = False
	else:                                                activation = True

	if 'Bottleneck32'     in config_network['discriminator']: bottleneck=32
	elif 'Bottleneck64'   in config_network['discriminator']: bottleneck=64
	elif 'Bottleneck128'  in config_network['discriminator']: bottleneck=128
	elif 'Bottleneck256'  in config_network['discriminator']: bottleneck=256
	elif 'Bottleneck512'  in config_network['discriminator']: bottleneck=512
	elif 'Bottleneck1024' in config_network['discriminator']: bottleneck=1024
	else: bottleneck=512

	if   use_condition == 'none' or use_transform == False: discriminator = DCGAN3_Discriminator32(size_real['channel'],activation=activation,bottleneck=bottleneck).to(device)
	elif use_condition == 'classifier':                     discriminator = DCGAN3_Discriminator32_Classifier_Condition(size_real['channel'], dim_condition=dim_condition, activation=activation,bottleneck=bottleneck).to(device)
	elif 'estimate' in use_condition:                       discriminator = DCGAN3_Discriminator32_Estimate_Condition(size_real['channel'], dim_condition=dim_condition, activation=activation).to(device)
	else:
		print('use_condition = {} is not implemented'.format(use_condition))
		exit(-1)


elif 'DCGAN3_Discriminator' in config_network['discriminator']: # 'DCGAN_Discriminator' or  'DCGAN_Discriminator_wo_Sigmoid'

	if 'wo_Sigmoid' in  config_network['discriminator']: activation = False
	else:                                                activation = True

	if 'Bottleneck32'     in config_network['discriminator']: bottleneck=32
	elif 'Bottleneck64'   in config_network['discriminator']: bottleneck=64
	elif 'Bottleneck128'  in config_network['discriminator']: bottleneck=128
	elif 'Bottleneck256'  in config_network['discriminator']: bottleneck=256
	elif 'Bottleneck512'  in config_network['discriminator']: bottleneck=512
	elif 'Bottleneck1024' in config_network['discriminator']: bottleneck=1024
	else: bottleneck=512


	if   use_condition == 'none' or use_transform == False: discriminator = DCGAN3_Discriminator(size_real['channel'],activation=activation,bottleneck=bottleneck).to(device)
	elif use_condition == 'classifier':                     discriminator = DCGAN3_Discriminator_Classifier_Condition(size_real['channel'], dim_condition=dim_condition, activation=activation,bottleneck=bottleneck).to(device)
	elif use_condition == 'conv2':                          discriminator = DCGAN3_Discriminator_Conv2_Condition(length_latent, size_real['channel'], dim_condition=dim_condition, activation=activation).to(device)
	elif 'estimate' in use_condition:                       discriminator = DCGAN3_Discriminator_Estimate_Condition(size_real['channel'], dim_condition=dim_condition, activation=activation).to(device)

	else:
		print('use_condition = {} is not implemented'.format(use_condition))
		exit(-1)


elif 'GAN_Discriminator' in config_network['discriminator']: # 'DCGAN_Discriminator' or  'DCGAN_Discriminator_wo_Sigmoid'

	if 'wo_Sigmoid' in  config_network['discriminator']: activation = False
	else:                                                activation = True

	if   use_condition == 'none' or use_transform == False: discriminator = GAN_Discriminator(image_shape=dim_discriminator_input, activation=activation).to(device)
	elif use_condition == 'classifier':                     discriminator = GAN_Discriminator_Condition(dim_condition=dim_condition, image_shape=dim_discriminator_input, activation=activation).to(device)
	else:
		print('use_condition = {} is not implemented'.format(use_condition))
		exit(-1)
else:
	if use_transform and use_condition: discriminator = Network_Discriminator_Condition(dim_discriminator_input, dim_discriminator_output, dim_feature, dim_condition, config_network['discriminator']).to(device) 
	else:                               discriminator = Network_Discriminator(dim_discriminator_input, dim_discriminator_output, dim_feature, config_network['discriminator']).to(device) 


#discriminator   = models.resnet101(pretrained=False).to(device)


# ======================================================================
# Epoch, and Scale
# ======================================================================
number_epoch = config_optimize['number_epoch']
number_scale = int(np.log2(size_real['height'] / size_fake['height'])) + 1


# ======================================================================
# hypter-parameters for the discriminator 
# ======================================================================
option_loss_discriminator   = config_optimize_discriminator['loss']
gradient_regularization     = config_optimize_discriminator['gradient_regularization']
weight_decay_discriminator  = config_optimize_discriminator['weight_decay']
momentum_discriminator      = config_optimize_discriminator['momentum']
lr_initial_discriminator    = config_optimize_discriminator['learning_rate_initial']
lr_middle_discriminator     = config_optimize_discriminator['learning_rate_middle']
lr_final_disciminator       = config_optimize_discriminator['learning_rate_final']
lr_slope_discriminator      = config_optimize_discriminator['learning_rate_slope']
lr_ratio_division_discriminator = config_optimize_discriminator['learning_rate_ratio_division']

# ======================================================================
# hypter-parameters for the generator 
# ======================================================================
option_loss_generator     = config_optimize_generator['loss']
weight_total_variation    = config_optimize_generator['weight_total_variation']
weight_decay_generator    = config_optimize_generator['weight_decay']
momentum_generator        = config_optimize_generator['momentum']
lr_initial_generator      = config_optimize_generator['learning_rate_initial']
lr_middle_generator       = config_optimize_generator['learning_rate_middle']
lr_final_generator        = config_optimize_generator['learning_rate_final']
lr_slope_generator        = config_optimize_generator['learning_rate_slope']
lr_ratio_division_generator = config_optimize_generator['learning_rate_ratio_division']


# ======================================================================
# optimizer
# ======================================================================
if config_optimize['algorithm'].lower() == 'sgd':
	optimizer_generator      = torch.optim.SGD(generator.parameters(),      lr=lr_initial_generator,     weight_decay=weight_decay_generator,     momentum=momentum_generator)
	optimizer_discriminator  = torch.optim.SGD(discriminator.parameters(),  lr=lr_initial_discriminator, weight_decay=weight_decay_discriminator, momentum=momentum_discriminator)
	use_scheduler = True

elif 'Adam'.lower() in config_optimize['algorithm'].lower():

	# How to call Adam with hyper-paramters:
	# 'Adam'           -> b1=0.5, b2=0.999 as default value
	# 'Adam-0.8'       -> b1=0.8, b2=0.999
	# 'Adam-0.8-0.888' -> b1=0.8, b2=0.888

	adam_option = config_optimize['algorithm'].split('-')

	if len(adam_option) == 1:
		beta1 = 0.5
		beta2 = 0.999
	elif len(adam_option) == 2:
		beta1 = float( adam_option[1] )
		beta2 = 0.999
	else:
		beta1 = float( adam_option[1] )
		beta2 = float( adam_option[2] )

	optimizer_generator      = torch.optim.Adam(generator.parameters(),      lr=lr_initial_generator,     betas=(beta1,beta2), weight_decay=weight_decay_generator)
	optimizer_discriminator  = torch.optim.Adam(discriminator.parameters(),  lr=lr_initial_discriminator, betas=(beta1,beta2), weight_decay=weight_decay_discriminator)
	use_scheduler = False
else:
	pass

if use_scheduler:
	scheduler_discriminator = scheduler_learning_rate_sigmoid_double(optimizer_discriminator, number_epoch, 
		[lr_initial_discriminator, lr_middle_discriminator],
		[lr_middle_discriminator, lr_final_disciminator],
		[lr_slope_discriminator, lr_slope_discriminator],
		[0, 0],
		lr_ratio_division_discriminator)

	scheduler_generator = scheduler_learning_rate_sigmoid_double(optimizer_generator, number_epoch, 
		[lr_initial_generator, lr_middle_generator],
		[lr_middle_generator, lr_final_generator],
		[lr_slope_generator, lr_slope_generator],
		[0, 0],
		lr_ratio_division_generator)


# ======================================================================
# for saving results
# ======================================================================

#logger              = Logger(file_log)


value_loss_total_mean         = np.zeros(number_epoch)
value_loss_generator_mean     = np.zeros(number_epoch)
value_loss_generator_std      = np.zeros(number_epoch)
value_loss_discriminator_mean = np.zeros(number_epoch)
value_loss_discriminator_std  = np.zeros(number_epoch)

value_prediction_real_mean    = np.zeros(number_epoch)
value_prediction_real_std     = np.zeros(number_epoch)
value_prediction_fake_mean    = np.zeros(number_epoch)
value_prediction_fake_std     = np.zeros(number_epoch)
value_prediction_sum          = np.zeros(number_epoch)

value_test_data_fake_L1_mean  = []
value_test_data_fake_L1_std   = []

value_test_data_FID           = []


# ---------------------------------------------------------------------------
#  
# trasform functions
#  
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
#  
# transform the latent data to the input to the discriminator
# The input to the discriminator is considered as the generated fake images
#  
# ---------------------------------------------------------------------------
def transform_data(data, transform):

	# note: given L length hot-vector, torch.where() resturns [0,1,...,(L-1)]

	for name in list( transform ):

		if transform[name].sum() == 0:
			#print('we skip non activated transform {}, sum {}:'.format(name, transform[name].sum()))
			pass

		elif name == 'orignal':
			pass


		elif name == 'hue-a3': # hue in the stylegan code
			batch_size, num_channels, height, width = data.shape
			#
			v     = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=data.device) # Luma axis.
			theta = np.random.choice([-0.5,0.5,1], size=len(data))  # -0.5:purple, 0:original, 0.5:green, 1.0:blue faces (1.5: purple)
			theta[np.random.random(size=theta.shape) > transform[name].numpy()] = 0.0 # use the orignal data with probability (transform[name])
			theta = torch.from_numpy(theta).type(torch.FloatTensor).to(data.device)
			theta = theta * np.pi
			C     = rotate3d(v, theta)
			#
			data = data.reshape([batch_size, num_channels, height * width])
			data = C[:, :3, :3] @ data + C[:, :3, 3:]
			data = data.reshape([batch_size, num_channels, height, width])

		elif name == 'hue-a4': # hue in the stylegan code
			batch_size, num_channels, height, width = data.shape
			#
			v     = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=data.device) # Luma axis.
			theta = np.random.choice([-0.5,0,0.5,1], size=len(data))  # -0.5:purple, 0:original, 0.5:green, 1.0:blue faces (1.5: purple)
			theta[np.random.random(size=theta.shape) > transform[name].numpy()] = 0.0 # use the orignal data with probability (transform[name])
			theta = torch.from_numpy(theta).type(torch.FloatTensor).to(data.device)
			theta = theta * np.pi
			C     = rotate3d(v, theta)
			#
			data = data.reshape([batch_size, num_channels, height * width])
			data = C[:, :3, :3] @ data + C[:, :3, 3:]
			data = data.reshape([batch_size, num_channels, height, width])


		elif name == 'hue-3-3': # hue in the stylegan code
			batch_size, num_channels, height, width = data.shape
			#
			v     = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=data.device) # Luma axis.
			theta = np.random.choice([-0.6666,0,0.6666], size=len(data))  
			theta[np.random.random(size=theta.shape) > transform[name].numpy()] = 0.0 # use the orignal data with probability (transform[name])
			theta = torch.from_numpy(theta).type(torch.FloatTensor).to(data.device)
			theta = theta * np.pi
			C     = rotate3d(v, theta)
			#
			data = data.reshape([batch_size, num_channels, height * width])
			data = C[:, :3, :3] @ data + C[:, :3, 3:]
			data = data.reshape([batch_size, num_channels, height, width])

		elif name == 'hue-2-3': # hue in the stylegan code
			batch_size, num_channels, height, width = data.shape
			#
			v     = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=data.device) # Luma axis.
			theta = np.random.choice([-0.6666,0.6666], size=len(data))  
			theta[np.random.random(size=theta.shape) > transform[name].numpy()] = 0.0 # use the orignal data with probability (transform[name])
			theta = torch.from_numpy(theta).type(torch.FloatTensor).to(data.device)
			theta = theta * np.pi
			C     = rotate3d(v, theta)
			#
			data = data.reshape([batch_size, num_channels, height * width])
			data = C[:, :3, :3] @ data + C[:, :3, 3:]
			data = data.reshape([batch_size, num_channels, height, width])

		elif name == 'hue-4-5': # hue in the stylegan code
			batch_size, num_channels, height, width = data.shape
			#
			v     = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=data.device) # Luma axis.
			theta = np.random.choice([-0.8,-0.4,0.4,0.8], size=len(data))  
			theta[np.random.random(size=theta.shape) > transform[name].numpy()] = 0.0 # use the orignal data with probability (transform[name])
			theta = torch.from_numpy(theta).type(torch.FloatTensor).to(data.device)
			theta = theta * np.pi
			C     = rotate3d(v, theta)
			#
			data = data.reshape([batch_size, num_channels, height * width])
			data = C[:, :3, :3] @ data + C[:, :3, 3:]
			data = data.reshape([batch_size, num_channels, height, width])


		elif name == 'hue-5-6': # hue in the stylegan code
			batch_size, num_channels, height, width = data.shape
			#
			v     = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=data.device) # Luma axis.
			theta = np.random.choice([-0.6666,-0.3333,0.3333,0.6666,1], size=len(data))  
			theta[np.random.random(size=theta.shape) > transform[name].numpy()] = 0.0 # use the orignal data with probability (transform[name])
			theta = torch.from_numpy(theta).type(torch.FloatTensor).to(data.device)
			theta = theta * np.pi
			C     = rotate3d(v, theta)
			#
			data = data.reshape([batch_size, num_channels, height * width])
			data = C[:, :3, :3] @ data + C[:, :3, 3:]
			data = data.reshape([batch_size, num_channels, height, width])


		elif name == 'hue-7-8': # hue in the stylegan code
			batch_size, num_channels, height, width = data.shape
			#
			v     = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=data.device) # Luma axis.
			theta = np.random.choice([-0.75,-0.5,-0.25,0.25,0.50,0.75,1], size=len(data))  
			theta[np.random.random(size=theta.shape) > transform[name].numpy()] = 0.0 # use the orignal data with probability (transform[name])
			theta = torch.from_numpy(theta).type(torch.FloatTensor).to(data.device)
			theta = theta * np.pi
			C     = rotate3d(v, theta)
			#
			data = data.reshape([batch_size, num_channels, height * width])
			data = C[:, :3, :3] @ data + C[:, :3, 3:]
			data = data.reshape([batch_size, num_channels, height, width])


		elif name == 'rot-a4':
			option_rotate = torch.randint(low=0,high=4,size=(len(data),)) # low is inclusive and high is *exclusive*
			option_rotate[np.random.random(size=option_rotate.shape) > transform[name].numpy()] = 0 # use the orignal data with probability (transform[name])
			transformed = [torch.rot90(data[i].unsqueeze(0), option_rotate[i], [2,3]) for i in range(len(data))]
			data        = torch.cat(transformed, dim=0)


		elif name == 'rot-a3': # w/o the original data 
			option_rotate = torch.randint(low=1,high=4,size=(len(data),)) # low is inclusive and high is *exclusive*
			option_rotate[np.random.random(size=option_rotate.shape) > transform[name].numpy()] = 0 # use the orignal data with probability (transform[name])
			transformed = [torch.rot90(data[i].unsqueeze(0), option_rotate[i], [2,3]) for i in range(len(data))]
			data        = torch.cat(transformed, dim=0)
		

		elif name == 'Ada':
			
			# Given global_P_nvidia_ada (P), apply rotation-4 and hue-4 with P
			# rot-4
			option_rotate = torch.randint(low=0,high=4,size=(len(data),)) # low is inclusive and high is *exclusive*
			option_rotate[np.random.random(size=option_rotate.shape) > global_P_nvidia_ada] = 0 # use the orignal data with probability (transform[name])
			transformed = [torch.rot90(data[i].unsqueeze(0), option_rotate[i], [2,3]) for i in range(len(data))]
			data        = torch.cat(transformed, dim=0)
			batch_size, num_channels, height, width = data.shape
			#
			# hue-4
			v     = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=data.device) # Luma axis.
			theta = np.random.choice([-0.5,0,0.5,1], size=len(data))  # -0.5:purple, 0:original, 0.5:green, 1.0:blue faces (1.5: purple)
			theta[np.random.random(size=theta.shape) > global_P_nvidia_ada] = 0.0 # use the orignal data with probability (transform[name])
			theta = torch.from_numpy(theta).type(torch.FloatTensor).to(data.device)
			theta = theta * np.pi
			C     = rotate3d(v, theta)
			#
			data = data.reshape([batch_size, num_channels, height * width])
			data = C[:, :3, :3] @ data + C[:, :3, 3:]
			data = data.reshape([batch_size, num_channels, height, width])




	return data




# ---------------------------------------------------------------------------
#  
# function for computing the loss of the discriminator
#  
# ---------------------------------------------------------------------------

# gradient regularization for GAN-gr
#https://github.com/LMescheder/GAN_stability/blob/c1f64c9efeac371453065e5ce71860f4c2b97357/gan_training/train.py#L123
def compute_grad2(d_out, x_in):

	batch_size = x_in.shape[0]
	grad_dout  = autograd.grad( outputs=d_out.sum(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True )[0]
	grad_dout2 = grad_dout.pow(2)
	assert(grad_dout2.shape == x_in.shape)
	#reg = grad_dout2.view(batch_size, -1).sum(1)
	reg = grad_dout2.reshape(batch_size, -1).sum(1)
	return reg


# gradient penalty for wgan
def compute_gradient_penalty(D, real_samples, fake_samples):
	"""Calculates the gradient penalty loss for WGAN GP"""
	# Random weight term for interpolation between real and fake samples
	alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
	# Get random interpolation between real and fake samples
	interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
	d_interpolates = D(interpolates).unsqueeze(-1)

	fake = Variable( torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False ).to(device)
	# Get gradient w.r.t. interpolates
	gradients = autograd.grad(
		outputs=d_interpolates,
		inputs=interpolates,
		grad_outputs=fake,
		create_graph=True,
		retain_graph=True,
		only_inputs=True,
	)[0]
	gradients = gradients.view(gradients.size(0), -1)
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
	return gradient_penalty


def compute_prediction(discriminator, data, transform, use_condition):
	
	# -----------------------------------------------------------------------
	# Step 1: compute prediction
	# -----------------------------------------------------------------------
	if use_transform:
		transform_feature          = convert_transform_to_feature(transform)
		condition                  = transform_feature.repeat(data.shape[0], 1).to(device)
		data_transform             = transform_data(data, transform)

		# compute prediction
		#if use_condition == 'none':       prediction           = discriminator(data_transform.requires_grad_())
		if use_condition == 'none':       prediction           = discriminator(data_transform)
		elif 'estimate' in use_condition: prediction, estimate = discriminator(data_transform)
		else:                             prediction           = discriminator(data_transform, condition)

	else:
		data_transform = data
		prediction     = discriminator(data_transform)

	return prediction



def transform_batch(data, transform, use_condition):
	if use_transform:
		transform_feature  = convert_transform_to_feature(transform)
		condition          = transform_feature.repeat(data.shape[0], 1).to(device)
		data               = transform_data(data, transform)
	return data




def compute_loss_discriminator(discriminator, data_fake, data_real, transform, use_condition, option_loss, gradient_regularization=0):
	
	# -----------------------------------------------------------------------
	# Step 1: compute prediction
	# -----------------------------------------------------------------------
	if use_transform:

		# real data
		transform_feature          = convert_transform_to_feature(transform)
		condition                  = transform_feature.repeat(data_real.shape[0], 1).to(device)
		data_real_transform        = transform_data(data_real, transform)

		# fake data
		transform_feature          = convert_transform_to_feature(transform)
		condition                  = transform_feature.repeat(data_fake.shape[0], 1).to(device)
		data_fake_transform        = transform_data(data_fake, transform)

		# compute prediction
		if use_condition == 'none':
			prediction_real     = discriminator(data_real_transform.requires_grad_())
			prediction_fake     = discriminator(data_fake_transform)

		elif 'estimate' in use_condition:
			prediction_real, estimate_condition_real = discriminator(data_real_transform.requires_grad_())
			prediction_fake, estimate_condition_fake = discriminator(data_fake_transform)

		else:
			prediction_real     = discriminator(data_real_transform.requires_grad_(), condition)
			prediction_fake     = discriminator(data_fake_transform, condition)

	else:
			data_real_transform = data_real
			data_fake_transform = data_fake
			prediction_real     = discriminator(data_real_transform.requires_grad_())
			prediction_fake     = discriminator(data_fake_transform)


	# -----------------------------------------------------------------------
	# Step 2: compute loss
	# -----------------------------------------------------------------------
	if option_loss == 'cross_entropy':
		print('cross_entropy is not available.  Choose one of \'bce\' and \'bce_with_logits\'')
		exit(-1)

	if option_loss == 'bce':
		criterion   = nn.BCELoss()
		label_real  = torch.ones_like(prediction_real)
		label_fake  = torch.zeros_like(prediction_fake)
		loss_real   = criterion(prediction_real, label_real)
		loss_fake   = criterion(prediction_fake, label_fake)
		loss_discriminator = (loss_real + loss_fake) / 2.0
		
	elif option_loss == 'bce_with_logits':
		criterion   = nn.BCEWithLogitsLoss()
		label_real  = torch.ones_like(prediction_real)
		label_fake  = torch.zeros_like(prediction_fake)
		loss_real   = criterion(prediction_real, label_real)
		loss_fake   = criterion(prediction_fake, label_fake)
		loss_discriminator = (loss_real + loss_fake) / 2.0
		
	elif option_loss == 'wasserstein':
		loss_discriminator  = torch.mean(prediction_fake) - torch.mean(prediction_real)

	elif option_loss == 'l2':
		criterion   = nn.MSELoss()
		#sig         = nn.Sigmoid() 
		label_real  = torch.ones_like(prediction_real)
		label_fake  = torch.zeros_like(prediction_fake)
		loss_real   = criterion(prediction_real, label_real)
		loss_fake   = criterion(prediction_fake, label_fake)
		loss_discriminator = (loss_real + loss_fake) / 2.0

	else:
		loss_discriminator = None



	# -----------------------------------------------------------------------
	# Step 2.2: compute loss of estimated condition loss
	# -----------------------------------------------------------------------
	if 'estimate' in use_condition:

		estimate_option = use_condition.split('-')
		
		if len(estimate_option) == 1: coefficient_loss_condition = 1
		else:                         coefficient_loss_condition = float(estimate_option[-1])


		if option_loss == 'bce_with_logits':
			F = torch.nn.functional
			loss_condition_real = torch.sum( F.binary_cross_entropy_with_logits(input=estimate_condition_real, target=condition) )
			loss_condition_fake = torch.sum( F.binary_cross_entropy_with_logits(input=estimate_condition_fake, target=condition) )
			loss_condition = (loss_condition_real + loss_condition_fake) / 2.0

			loss_discriminator = loss_discriminator + coefficient_loss_condition * loss_condition # coefficient will be here
		else:
			print('error: criterion for esimated condition loss is not implemented: ', option_loss)
			exit(-1)

	# the following code is applied after the update with wasserstein loss
	#if option_loss_discriminator == 'wasserstein':
	#	threshold = 1 
	#
	#	for p in discriminator.parameters():
	#		p.data.clamp_(-threshold, threshold)

	if gradient_regularization == 0:  
		loss_gr = 0.
	else:
		# gradient PENALY
		if option_loss == 'wasserstein': loss_gr = gradient_regularization * compute_gradient_penalty(discriminator, data_real.data, data_fake.data)
		
		# gradient REGULARIZATION
		else:                            loss_gr = gradient_regularization * compute_grad2(prediction_real, data_real_transform).mean()
	
	loss_discriminator  = loss_discriminator + loss_gr



	return loss_discriminator, prediction_real, prediction_fake



# ---------------------------------------------------------------------------
#  
# function for computing the loss of the generator
#  
# ---------------------------------------------------------------------------
def compute_loss_generator(discriminator, data_fake, transform, use_condition, option_loss, weight_total_variation=0):

	# -----------------------------------------------------------------------
	# Step 1: compute prediction
	# -----------------------------------------------------------------------
	if use_transform:
		if use_condition == 'none': # discriminator w/o condition
			data_fake_transform = transform_data(data_fake, transform)
			prediction_fake     = discriminator(data_fake_transform)
		elif 'estimate' in use_condition:
			transform_feature          = convert_transform_to_feature(transform)
			condition                  = transform_feature.repeat(data_fake.shape[0], 1).to(device)
			data_fake_transform = transform_data(data_fake, transform)
			prediction_fake, estimate_condition_fake = discriminator(data_fake_transform)
		else:
			transform_feature          = convert_transform_to_feature(transform)
			condition                  = transform_feature.repeat(data_fake.shape[0], 1).to(device)
			data_fake_transform        = transform_data(data_fake, transform)
			prediction_fake            = discriminator(data_fake_transform, condition)

			#print(transform)
			#print(transform_feature)
			#time.sleep(2)
	else:
			data_fake_transform = data_fake
			prediction_fake     = discriminator(data_fake_transform)



	# -----------------------------------------------------------------------
	# Step 2: compute loss
	# -----------------------------------------------------------------------	
	
	#if option_loss == 'cross_entropy':
	#
	#	criterion = nn.BCEWithLogitsLoss()
	#	
	#	label_real  = torch.ones_like(prediction_fake)
	#	loss_adv    = criterion(prediction_fake, label_real)


	if option_loss == 'cross_entropy':
		print('cross_entropy is not available.  Choose one of \'bce\' and \'bce_with_logits\'')
		exit(-1)

	if option_loss == 'bce':
		criterion   = nn.BCELoss()
		label_real  = torch.ones_like(prediction_fake)
		loss_adv    = criterion(prediction_fake, label_real)

	elif option_loss == 'bce_with_logits':
		criterion   = nn.BCEWithLogitsLoss()
		label_real  = torch.ones_like(prediction_fake)
		loss_adv    = criterion(prediction_fake, label_real)

	elif option_loss == 'wasserstein':
		# incomplete
		loss_adv    = torch.mean(prediction_fake)

	elif option_loss == 'l2':
		# incomplete
		criterion   = nn.MSELoss()
		sig         = nn.Sigmoid() 

		label_real  = torch.ones_like(prediction_fake)
		loss_adv    = criterion(sig(prediction_fake), label_real)
	else:
	
		loss_adv = None


	# -----------------------------------------------------------------------
	# Step 2.2: compute loss of estimated condition loss
	# -----------------------------------------------------------------------
	if 'estimate' in use_condition:

		estimate_option = use_condition.split('-')
		
		if len(estimate_option) == 1: coefficient_loss_condition = 1
		else:                         coefficient_loss_condition = float(estimate_option[-1])

		if option_loss == 'bce_with_logits':
			F = torch.nn.functional
			loss_condition_fake = torch.sum( F.binary_cross_entropy_with_logits(input=estimate_condition_fake, target=condition) )

			loss_discriminator  = loss_adv + coefficient_loss_condition * loss_condition_fake
		else:
			print('error: criterion for esimated condition loss is not implemented: ', option_loss)
			exit(-1)
		
	if weight_total_variation == 0:  
		loss_tv = 0.
	else:
		loss_tv = 0.5 * weight_total_variation * (
			torch.mean(torch.abs(data_fake[:, :, :, :-1] - data_fake[:, :, :, 1:])) + 
			torch.mean(torch.abs(data_fake[:, :, :-1, :] - data_fake[:, :, 1:, :])))
	
	loss_generator  = loss_adv + loss_tv

	return loss_generator







# ---------------------------------------------------------------------------
#  
# Example of real
#  
# ---------------------------------------------------------------------------
if show_real_images:

	if is_2d_image:
		if use_temporal_real: 
			batch_data_real = nonpermute_data_real[0:num_images_in_grid].clone()
		else:
			(batch_data_real, batch_label_real) = next(iter(dataloader_real))
			if config_dataset_real['normalize'].lower() == 'scale':
				value_min       = config_dataset_real['normalize_value1']
				value_max       = config_dataset_real['normalize_value2'] 
				batch_data_real = normalize_tensor(batch_data_real, value_min, value_max)
			else:
				batch_data_real = batch_data_real
			batch_data_real = batch_data_real[0:num_images_in_grid]
		draw_image_grid(batch_data_real, file_image_real)

	elif 'two_moons' :
		if use_temporal_real: 
			batch_two_moons = train_data_real.clone()
		else:
			batch_two_moons = examples_two_moons.clone()
		two_moons.plot_two_moons_v1(batch_two_moons[0:1000], file_image_real)

#save_image(batch_data_real, './result/figure/real/debug_real.png')
#save_image(batch_data_real, './result/figure/real/debug_real_normalize.png', normalize=True)

#print(torch.FloatTensor(batch_data_real).max(), torch.FloatTensor(batch_data_real).min()) # --> [1,-1] , celeba w/o scaling
#exit(-1)

# ---------------------------------------------------------------------------
#  
# Example of trasformed data
#  
# ---------------------------------------------------------------------------

# Given batch_data_real,
#if use_transform:
if show_real_images:

	if is_2d_image: 
		

		# Example) torch.Size([100, 1, 32, 32]) -> torch.Size([5, 20, 1, 32, 32])
		batch_data_real_reshape = batch_data_real.reshape(num_image_per_trans, -1 , batch_data_real.shape[-3],  batch_data_real.shape[-2], batch_data_real.shape[-1])


		for i,batch in enumerate(batch_data_real_reshape):
			(transform, dim_transform) = generate_transform(option_transform) # create a new transform
			batch                      = transform_data(batch, transform)
			batch_data_real_reshape[i] = batch.clone()
			#print(transform)

		batch_data_real_transform = batch_data_real_reshape.reshape(batch_data_real.shape)

		draw_image_grid(batch_data_real, file_image_transform)
		#save_image(batch_data_real, './result/figure/real/debug_transformed.png')
		#save_image(batch_data_real, './result/figure/real/debug_transformed_normalize.png', normalize=True)
	else:
		pass		

	print('Example of transformed real data --> ', file_image_transform)


	
# ---------------------------------------------------------------------------
#  
# Visualization of fake image process
#  
# ---------------------------------------------------------------------------
if show_fake_process:
	num_fake_image_process        = 4
	log_fake_process_epoch        = np.linspace(0,int(number_epoch)-1,num=11).astype(int)
	log_fake_process_image        = []
	log_fake_process_batch_latent = torch.randn(dim_batch_latent).to(device) # fixed latent vector




	
# ---------------------------------------------------------------------------
#  
# Multiple GPUs
#  
# ---------------------------------------------------------------------------
generator     = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

print('The main iteration started: ', file_id) 
# ---------------------------------------------------------------------------
#  
# main iterations (epoch)
#  
# ---------------------------------------------------------------------------

	

# predictions of training real, validation real, and fake data to compute the overfitting heuristics:
value_D_train = []
value_D_valid = []
value_D_fake  = []

value_rv_epoch = []
value_rt_epoch = []

for scale in range(number_scale):
		
	if use_scheduler:
		scheduler_discriminator.initialize()
		scheduler_generator.initialize()

	#batch_latent_test = torch.randn(dim_batch_latent).to(device)

	# ---------------------------------------------------------------------------
	#  
	# main iterations (epoch)
	#  
	# ---------------------------------------------------------------------------
	for epoch in range(number_epoch):

		if show_fake_process and (epoch in log_fake_process_epoch):
			
			if is_2d_image:
				log_batch_data_fake = generator( log_fake_process_batch_latent )
				log_fake_process_image.append( log_batch_data_fake.detach().clone()[0:num_fake_image_process] )
		
			elif 'two_moons' in name_dataset: 
				file_image_test_epoch  = path_figure + file_id + ',test,ep.{:04d}.png'.format(epoch+1) 
				test_data_fake  = generator(test_batch_latent)
				two_moons.plot_two_moons_v1(test_data_fake[0:1000], file_image_test_epoch)


		time_epoch = time.time()
		
		# step for the learning rate
		if use_scheduler:
			scheduler_discriminator.step()
			scheduler_generator.step()

		# for train_data_real
		index_batch_real        = np.random.permutation(index_data_epoch_real)
		index_batch_real_slice  = np.reshape(index_batch_real, [number_batch_real, size_batch]) 

		value_loss_batch                = []
		value_loss_batch_generator      = []
		value_loss_batch_discriminator  = []

		value_prediction_batch_real     = []
		value_prediction_batch_fake     = []
		value_prediction_batch_sum      = []

		value_D_epoch_train = []
		value_D_epoch_valid = []
		value_D_epoch_fake  = []


		# ---------------------------------------------------------------------------
		#  
		# mini-batch iterations for real data
		#  
		# ---------------------------------------------------------------------------
		#for i in tqdm(range(number_batch_real)):
		#for i, (batch_data_real, batch_label_real) in enumerate(dataloader_real):
		# for train_data_real
		for i in range(number_batch_real):

			# ---------------------------------------------------------------------------
			#  
			# determine the transformation
			#  
			# ---------------------------------------------------------------------------
			(transform, dim_transform) = generate_transform(option_transform)


			# ---------------------------------------------------------------------------
			#  
			# get real data
			#
			#    batch_data_real:  a batch of real data for training
			#    batch_data_valid: a batch of real data for validation
			# ---------------------------------------------------------------------------
			# sample a training real data
			batch_data_real = train_data_real[index_batch_real_slice[i]]
			batch_data_real = batch_data_real.to(device)


			# sample a validation real data 
			if index_j_epoch_valid >= number_batch_valid: # for valid_data_real
				index_batch_valid        = np.random.permutation(index_data_epoch_valid)
				index_batch_valid_slice  = np.reshape(index_batch_valid, [number_batch_valid, size_batch]) 
				index_j_epoch_valid      = 0

				# incomplete: last swap is not implemented yet.

			j                = index_j_epoch_valid 
			batch_data_valid = valid_data_real[index_batch_valid_slice[j]]
			batch_data_valid = batch_data_valid.to(device)
			index_j_epoch_valid += 1 # next validation epoch
			

			# ---------------------------------------------------------------------------
			#  
			# update the generator
			#  
			# ---------------------------------------------------------------------------
			#generator.train()
			#discriminator.eval()
			optimizer_generator.zero_grad()

			batch_latent    = torch.randn(dim_batch_latent).to(device)
			batch_data_fake = generator(batch_latent)

			loss_generator  = compute_loss_generator(discriminator, batch_data_fake, transform, use_condition, option_loss_generator, weight_total_variation)
			#loss_generator  = compute_loss_generator(discriminator, batch_data_fake, option_transform, use_condition, use_transform, option_loss_generator, weight_total_variation)

			loss_generator.backward()
			if epoch > 0: optimizer_generator.step()

			# ---------------------------------------------------------------------------
			#  
			# update the discriminator
			#  
			# ---------------------------------------------------------------------------
			#generator.eval()
			#discriminator.train()
			optimizer_discriminator.zero_grad()

			#batch_data_fake = batch_data_fake.detach() 
			batch_latent    = torch.randn(dim_batch_latent).to(device)
			batch_data_fake = generator(batch_latent)
			
			(loss_discriminator, prediction_batch_real, prediction_batch_fake) = compute_loss_discriminator(discriminator, batch_data_fake, batch_data_real, transform, use_condition, option_loss_discriminator, gradient_regularization)
			#(loss_discriminator, prediction_batch_real, prediction_batch_fake) = compute_loss_discriminator(discriminator, batch_data_fake, batch_data_real, option_transform, use_condition, use_transform, option_loss_discriminator, gradient_regularization)
		

			loss_discriminator.backward()
			if epoch > 0: optimizer_discriminator.step()
			

			# ---------------------------------------------------------------------------
			#  
			# validation
			#  
			# ---------------------------------------------------------------------------
			with torch.no_grad():
				prediction_batch_valid = compute_prediction(discriminator, batch_data_valid, transform, use_condition)

			#print(prediction_batch_valid.shape)


			# ---------------------------------------------------------------------------
			#  
			# save loss and prediction for each real data batch 
			#  
			# ---------------------------------------------------------------------------


			value_loss_batch_discriminator.append( loss_discriminator.item() )
			value_loss_batch_generator.append( loss_generator.item() )
			value_prediction_batch_real.append( prediction_batch_real.squeeze().detach().cpu().numpy() )
			value_prediction_batch_fake.append( prediction_batch_fake.squeeze().detach().cpu().numpy() )

			value_D_train.append( prediction_batch_real.squeeze().detach().cpu().numpy() )
			value_D_valid.append( prediction_batch_valid.squeeze().detach().cpu().numpy() )
			value_D_fake.append(  prediction_batch_fake.squeeze().detach().cpu().numpy() )


			value_D_epoch_train.append( prediction_batch_real.squeeze().detach().cpu().numpy() )
			value_D_epoch_valid.append( prediction_batch_valid.squeeze().detach().cpu().numpy() )
			value_D_epoch_fake.append(  prediction_batch_fake.squeeze().detach().cpu().numpy() )


			# nvidia-ada
			if option_transform['Ada']:
				if len(value_prediction_batch_real) > 0:
					rt = np.sign( np.array(value_prediction_batch_real) ).flatten() # a heuristic of Equation (1) in the Stylegan2-Ada paper
					update_global_P_nvidia_ada(rt)


		# ---------------------------------------------------------------------------
		#  
		# include the index for the last drop batch in the next iteration
		#  
		# ---------------------------------------------------------------------------
		# for train_data_real
		swap_last = index_data_epoch_real[0:number_data_epoch_real_last].copy()
		index_data_epoch_real[0:number_data_epoch_real_last] = index_data_epoch_real_last
		index_data_epoch_real_last = swap_last.copy()

		# ---------------------------------------------------------------------------
		#  
		# Online evaluation of the generator
		#  
		# ---------------------------------------------------------------------------

		#batch_data_fake_test = generator(batch_latent_test)
		test_data_fake  = generator(test_batch_latent)


		# L1 distance between the present test fakes with the past test fakes
		if epoch > 0:
			with torch.no_grad():
				# L1 norm: data-wise mean
				L1_norm = torch.abs( test_data_fake  - past_test_data_fake)
				L1_norm = L1_norm.reshape(L1_norm.shape[0],-1).mean(dim=1)
				# mean and std of L1 norm
				value_test_data_fake_L1_mean.append(torch.mean( L1_norm ).item())
				value_test_data_fake_L1_std.append(torch.std( L1_norm ).item())
 
		past_test_data_fake = test_data_fake.clone()
	


		# FID
		FID_test = None
		if compute_FID:
			if (epoch % interval_compute_FID == 0) or (epoch == number_epoch-1):

				print('computing FID...')

				time_eval = time.time()

				# sample a subset of the training set and fake set
				test_data_real = train_data_real[np.random.permutation(len(train_data_real))][:num_test_data]
				test_data_fake = generator(test_batch_latent)  
				FID_test = fid.calculate_fid_given_batches(test_data_real,   test_data_fake, batch_size=batch_size_FID)
				value_test_data_FID.append( FID_test )
				if scale == number_scale-1 and epoch == number_epoch-1:
					final_FID = FID_test

				time_eval = time.time() - time_eval

				print('Computed FID of test data {}: {:.2f} sec'.format(FID_test, time_eval))
		# Find best model
		is_best_model = False

		if is_2d_image and FID_test is not None:
			if FID_test < best_FID:
				best_FID      = FID_test
				best_FID_epoch = epoch + 1
				is_best_model = True
				print('Updated the best model with test FID {}: {}'.format(FID_test, file_id))
		elif not (is_2d_image and compute_FID): # If error metrics are disabled,
			if epoch == number_epoch-1:
				is_best_model = True
				print('We use the final model as the best model, when metrics are diabled.')

		if is_best_model:
			torch.save(generator.state_dict(), file_generator_best)

			# save images
			if is_2d_image:
				draw_image_grid(test_data_fake[0:num_images_in_grid], file_image_best)	
			elif 'two_moons' in name_dataset: 
				two_moons.plot_two_moons_v1(test_data_fake[0:1000], file_image_best)


		# ---------------------------------------------------------------------------
		#  
		# save results 
		#  
		# ---------------------------------------------------------------------------
		value_loss_batch_discriminator       = np.array( value_loss_batch_discriminator )
		value_loss_batch_generator           = np.array( value_loss_batch_generator )
		value_prediction_batch_real          = np.array( value_prediction_batch_real )
		value_prediction_batch_fake          = np.array( value_prediction_batch_fake )


		value_loss_discriminator_mean[epoch] = np.mean( value_loss_batch_discriminator )
		value_loss_discriminator_std[epoch]  = np.std(  value_loss_batch_discriminator )
		value_loss_generator_mean[epoch]     = np.mean( value_loss_batch_generator )
		value_loss_generator_std[epoch]      = np.std(  value_loss_batch_generator )
		value_loss_total_mean[epoch]         = (value_loss_discriminator_mean[epoch] + value_loss_generator_mean[epoch]) * 0.5 
		
		value_prediction_real_mean[epoch]    = np.mean( value_prediction_batch_real )
		value_prediction_real_std[epoch]     = np.std(  value_prediction_batch_real )
		value_prediction_fake_mean[epoch]    = np.mean( value_prediction_batch_fake )
		value_prediction_fake_std[epoch]     = np.std(  value_prediction_batch_fake )
		value_prediction_sum[epoch]          = (value_prediction_real_mean[epoch] + value_prediction_fake_mean[epoch]) * 0.5


		if (epoch == number_epoch-1):

			# visualize fakes
			if is_2d_image: 
				draw_image_grid(batch_data_fake[0:num_images_in_grid], file_image_train)
				draw_image_grid(test_data_fake[0:num_images_in_grid],  file_image_test)
			else:
				if 'two_moons' in name_dataset: 
					two_moons.plot_two_moons_v1(batch_data_fake[0:1000], file_image_train)
					two_moons.plot_two_moons_v1(test_data_fake[0:1000],  file_image_test)

			# plot curves
			plot_curve_errorbar([value_loss_generator_mean, value_loss_discriminator_mean, value_loss_total_mean], [value_loss_generator_std, value_loss_discriminator_std, np.zeros_like(value_loss_generator_std)], labels=['generator', 'discriminator', 'average'], filename=file_figure_loss)
			plot_curve_errorbar([value_prediction_real_mean, value_prediction_fake_mean, value_prediction_sum],    [value_prediction_real_std, value_prediction_fake_std, np.zeros_like(value_prediction_real_std)],   labels=['real', 'fake', 'sum'], filename=file_figure_prediction)
			plot_curve_errorbar(np.array(value_test_data_fake_L1_mean), np.array(value_test_data_fake_L1_std), labels='L1 norm btw test(t) with (t-1)', filename=file_figure_test_L1)
			plot_curve_errorbar([np.array(value_test_data_FID)], \
					    None, \
					    labels=['FID with test set'], \
					    x_intervals=None, ylim=ylim_FID, grid=True, filename=file_figure_FID)

			# store curve data
			np.savez(file_plot_loss,        means  = [value_loss_generator_mean, value_loss_discriminator_mean, value_loss_total_mean], \
						        stds   = [value_loss_generator_std, value_loss_discriminator_std, np.zeros_like(value_loss_generator_std)], \
						        labels = ['generator', 'discriminator', 'average'])

			np.savez(file_plot_prediction , means  = [value_prediction_real_mean, value_prediction_fake_mean, value_prediction_sum],\
						        stds   = [value_prediction_real_std, value_prediction_fake_std, np.zeros_like(value_prediction_real_std)],\
							labels = ['real', 'fake', 'sum'])

			np.savez(file_plot_FID,  means      = [np.array(value_test_data_FID)],\
						 stds       = None, \
						 labels=['FID with test set'],\
						 x_interval = interval_compute_FID, ylim=ylim_FID, grid=True)

	
		time_epoch = time.time() - time_epoch

		message = '(%1d/%1d) [%04d/%04d] %10.2f sec, (D) %10.7f, (G) %10.7f, (D+G) %10.7f %s' % (scale, number_scale, epoch, number_epoch, time_epoch, value_loss_discriminator_mean[epoch], value_loss_generator_mean[epoch], value_loss_total_mean[epoch], file_id)

		print(message)




	# ---------------------------------------------------------------------------
	#  
	# update scale in the scale-space 
	#  
	# --------------------------------------------------------------------------- 
	if scale < number_scale - 1: 
	
		size_fake['height'] = size_fake['height'] * 2
		size_fake['idth']   = size_fake['width'] * 2

		interp_size         = [0, 0]
		interp_size[0]      = size_fake['height']
		interp_size[1]      = size_fake['width']

		dataset_fake        = nn.Upsample(size=interp_size, mode='bilinear', align_corners=False)(dataset_fake)
		dataset_fake_test   = nn.Upsample(size=interp_size, mode='bilinear', align_corners=False)(dataset_fake_test)

		print('size fake: height: ', size_fake['height'])
		print('size fake: width: ', size_fake['width'])



#
#
# Post process
#
#


# save the trained generator
# Load: https://pytorch.org/tutorials/beginner/saving_loading_models.html
torch.save(generator.state_dict(), file_generator_final)

# The final-epoch FID was already computed during training.
best_FID_text = '{:.6f}'.format(best_FID) if best_FID_epoch is not None else 'N/A'
final_FID_text = '{:.6f}'.format(final_FID) if final_FID is not None else 'N/A'
fid_summary = ('BestFID: {} (epoch {})\nFinalFID: {} (epoch {})\n'
               'BestGenerator: {}\nFinalGenerator: {}\n').format(
                   best_FID_text, best_FID_epoch if best_FID_epoch is not None else 'N/A',
                   final_FID_text, number_epoch, file_generator_best, file_generator_final)
print(fid_summary, end='')
with open(file_log, 'a') as log_file:
	log_file.write(fid_summary)

def draw_image_sequence(log_sequential, filename):
	E=len(log_sequential)
	B,C,H,W=log_sequential[0].shape

	tmp = torch.cat(log_sequential).view(E,B,C,H,W)
	tmp = torch.transpose(tmp,0,1)
	tmp = tmp.reshape(-1,C,H,W)

	if size_real['channel']==1: normalize=False
	else:                       normalize=True

	save_image(tmp, filename, nrow=E, normalize=normalize)


if show_fake_process:
	if is_2d_image: draw_image_sequence(log_fake_process_image, file_fake_process)



time_elapsed = time.time() - time_elapsed
print('{:.2f} sec'.format(time_elapsed))
txt_file = open('./result/runtime.txt', 'at')
txt_file.write('{}, {}, seconds\n'.format(file_id, time_elapsed))
txt_file.close()


with open(file_option, 'w') as configfile:   # save
	config.write(configfile)

time.sleep(10)
print('done, ', file_id)
