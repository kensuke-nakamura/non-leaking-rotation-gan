

#
'''
# installation
pip3 install prdc
pip3 install install pytorch-fid-wrapper; pip3 install git+https://github.com/evenmn/pytorch-sfid
'''
#


import torch
import numpy
import sys 
import os
import matplotlib.pyplot as plt
import glob
import fnmatch
import time
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler as SubsetRandomSampler
from torch.utils.data.sampler import RandomSampler	 as RandomSampler

#
# utility
#
from option import read_config 
dir_utility = '../../../utility/'
sys.path.append(dir_utility)
from torchvision.datasets import * # Training dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from utility_visualize.visualize import *

# ======================================================================
# public evaluation metrics
# ======================================================================



from utility_evaluation.fidis import fid_score as fid 
from utility_evaluation.fidis import inception_score as inception_score
from utility_network          import DCGAN2_Generator,   DCGAN2_Discriminator,   DCGAN2_Discriminator_Classifier_Condition
from utility_network          import DCGAN2_Discriminator_Estimate_Condition
from utility_network          import DCGAN2_Generator32, DCGAN2_Discriminator32, DCGAN2_Discriminator32_Classifier_Condition, DCGAN2_Discriminator_Conv2_Condition
from utility_network          import DCGAN2_Discriminator32_Estimate_Condition
from utility_network          import GAN_Generator, GAN_Discriminator, GAN_Discriminator_Condition

# ======================================================================
# cuda
# ======================================================================
dir_temporal_real = os.path.join(os.environ.get('DCGAN_TEMPORAL_REAL_DIR', '/SSD/dataset/temporal_data'), '')
root_dataset = os.path.join(os.environ.get('DCGAN_DATASET_ROOT', '/hdd1/dataset'), '')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
	torch.cuda.set_device(device)

print(device)

enable_prdc       = False
use_temporal_real = True



def evaluate_trained_generator(file_id, final_or_best='best', \
num_data_evaluation=10000, normalize_real_data=True, \
batch_size=20,
root_dataset=root_dataset,path_option='./result/option/', path_model='./result/model/', path_evaluation='./result/evaluation/',
feature_extractor=None
):

	time_elapsed = time.time()

	is_2d_image = True # automatically False if One-demensional data is called.


	# input file 
	file_option = path_option + file_id + '.ini'
	file_model  = path_model  + file_id + ',generator,' + final_or_best + '.pt'

	# output file
	file_evaluation_image_test = path_evaluation + file_id + '.png'


	'''
	# When the training process was terminated, the model file does not exist.
	# We skip such results.
	if not os.path.isfile(file_model):
		#print('The file does not exist: {}'.format(file_model))
		return ''
	'''

	if os.path.exists(file_option) == False:
		print('file_option does not exist:', file_option)
		return

	# ======================================================================
	# Step 1: Read options and create a generator
	#
	#   *CAUTION* this part should be exactly the same with main.py
	# ======================================================================

	# options
	config_dataset_real,             \
	config_dataset_fake,             \
	config_transform,                \
	config_network,                  \
	config_optimize,                 \
	config_optimize_generator,       \
	config_optimize_discriminator,   \
	config                           = read_config(file_option)

	# size of the data
	size_real       = {'channel' : config_dataset_real['channel'], 'height' : config_dataset_real['height'], 'width' : config_dataset_real['width']}
	size_fake       = {'channel' : config_dataset_fake['channel'], 'height' : config_dataset_fake['height'], 'width' : config_dataset_fake['width']}
	length_latent   = config_dataset_fake['length_latent']

	# neural network model
	dim_feature              = 32
	dim_generator_input      = [length_latent, 1, 1]
	dim_generator_output     = [size_fake['channel'], size_fake['height'], size_fake['width']]

	# iteration for the mini-batch of real data
	size_batch                = config_optimize['size_batch'] 
	dim_batch_latent          = [size_batch, length_latent, 1, 1]

	# ======================================================================
	# neural networks
	# ======================================================================
	# Generator
	if config_network['generator'] == 'DCGAN2_Generator':   generator = DCGAN2_Generator(length_latent, size_real['channel']).to(device)
	elif config_network['generator'] == 'DCGAN2_Generator32': generator = DCGAN2_Generator32(length_latent, size_real['channel']).to(device)
	elif config_network['generator'] == 'GAN_Generator':      generator = GAN_Generator(length_latent, dim_generator_output).to(device)
	else:                                                  generator = Network_Generator(dim_generator_input, dim_generator_output, dim_feature, config_network['generator']).to(device)


	# ======================================================================
	# Step 2: Load parameters of generator
	# ======================================================================
	try:
		generator.load_state_dict( torch.load(file_model, map_location=device) )
	except:
		# multiple GPUs, parallel
		generator = nn.DataParallel(generator)
		generator.load_state_dict( torch.load(file_model, map_location=device) )
	generator.eval()


	# ======================================================================
	# Step 3: Call real dataloader
	#
	#   *CAUTION* This part can be DIFFERENT with the main.py
	# ======================================================================

	# dataset
	name_dataset    = config_dataset_real['name'].lower() 
	path_dataset    = root_dataset + name_dataset

	transform_list = []
	transform_list.append(transforms.ToTensor())
	transform_list.append(transforms.Resize((size_real['height'], size_real['width'])))


	def split_digit(dataset, labels):
		labels = [int(label) for label in labels]  # string to interger '3' -> 3
		indices = [True if target in labels else False for target in dataset.targets]
		dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices]
		return dataset


	if 'MNIST'.lower() in name_dataset:
		transform_compose   = transforms.Compose(transform_list)
		dataset_real        = MNIST(root_dataset, download=False, train=True, transform=transform_compose)
		# subset training:  'MNIST-3-6' -> 3 and 6 digits in MNIST
		subset_labels = name_dataset.split('-')[1:]     
		if len(subset_labels) > 0: dataset_real = split_digit(dataset_real, subset_labels)
	elif 'cifar10.class'.lower() in name_dataset: # cifar10.class.0
		target_class  = int( name_dataset.split('.')[-1] )

		transform = transforms.Compose([
			transforms.Resize(size_real['width']),
			transforms.CenterCrop(size_real['width']),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
		])
		dataset_real = CIFAR10(root_dataset+'/CIFAR', train=True,  download=True, transform=transform)

		print('We use images in CIFAR-10 with the label {}'.format(target_class))

		# https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276
		dataset_real.targets = torch.tensor(dataset_real.targets)
		idx = dataset_real.targets==target_class
		dataset_real.data    = dataset_real.data[idx.numpy().astype(np.bool)]
		dataset_real.targets = dataset_real.targets[idx.numpy().astype(np.bool)]
	elif 'cifar10'.lower() in name_dataset:
		transform = transforms.Compose([
			transforms.Resize(size_real['width']),
			transforms.CenterCrop(size_real['width']),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
		])
		dataset_real = CIFAR10(path_dataset, train=True,  download=True, transform=transform)
	elif 'CelebA'.lower() in name_dataset:
		#transform_list.append(transforms.CenterCrop((size_real['height'], size_real['width'])))
		transform_list = []
		transform_list.append( transforms.Resize(size_real['width']) )
		transform_list.append( transforms.CenterCrop(size_real['height']) )
		transform_list.append( transforms.ToTensor() )
		transform_list.append( transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) )
		transform_compose   = transforms.Compose(transform_list)
		#dataset_real        = CelebA(path_dataset, download=False, transform=transform_compose, split='train')
		dataset_real         = ImageFolder(path_dataset, transform=transform_compose)

	elif name_dataset == 'oxflower':
		#print('{} is called.  A subset of the digits can be used by \'MNIST-1-3-6\' for example.'.format(name_dataset))
		transform_list = []
		transform_list.append( transforms.ToTensor() )
		transform_list.append( transforms.Resize(size_real['height']) )
		transform_list.append( transforms.CenterCrop(size_real['height']) )
		transform_list.append( transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) )

		transform_compose   = transforms.Compose(transform_list)
		dataset_real        = ImageFolder(path_dataset, transform=transform_compose)

	elif 'flowers102' in name_dataset:
		transform_list = []
		transform_list.append( transforms.ToTensor() )
		transform_list.append( transforms.Resize(size_real['height']) )
		transform_list.append( transforms.CenterCrop(size_real['height']) )
		transform_list.append( transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) )
		transform_compose   = transforms.Compose(transform_list)
		dataset_real        = Flowers102(path_dataset, split='train',  download=True, transform=transform_compose)

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


	else:
		pass


	size_data_real  = config_dataset_real['size_data'] # 'all' or number of data

	if size_data_real.lower() == 'all' or size_data_real.lower() == '1':
		num_size_data_real   = int( len(dataset_real ) )
		sampler=RandomSampler(range(len(dataset_real)))
	else:
		print('1/({}) subset of data are used.'.format(size_data_real))
		num_size_data_real   = int( len(dataset_real ) / float(size_data_real) )
		subset_indices   = torch.randperm(len(dataset_real))[:num_size_data_real]
		sampler=SubsetRandomSampler(subset_indices)

	# real dataset 
	dataloader_real         = DataLoader(dataset=dataset_real, batch_size=1, sampler=sampler, drop_last=True, num_workers=2, pin_memory=True)

	#dataloader_real         = DataLoader(dataset=dataset_real, batch_size=1, drop_last=False, shuffle=True)

	number_data_real        = len(dataset_real)
	dim_dataset_real        = [number_data_real, size_real['channel'], size_real['height'], size_real['width']]
	dataset_real            = None

	print(number_data_real)


	#file_temporal_real      = dir_temporal_real + name_dataset + ',h,' + str(size_real['height']) + ',w,' + str(size_real['width']) + ',' + str(config_dataset_real['normalize'].lower()) + '.pt'
	file_temporal_real      = dir_temporal_real + name_dataset + ',M,' + size_data_real + ',h,' + str(size_real['height']) + ',w,' + str(size_real['width']) + ',' + str(config_dataset_real['normalize'].lower()) + '.pt'

	print(file_temporal_real)

	on_memory_dataset_real = []
	if use_temporal_real and os.path.exists(file_temporal_real):
		print('Loading on-memory real data from ', file_temporal_real)
		dataset_real = torch.load( file_temporal_real )
		print('done')

	else:
		print('on-memory data {} was not found.'.format(file_temporal_real))
		print('preparing data.. this can take mininutes')
		dataset_real = torch.zeros(dim_dataset_real)
		for i, (batch_data_real, batch_label_real) in enumerate(dataloader_real):
			value_min       = config_dataset_real['normalize_value1']
			value_max       = config_dataset_real['normalize_value2'] 
			if normalize_real_data:  batch_data_real = normalize_tensor(batch_data_real, value_min, value_max)
			dataset_real[i] = batch_data_real

	print('data loaded')

	# ======================================================================
	# Step 4: Given the generator with the real dataset, do evaluation
	# ======================================================================

	if num_data_evaluation > len(dataset_real):
		previous            = num_data_evaluation
		num_data_evaluation = int( batch_size * np.floor(len(dataset_real) / batch_size) )
		print('warning: num_data_evaluation = {} exceeds the dataset size {}, and is reduced to {} based on the batchsize'.format(previous, len(dataset_real), num_data_evaluation, batch_size))


	# real data batch
	batch_data_real     = dataset_real[0:num_data_evaluation]

	# fake data batch
	with torch.no_grad():
		batch_data_fake = torch.cat([
			generator(torch.randn((min(batch_size, num_data_evaluation - start), length_latent, 1, 1), device=device)).cpu()
			for start in range(0, num_data_evaluation, batch_size)
		], dim=0)

	evaluation_results = file_id + '\t'
	
	if torch.any(torch.isnan(batch_data_fake)):
		print('Invalid fakes')
		return evaluation_results
		
	
	# ------------------------------------------------------------------------------------------------------
	# Step1: Given real, fake images, with a pretrained network, we compute features of the data
	# https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/5ad4629b07f3f3a51184c39d3dbe9085a60e264c/improved_precision_recall.py
	# ------------------------------------------------------------------------------------------------------
	if enable_prdc and feature_extractor is None:
		print('loading vgg16 for improved precision and recall...', end='', flush=True)
		feature_extractor = models.vgg16(pretrained=True).eval().to(device)

	if is_2d_image and enable_prdc:
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

		print('image data is converted to (3,224,224) to compute precision and recall')
		resized_batch_real = data2batch_to_compute_prec_rec(batch_data_real, batch_size=batch_size)	
		resized_batch_fake = data2batch_to_compute_prec_rec(batch_data_fake, batch_size=batch_size)

		print('converting resized data to feature...')
		# real
		real_features = []
		for i in range(len(resized_batch_real)):
			batch    = resized_batch_real[i].squeeze(0)
			features = feature_extractor.features(batch.to(device)).view(-1, 7 * 7 * 512)
			features = feature_extractor.classifier[:4](features)
			real_features.append( features.detach().cpu() )
		real_features = torch.cat( real_features, dim=0 ).numpy()	
		# fake
		fake_features = []
		for i in range(len(resized_batch_fake)):
			batch    = resized_batch_fake[i].squeeze(0)
			features = feature_extractor.features(batch.to(device)).view(-1, 7 * 7 * 512)
			features = feature_extractor.classifier[:4](features)
			fake_features.append( features.detach().cpu() )
		fake_features = torch.cat( fake_features, dim=0 ).numpy()
		print('done')
	elif enable_prdc:
		# 1d data
		real_features = batch_data_real.reshape(batch_data_real.shape[0],-1).detach().cpu().numpy()
		fake_features = batch_data_fake.reshape(batch_data_fake.shape[0],-1).detach().cpu().numpy()

	# ------------------------------------------------------------------------------------------------------
	# Step2: Given features, we compute precision, recall, and others
	# Reliable Fidelity and Diversity Metrics for Generative Models (ICML 2020)
	# https://github.com/clovaai/generative-evaluation-prdc
	# ------------------------------------------------------------------------------------------------------
	if enable_prdc:
		from prdc import compute_prdc
		for nearest_k in [3, 5, 10]:
			metrics = compute_prdc(real_features=real_features, fake_features=fake_features, nearest_k=nearest_k)
					
			result = 'precision_recall_density_coverage_k{}\t'.format(nearest_k)
			result = result + '{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t'.format(metrics['precision'], metrics['recall'], metrics['density'], metrics['coverage'])

			print(result)

			evaluation_results += result

	

	# ------------------------------------------------------------------------------------------------------
	# Given real and fake data, we compute FID and IS
	# ------------------------------------------------------------------------------------------------------
	if is_2d_image:
		FID        = fid.calculate_fid_given_batches(batch_data_real, batch_data_fake, batch_size=batch_size)
		IS, IS_std = inception_score(batch_data_fake, cuda=True, batch_size=batch_size, resize=True, splits=100)
		print('FID, IS:',FID,IS)
	else:
		FID	   = 99
		IS, IS_std = 0, 0.0001
	evaluation_results += 'FID\t{:.4f}\tIS\t{:.4f}\t'.format(FID,IS)


	time_elapsed = time.time() - time_elapsed
	print('{:.2f} sec'.format(time_elapsed))


	return evaluation_results
	







def visualize_fakes_using_trained_generator(file_id, final_or_best='best', \
num_data_evaluation=25, path_option='./result/option/', path_model='./result/model/'):


	# input file 
	file_option = path_option + file_id + '.ini'
	file_model  = path_model  + file_id + ',generator,' + final_or_best + '.pt'

	# output file
	file_evaluation_image_test = path_evaluation + file_id + '.png'


	# ======================================================================
	# Step 1: Read options and create a generator
	#
	#   *CAUTION* this part should be exactly the same with main.py
	# ======================================================================

	if os.path.exists(file_option) == False:
		print('file_option does not exist:', file_option)
		return


	# options
	config_dataset_real,             \
	config_dataset_fake,             \
	config_transform,                \
	config_network,                  \
	config_optimize,                 \
	config_optimize_generator,       \
	config_optimize_discriminator,   \
	config                           = read_config(file_option)

	# size of the data
	size_real       = {'channel' : config_dataset_real['channel'], 'height' : config_dataset_real['height'], 'width' : config_dataset_real['width']}
	size_fake       = {'channel' : config_dataset_fake['channel'], 'height' : config_dataset_fake['height'], 'width' : config_dataset_fake['width']}
	length_latent   = config_dataset_fake['length_latent']

	# neural network model
	dim_feature              = 32
	dim_generator_input      = [length_latent, 1, 1]
	dim_generator_output     = [size_fake['channel'], size_fake['height'], size_fake['width']]

	# iteration for the mini-batch of real data
	size_batch                = config_optimize['size_batch'] 
	dim_batch_latent          = [size_batch, length_latent, 1, 1]

	# neural networks 
	# Generator
	if config_network['generator'] == 'DCGAN2_Generator':   generator = DCGAN2_Generator(length_latent, size_real['channel']).to(device)
	elif config_network['generator'] == 'DCGAN2_Generator32': generator = DCGAN2_Generator32(length_latent, size_real['channel']).to(device)
	elif config_network['generator'] == 'GAN_Generator':      generator = GAN_Generator(length_latent, dim_generator_output).to(device)
	else:                                                  generator = Network_Generator(dim_generator_input, dim_generator_output, dim_feature, config_network['generator']).to(device)

	# ======================================================================
	# Step 2: Load parameters of generator
	# ======================================================================
	try:
		generator.load_state_dict( torch.load(file_model, map_location=device) )
	except:
		# multiple GPUs, parallel
		generator = nn.DataParallel(generator)
		generator.load_state_dict( torch.load(file_model, map_location=device) )

	# fake data batch
	generator.eval()
	with torch.no_grad():
		batch_latent = torch.randn((128, length_latent, 1, 1)).to(device)
		batch_data_fake = generator(batch_latent)

	# visualization
	#figure_image  = plt.figure()
	#plot_image_grid(figure_image, batch_data_fake, file_evaluation_image_test)
	#plt.close(figure_image)
	draw_image_grid(batch_data_fake[0:num_data_evaluation], file_evaluation_image_test)
	
	print('  fake images are visualized in ', file_evaluation_image_test)

	return 
	




import argparse
parser = argparse.ArgumentParser(description='learning curves')
parser.add_argument('--I', type=int, default=1, help='number of units in parallel comput')
parser.add_argument('--i', type=int, default=0, help='thread id in parallel comput')
parser.add_argument('--pattern', default='*', help='evaluate only matching run identifiers')
parser.add_argument('--checkpoint', choices=['best', 'final'], default='best')
parser.add_argument('--overwrite', action='store_true', help='recompute existing results')
args = parser.parse_args()
i=args.i
I=args.I

#file_id = 'default'
#valuate_trained_generator(file_id)
if __name__=='__main__':

	# precision and recall
	if enable_prdc:
		print('loading vgg16 for improved precision and recall...', end='', flush=True)
		vgg16 = models.vgg16(pretrained=True).eval().to(device)
	else:
		vgg16 = None

	final_or_best = args.checkpoint
	over_write    = args.overwrite
	
	# path to evaluation results of each model
	path_evaluation        = './result/evaluation/'
	path_evaluation_txt = path_evaluation + 'txt/'

	os.makedirs(path_evaluation, exist_ok=True)
	os.makedirs(path_evaluation_txt, exist_ok=True)

	# final result
	file_evaluation_results = path_evaluation + 'error_metrics.txt'

	# ---------------------------------------------------------------------------
	# Step 1: 
	#     Given *.pt files, compute evaluation metrics of each file 
	#     and store the evaluation result in ./evaluation/txt/
	# ---------------------------------------------------------------------------

	# We will evaluate the 'best' generator but seek 'final' model first,
	# to avoid evaluating temporal result of which training is not ended.
	arr_txt = glob.glob('./result/model/*,generator,final.pt') 

	arr_txt = arr_txt[i::I] # parallel computation
	print('{} of generators are found.'.format(len(arr_txt)))
		
	for idx,txt in enumerate(arr_txt):
		file_id = os.path.splitext( os.path.basename(txt) )[0]
		file_id = file_id.replace(',generator,final','') 
		if not fnmatch.fnmatch(file_id, args.pattern):
			continue
		print('[{}/{}] {}'.format(idx+1,len(arr_txt),file_id))

		# Numerical evaluation of each file
		file_evaluation_txt = path_evaluation_txt + file_id + (',final' if final_or_best == 'final' else '') + '.txt'
		if os.path.isfile(file_evaluation_txt) and over_write == False:
			print('  we skip evaluated model.')
		else:
			result  = evaluate_trained_generator(file_id=file_id, final_or_best=final_or_best,normalize_real_data=False, path_evaluation=path_evaluation, feature_extractor=vgg16)

			txt_file_output = open(file_evaluation_txt, 'wt')
			txt_file_output.write('{}\n'.format(result))
			txt_file_output.close()

		# visualization
		visualize_fakes_using_trained_generator(file_id=file_id, final_or_best=final_or_best,num_data_evaluation=25)


	# ---------------------------------------------------------------------------
	# Step 2: 
	#     Given evaluation results in ./evaluation/txt/, 
	#     concanate them into ./ev
	# ---------------------------------------------------------------------------
	print('Concanate evaluated results > ', file_evaluation_results)

	with open(file_evaluation_results, 'w') as outfile:
		arr_txt = glob.glob(path_evaluation_txt+'*.txt')
		for idx,txt in enumerate(arr_txt):
			result_name = os.path.basename(txt)
			if final_or_best == 'best' and result_name.endswith(',final.txt'):
				continue
			if final_or_best == 'final' and not result_name.endswith(',final.txt'):
				continue
			result_id = result_name[:-len(',final.txt')] if final_or_best == 'final' else result_name[:-len('.txt')]
			if not fnmatch.fnmatch(result_id, args.pattern):
				continue
			with open(txt) as infile:
				for line in infile:
					outfile.write(line)

	print('done')
