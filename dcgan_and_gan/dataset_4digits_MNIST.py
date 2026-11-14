
import os
import shutil
import torch
import numpy as np
from torch import nn
from torchvision.datasets import * # Training dataset
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image








# get real data with the specific labels
def split_digit(dataset, labels):
	labels = [int(label) for label in labels]  # string to interger '3' -> 3
	indices = [True if target in labels else False for target in dataset.targets]
	dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices]
	return dataset


def make_images_with_four_digits(root_dataset='/hdd1/dataset/', digits=[1,2,3,4], nexample=10000, datasetname='4digits', reflesh=True):

	print('make images with four digits')

	if len(digits) != 4:
		print('error: make_images_with_four_digits(): digits should be four, given ', digits)
		exit(-1)

	dir_output = root_dataset + datasetname + '/img_align_{}/'.format(datasetname)
	
	if reflesh and os.path.isdir(dir_output): 
		print('{} already exist. We remove it and make a new dataset.'.format(dir_output))
		shutil.rmtree( dir_output )

	os.makedirs(dir_output, exist_ok=True)
	transform = transforms.Compose( [transforms.ToTensor(),transforms.Resize(32)] )


	# ----------------------------------------------------------------------------------------
	#
	# Given 'digits', load dataloaders.
	#                   
	# ----------------------------------------------------------------------------------------
	dataloaders=[]
	for digit in digits:
		dataset = MNIST(root_dataset, download=False, train=True, transform=transform)
		subset = split_digit(dataset, [digit])
		dataloader = DataLoader(dataset=subset, batch_size=1, drop_last=True, num_workers=2, pin_memory=True)
		dataloaders.append(dataloader)
		print('{} of {}-digit images are loaded.'.format(len(dataloader),digit))
		
	# ----------------------------------------------------------------------------------------
	#
	# Given dataloaders, construct an array of images with [num of digits][num of examples]
	#
	# ----------------------------------------------------------------------------------------
	images=[] 
	for i in range( len(digits) ):
		images.append([])
		nepoch = np.floor( nexample/len(dataloaders[i]) ) + 1
		nepoch = int(nepoch)

		if nepoch > 1:
			print('*warning* Given example={}, we have {} of individual {}-digit images.'.format(nexample, len(dataloaders[i]), digits[i]))

		for epoch in range(nepoch):
			for j, (img,label) in enumerate(dataloaders[i]):
				images[i].append(img.squeeze(0))
				if len(images[i]) >= nexample: break

	# ----------------------------------------------------------------------------------------
	#
	# Given the array of images, concatenate them one by one.
	#
	# ----------------------------------------------------------------------------------------
	for k in range(nexample):

		# images to be concatenated as
		#  [00] [10]
		#  [01] [11]
		image_00 = images[0][k]
		image_10 = images[1][k]
		image_01 = images[2][k]
		image_11 = images[3][k]

		image_0010 = torch.cat((image_00,image_10),     dim=2) # width dimension *each images has [1,28,28] shape
		image_0111 = torch.cat((image_01,image_11),     dim=2)
		image_1111 = torch.cat((image_0010,image_0111), dim=1) # height dimension

		image    = image_1111
		filename = dir_output + '{}.png'.format(k)
		save_image(image, filename)
		print(filename)

	print('done')



if __name__ == '__main__':
	make_images_with_four_digits()
