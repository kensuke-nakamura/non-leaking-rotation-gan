from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models_64x64
import PIL.Image as Image
import tensorboardX
import torch
from torch.autograd import grad
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils


# added
import torchvision.datasets as datasets
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import time
t_elapsed_start=time.time()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='/hdd1/dataset/', help="path to dataset root")
parser.add_argument("--algorithm", type=str, default='dragan', help="algorithm") # 'dragan' or ''dragan_dn'
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--dataset", type=str, default='celeba', help="dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,    help="adam: b1")
parser.add_argument("--b2", type=float, default=0.999,  help="adam: b2")
parser.add_argument('--reg-lambda', type=float, default=10) # coefficient for regularization (gradient penalty, gradient regularization)
parser.add_argument("--epoch", type=int, default=5, help="number of epochs of training")
parser.add_argument("--trial", type=int, default=1)
parser.add_argument('--ada',       action='store_true', help='Nvidia Ada')
parser.add_argument('--nonleak',   action='store_true', help='proposed method')
opt = parser.parse_args()
print(opt)


algorithm=opt.algorithm
if opt.nonleak: algorithm = algorithm + '_ours'
if opt.ada:     algorithm = algorithm + '_ada'


log_id='{},{},lr,{},{},reg,{},ep,{},tr,{}'.format(opt.dataset,algorithm,opt.lr,opt.b1,opt.reg_lambda,opt.epoch,opt.trial)
print(log_id)
# wait seconds for other processes
t_delay = (opt.trial % 4) * 2
time.sleep(t_delay)

""" param """
epochs = opt.epoch
batch_size = opt.batch_size
lr = opt.lr #0.0002
z_dim = 100


# DRAGAN regularization term
def gradient_penalty(x, f):
	# interpolation
	shape = [x.size(0)] + [1] * (x.dim() - 1)
	alpha = torch.rand(shape).to(device)
	beta = torch.rand(x.size()).to(device)

	y = x + 0.5 * x.std() * beta
	z = x + alpha * (y - x)

	# gradient penalty
	z = Variable(z, requires_grad=True).to(device)
	o = f(z)
	g = grad(o, z, grad_outputs=torch.ones(o.size()).to(device), create_graph=True)[0].view(z.size(0), -1)
	gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

	return gp

""" gpu """
#gpu_id = [2]
#utils.cuda_devices(gpu_id)




""" data """
'''
crop_size = 108
re_size = 64
offset_height = (218 - crop_size) // 2
offset_width = (178 - crop_size) // 2
crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Lambda(crop),
	 transforms.ToPILImage(),
	 transforms.Scale(size=(re_size, re_size), interpolation=Image.BICUBIC),
	 transforms.ToTensor(),
	 transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

#imagenet_data = dsets.ImageFolder('./data/img_align_celeba', transform=transform)
imagenet_data = dsets.ImageFolder('./../../data/celeba', transform=transform)
data_loader = torch.utils.dsata.DataLoader(imagenet_data,
										  batch_size=batch_size,
										  shuffle=True,
										  num_workers=4)
'''
dirroot=opt.root #'./../data/'
import torchvision.transforms as transforms
if opt.dataset == 'celeba':
	transform=transforms.Compose([
		transforms.Resize(opt.img_size),
		transforms.CenterCrop(opt.img_size),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])
	dataset = datasets.ImageFolder(root=dirroot+'celeba', transform=transform)
	nc=3

elif 'food-101' in opt.dataset: #
	# We introduce a challenging data set of 101 food categories, with 101'000 images. 
	# For each class, 250 manually reviewed test images are provided as well as 750 training images. 
	# On purpose, the training images were not cleaned, and thus still contain some amount of noise. 
	# This comes mostly in the form of intense colors and sometimes wrong labels. 
	# All images were rescaled to have a maximum side length of 512 pixels.
	transform_list = []
	transform_list.append( transforms.ToTensor() )
	transform_list.append( transforms.Resize(opt.img_size) )
	transform_list.append( transforms.CenterCrop(opt.img_size) )
	transform_list.append( transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) )
	transform_compose   = transforms.Compose(transform_list)
	dataset             = datasets.Food101(opt.root+'food-101', split='train',  download=True, transform=transform_compose)
	nc=3   

elif 'stanford-cars' in opt.dataset:
	# The Cars dataset contains 16,185 images of 196 classes of cars. 
	# The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split
	transform_list = []
	transform_list.append( transforms.ToTensor() )
	transform_list.append( transforms.Resize(opt.img_size) )
	transform_list.append( transforms.CenterCrop(opt.img_size) )
	transform_list.append( transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) )
	transform_compose   = transforms.Compose(transform_list)
	dataset             = datasets.StanfordCars(opt.root+'stanford-cars', split='train',  download=True, transform=transform_compose)
	nc=3

elif 'lsun' in opt.dataset: # 'lsun-church'
	transform=transforms.Compose([
		transforms.Resize(opt.img_size),
		transforms.CenterCrop(opt.img_size),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])
	classes=[]
	if 'church'     in opt.dataset:  classes.append('church_outdoor_train')
	if 'conference' in opt.dataset:  classes.append('conference_room_train')
	if 'restaurant' in opt.dataset:  classes.append('restaurant_train')
	dataset = datasets.LSUN(root=dirroot+'lsun', classes=classes, transform=transform)
	print('LSUN dataset with classes: {}'.format(classes))
	print('Total num. of examples: ', len(dataset))
	nc=3

elif opt.dataset == 'oxflower':
	transform=transforms.Compose([
		transforms.Resize(opt.img_size),
		transforms.CenterCrop(opt.img_size),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])
	dataset = datasets.ImageFolder(root=dirroot+'oxflower', transform=transform)
	nc=3


data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=2)



if opt.nonleak: from   transform_nonleaking import Phi
if opt.ada:     import transform_ada as ada


""" model """
D = models_64x64.DiscriminatorWGANGP(3).to(device)
G = models_64x64.Generator(z_dim).to(device)
bce = nn.BCEWithLogitsLoss().to(device)
#utils.cuda([D, G, bce])

d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(opt.b1, opt.b2))
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(opt.b1, opt.b2))

'''
""" load checkpoint """
ckpt_dir = './checkpoints/celeba_dragan'
utils.mkdir(ckpt_dir)
try:
	ckpt = utils.load_checkpoint(ckpt_dir)
	start_epoch = ckpt['epoch']
	D.load_state_dict(ckpt['D'])
	G.load_state_dict(ckpt['G'])
	d_optimizer.load_state_dict(ckpt['d_optimizer'])
	g_optimizer.load_state_dict(ckpt['g_optimizer'])
except:
	print(' [*] No checkpoint!')
	start_epoch = 0
'''

""" run """
#writer = tensorboardX.SummaryWriter('./summaries/celeba_dragan')

# FID
import os
import numpy as np
import fid_score as fid # FID metric
import inception_score as inception_score 
os.makedirs('output/img', exist_ok=True)
# FID process
compute_fid_epoch=True
fid_step = int(epochs / 10)
if fid_step < 1: fid_step = 1
print('FID sampling step:', fid_step)
buf_fid=[]
buf_fid_epoch=[]
if compute_fid_epoch:
	tmp_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=2)
# best FID
best_FID = 9999999
best_IS_mean, best_IS_std = 0,0 # IS using Generator with the best FID
os.makedirs('output/model', exist_ok=True)
os.makedirs('output/best_FID', exist_ok=True)


z_sample = Variable(torch.randn(100, z_dim)).to(device)

for epoch in range(epochs): #or epoch in range(start_epoch, epochs):
	for i, (imgs, _) in enumerate(data_loader):
		# step
		step = epoch * len(data_loader) + i + 1

		# set train
		G.train()

		# leafs
		imgs = Variable(imgs).to(device)
		if opt.nonleak: imgs = Phi(imgs)
		if opt.ada:     imgs = ada.Phi(imgs)
		

		bs = imgs.size(0)
		z = Variable(torch.randn(bs, z_dim)).to(device)
		r_lbl = Variable(torch.ones(bs)).to(device)
		f_lbl = Variable(torch.zeros(bs)).to(device)

		#print(imgs)
		#print(imgs.shape)

		f_imgs = G(z)
		if opt.nonleak: f_imgs = Phi(f_imgs)
		if opt.ada:     f_imgs = ada.Phi(f_imgs)

		# train D
		r_logit = D(imgs)
		f_logit = D(f_imgs.detach())
		d_r_loss = bce(r_logit, r_lbl)
		d_f_loss = bce(f_logit, f_lbl)
		gp = gradient_penalty(imgs.data, D)
		d_loss = d_r_loss + d_f_loss + gp * opt.reg_lambda

		D.zero_grad()
		d_loss.backward()
		d_optimizer.step()

		#writer.add_scalars('D/loss',
		#				   {"d_loss": d_loss.data.cpu().numpy(),
		#					"d_r_loss": d_r_loss.data.cpu().numpy(),
		#					"d_f_loss": d_f_loss.data.cpu().numpy()},
		#				   global_step=step)
		#writer.add_scalar('D/gp', gp.data.cpu().numpy(), global_step=step)

		# train G
		f_logit = D(f_imgs)
		g_loss = bce(f_logit, r_lbl)

		D.zero_grad()
		G.zero_grad()
		g_loss.backward()
		g_optimizer.step()

		#writer.add_scalars('G',
		#				   {"g_loss": g_loss.data.cpu().numpy()},
		#				   global_step=step)

		if opt.ada:  ada.update_global_P( D(imgs).detach().cpu().numpy().squeeze() )

		if (i + 1) % 50 == 0:
			print("Epoch: (%3d/%3d) (%5d/%5d): %s" % (epoch, epochs, i + 1, len(data_loader), log_id))

	


	'''
	utils.save_checkpoint({'epoch': epoch + 1,
						   'D': D.state_dict(),
						   'G': G.state_dict(),
						   'd_optimizer': d_optimizer.state_dict(),
						   'g_optimizer': g_optimizer.state_dict()},
						  '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
						  max_keep=2)
	'''

	# best FID
	# FID process
	if compute_fid_epoch and (epoch % fid_step == 0 or epoch == epochs - 1):

		with torch.no_grad():
			print('Computing FID..')

			original_real=[]
			original_fake=[]

			for i, (imgs, _) in enumerate(tmp_dataloader):
				original_real.append(Variable(imgs).type(torch.FloatTensor))
				z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (batch_size, z_dim))))
				fake_imgs = G(z).data
				original_fake.append(fake_imgs.type(torch.FloatTensor))

				# we sameple a subset of the examples
				if i * batch_size > 5000: break;

			original_real = torch.cat(original_real,dim=0)
			original_fake = torch.cat(original_fake,dim=0)

			# fid between fakes with the original real-data without any filtering
			fid_original = fid.calculate_fid_given_batches(original_real, original_fake, batch_size=opt.batch_size)
			#is_mean, is_std = inception_score(original_fake, cuda=True, batch_size=opt.batch_size, resize=True, splits=10)

			buf_fid_epoch.append(epoch)
			buf_fid.append(fid_original)

			os.makedirs('output/log_fid', exist_ok=True)
			filename = 'output/log_fid/{}.npz'.format(log_id)

			np.savez(filename, epoch=buf_fid_epoch, fid=buf_fid) 

			torchvision.utils.save_image(original_fake.data[:9], 'output/img/fake_raw_ep{:03d},{}.png'.format(epoch,log_id), nrow=3, normalize=True)
	
			# best FID
			if fid_original < best_FID:
				best_FID = fid_original
				print('Updated the best model with test FID {}: {}'.format(best_FID, log_id))
				filename = './output/model/' + log_id + ',generator,best.pt'
				torch.save(G.state_dict(), filename)
				print('saved best mode', filename)

				best_IS_mean, best_IS_std = inception_score.inception_score(original_fake, cuda=True, batch_size=batch_size, resize=True, splits=10)


if opt.ada:  ada.plot_global_P(filename='./output/img/' + log_id + ',AdaP.png')

# vasualize fakes
import os
save_dir='./output/'
os.makedirs(save_dir,exist_ok=True)
os.makedirs(save_dir+'img',exist_ok=True)
#
#
filename = save_dir + 'img/fake,{}.png'.format(log_id)
G.eval()
#f_imgs_sample = (G(z_sample).data + 1) / 2.0
imgs=G(z_sample).data
torchvision.utils.save_image(imgs[:9], filename, nrow=3, normalize=True)


# FID and inception score
with torch.no_grad():

	print('FID: computering discrepancy metrics between real data with fakes')

	original_real=[]
	original_fake=[]

	for i, (imgs, _) in enumerate(data_loader):
		original_real.append(Variable(imgs).type(torch.FloatTensor))
		z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (batch_size, z_dim))))
		fake_imgs = G(z).data
		original_fake.append(fake_imgs.type(torch.FloatTensor))

		# we sameple a subset of the examples
		if i * batch_size > 5000: break;

	original_real = torch.cat(original_real,dim=0)
	original_fake = torch.cat(original_fake,dim=0)

	# fid between fakes with the original real-data without any filtering
	fid_original = fid.calculate_fid_given_batches(original_real, original_fake, batch_size=batch_size)
	is_mean, is_std = inception_score.inception_score(original_fake, cuda=True, batch_size=batch_size, resize=True, splits=10)


	txt_file = open(save_dir+'debug_FID_IS.txt', 'at')
	txt_file.write('{} bestFID {} bestIS {} finalFID {} finalIS {}\n'.format(log_id, best_FID, best_IS_mean, fid_original, is_mean))
	txt_file.close()

	save_dir='./output/fid_is'
	os.makedirs(save_dir,exist_ok=True)
	filename=save_dir+'/{}.npz'.format(log_id)
	np.savez(filename, fid=fid_original, inception_score=is_mean) 

	
	if compute_fid_epoch: # FID process
		buf_fid_epoch.append(epoch)
		buf_fid.append(fid_original)

		os.makedirs('output/log_fid', exist_ok=True)
		filename = 'output/log_fid/{}.npz'.format(log_id)

		np.savez(filename, epoch=buf_fid_epoch, fid=buf_fid) 



# best FID
filename = './output/best_FID/' + log_id + '.txt'
txt_file = open(filename, 'w')
txt_file.write('{} bestFID {} bestIS {} \n'.format(log_id, best_FID, best_IS_mean))
txt_file.close()
# save final model
filename = './output/model/' + log_id + ',generator,final.pt'
torch.save(G.state_dict(), filename)





# computation time
time_elapsed = (time.time() - t_elapsed_start)
print('elapsed time = %d sec' % time_elapsed)

txt_file = open('output/runtime.txt', 'at')
txt_file.write('{}, {}, seconds\n'.format(log_id, time_elapsed))
txt_file.close()

