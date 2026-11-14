import argparse
import os
import numpy as np
import math

import torch
#from torch.autograd import Variable
import torch.nn as nn
#import torch.nn.functional as F
from matplotlib import pyplot as plt
import random 
import time

t_start = time.time()


fixed_random_seed=1
torch.manual_seed(fixed_random_seed)
random.seed(      fixed_random_seed)


#
parser = argparse.ArgumentParser()
parser.add_argument('--dir_output', type=str,   default='./result', help="output directory")
parser.add_argument('--optimizer',  type=str,   default='sgd',      help="sgd/adam/adam-0.5,...")
parser.add_argument('--epoch',      type=int,   default=10000,      help="number of epochs of training")
parser.add_argument('--batch_size', type=int,   default=64,         help="size of the batches")
parser.add_argument('--lr',         type=float, default=1,          help="adam: learning rate")
parser.add_argument('--nz',         type=int,   default=5,          help="dimensionality of the latent space")
parser.add_argument('--P',          type=float, default=0.75,       help="transform probability")
parser.add_argument('--n_element',  type=int,   default=4,          help="element size")
#
parser.add_argument('--label',      type=str,   default='n=4, P=0.75 (p=1)',      help="label of condition")
#
opt = parser.parse_args()
print(opt)
#
os.makedirs(opt.dir_output,          exist_ok=True)
os.makedirs(opt.dir_output + '/npz', exist_ok=True)
#
log_id = 'n,{},P,{:.02f}'.format(opt.n_element,opt.P)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)
#
#
#

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(opt.nz, int(opt.nz/2)),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(int(opt.nz/2), opt.n_element),
			#nn.Tanh()
			nn.Sigmoid()
		)

	def forward(self, z):
		value = self.model(z)
		#return value.reshape(2,2).unsqueeze(0)
		return value
#
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		n_node1=int(opt.n_element)
		n_node2=int(opt.n_element / 2.)
		self.model = nn.Sequential(
			nn.Linear(n_node1, n_node2),
			nn.Linear(n_node2, 1),
			nn.Sigmoid(),
		)
	def forward(self, value):
		return self.model(value)

def init_weights(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.uniform(m.weight, 0, 1)
		m.bias.data.fill_(0.01)


#
#
#
adversarial_loss = torch.nn.BCELoss()
generator        = Generator().to(device)
discriminator    = Discriminator().to(device)

generator.apply(init_weights)
discriminator.apply(init_weights)

# Optimizers
if 'adam' in opt.optimizer:
	if len(opt.optimizer.split('-')) == 1:
		b1 = 0.5
	else:
		b1 = float( opt.optimizer.split('-')[1] )
	print('adam, b1=', b1)
	optimizer_G = torch.optim.Adam(generator.parameters(),     lr=opt.lr, betas=(b1, 0.999))
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(b1, 0.999))
else:
	optimizer_G = torch.optim.SGD(generator.parameters(),     lr=opt.lr)
	optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=opt.lr)
#
#
#






def ordinal(n: int):
	if 11 <= (n % 100) <= 13:
		suffix = 'th'
	else:
		suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
	return str(n) + suffix

def plot_fake_process(mean, std, condition_label=None, xscale='linear', ylabel='intensity', legend=True, filename='fake_process.png'):

	#color_set = ['red','blue','green','black']
	#linestyle_set = ['-','--','--','--']
	#linewidth_set = [2,1,1,1]

	plt.clf()
	
	plt.set_cmap('jet')

	for n in range(len(mean)):
		y = np.array( mean[n] )
		x = np.arange( len(y) ) + 1

		if n == 0: plt.plot(x,y,label='{}-dim.'.format(ordinal(n+1)),linestyle='-' ,linewidth=2,color='black',zorder=999999,alpha=0.75)
		else:      plt.plot(x,y,label='{}-dim.'.format(ordinal(n+1)),linestyle='--',linewidth=2)

		if std is not None: 
			err = std[n]
			plt.fill_between(x, y-err, y+err, alpha=0.3, color=color_set[n])

	plt.xlabel('iteration', fontsize=20)
	plt.ylabel(ylabel,      fontsize=20)

	plt.xscale(xscale)

	plt.xticks(fontsize=18)
	plt.yticks([0,0.25,0.5,0.75,1], fontsize=18)

	plt.ylim([-0.05,1.05])

	if condition_label is not None: 
		plt.title(condition_label, fontsize=24) #, loc='right'

	plt.grid()

	#if legend: plt.legend(loc='lower left', fontsize=16)
	if legend: plt.legend(fontsize=18)
	plt.savefig(filename, bbox_inches='tight')
	
	#print(filename)





# https://pytorch.org/docs/stable/generated/torch.roll.html
def Phi(data, P):
	shift = np.random.choice(np.arange(1,opt.n_element), len(data)) # 0 is not included
	shift[np.random.random(size=len(data)) > P] = 0
	transformed = [torch.roll(d,a) for d,a in zip(data,shift)]
	return torch.stack(transformed)


#
#
#
#

#  example: (n_element = 4, batch size = 3)
#
#  0 1 0 0 
#  0 1 0 0
#  0 1 0 0 
#

real_data = torch.zeros(  (opt.batch_size, opt.n_element)  )
real_data[:,0] = 1  # activate 1 (2nd) element

# test data with fixed latent
num_test_fake = 1000 
fixed_z       = torch.FloatTensor(  np.random.normal(0, 1, (num_test_fake, opt.nz))  ).to(device)#.requires_grad_()


log_fake_fixedz_one = []
for n in range(opt.n_element):
	log_fake_fixedz_one.append([])


log_fake_fixedz_mean = []
log_fake_fixedz_std  = []
for n in range(opt.n_element):
	log_fake_fixedz_mean.append([])
	log_fake_fixedz_std.append([])

# ----------
#  Training
# ----------
prediction_real=[]
prediction_fake=[]
log_fake=[]
log_epoch=[]
#
t_lap = time.time()
for epoch in range(opt.epoch + 1):

	# Configure input
	real = real_data.clone().requires_grad_().to(device)

	# Adversarial ground truths
	label_real = torch.FloatTensor(len(real), 1).fill_(1.0).to(device)
	label_fake = torch.FloatTensor(len(real), 1).fill_(0.0).to(device)
		
	# ---------------------------------------------------------------
	#  Train Generator
	# ---------------------------------------------------------------

	optimizer_G.zero_grad()
	# Sample noise as generator input
	z    = torch.FloatTensor(  np.random.normal(0, 1, (len(real), opt.nz))  ).requires_grad_().to(device)
	fake = generator(z)

	# apply transform
	real = Phi(real, opt.P)
	fake = Phi(fake, opt.P)

	g_loss = adversarial_loss(discriminator(fake), label_real)
	g_loss.backward()
	if epoch > 0: optimizer_G.step()


	# ---------------------------------------------------------------
	#  Train Discriminator
	# ---------------------------------------------------------------
	optimizer_D.zero_grad()

	# Measure discriminator's ability to classify real from generated samples
	real_loss = adversarial_loss(discriminator(real),          label_real)
	fake_loss = adversarial_loss(discriminator(fake.detach()), label_fake)
	d_loss    = (real_loss + fake_loss) / 2
	d_loss.backward()
	if epoch > 0: optimizer_D.step()


	# ---------------------------------------------------------------
	#  Post process
	# ---------------------------------------------------------------
	with torch.no_grad():
		prediction_real.append( torch.Tensor( discriminator(real)          ).mean().detach().cpu() )
		prediction_fake.append( torch.Tensor( discriminator(fake.detach()) ).mean().detach().cpu() )

		log_epoch.append( epoch )

		z    = torch.FloatTensor(  np.random.normal(0, 1, (len(real), opt.nz))  ).to(device)
		fake = generator(z)
		log_fake.append( fake.mean().detach().cpu() )


	with torch.no_grad():
		#z    = torch.FloatTensor(  np.random.normal(0, 1, (10, opt.nz))  )
		#fake = generator(z)
		test_fake = generator(fixed_z)


		# pick a single example
		for n in range(opt.n_element):
			log_fake_fixedz_one[n].append( test_fake[0][n].detach().cpu() )


		mean = test_fake.detach().cpu().mean(dim=0)
		std  = test_fake.detach().cpu().std(dim=0)

		
		# mean across examples
		for n in range(opt.n_element):
			log_fake_fixedz_mean[n].append( mean[n] )
			log_fake_fixedz_std[n].append( std[n] )

	if epoch % (opt.epoch / 100) == 0:
	#if True:
		lap_time = time.time() - t_lap
		t_lap = time.time()
		print(
			'[Ep {:.2f}%] [D loss: {:.4f}] [G loss: {:.4f}] fake {:.06f}: {}: {:.2f} sec '.format(float( epoch / opt.epoch * 100. ), d_loss.item(), g_loss.item(), fake.mean().item(), log_id, lap_time)
		, end='')
		torch.set_printoptions(precision=4)

		print( test_fake.mean(dim=0) )

		if opt.n_element <= 10: legend=True
		else:                   legend=False
		if opt.P > 0:           legend=False
		filename = opt.dir_output + '/fake_process_' + log_id + '.png'
		plot_fake_process(log_fake_fixedz_mean, None, condition_label=opt.label, xscale='log', ylabel='intensity', legend=legend, filename=filename)



filename = opt.dir_output + '/npz/fake_process_' + log_id + '.png'
means= np.array(log_fake_fixedz_mean)
stds = np.array(log_fake_fixedz_std)
label = opt.label
np.savez(filename, means, stds, label)



#
# Visualize the trainiing process
#
prediction_real = np.array(prediction_real)
prediction_fake = np.array(prediction_fake)
log_epoch       = np.array( log_epoch )



def plot_prediction(real, fake, filename='prediction.png', xscale='linear'):
	plt.clf()
	plt.axhline(y=0.5, color = 'gray', linestyle = '--', zorder=0)
	#
	plt.plot(real, label='real')
	plt.plot(fake, label='fake')
	plt.yticks([0,1])
	plt.ylim([-0.05,1.05])
	plt.ylabel('prediction')
	plt.xlabel('epoch')
	plt.xscale(xscale)
	plt.legend()
	#
	plt.savefig(filename, bbox_inches='tight')

#filename = opt.dir_output + '/' + log_id + ',prediction.png'
#plot_prediction(prediction_real, prediction_fake, filename, xscale='log')


t_end = time.time()
time_elapsed = t_end - t_start
print('elapsed time: {}'.format(time_elapsed))

txt_file = open('./result/runtime.txt', 'at')
txt_file.write('{}, {}, seconds\n'.format(log_id, time_elapsed))
txt_file.close()


