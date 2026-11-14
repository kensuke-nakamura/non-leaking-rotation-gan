

import torch
import numpy as np
import os.path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import os
import glob
from scipy import stats
import pylab
import time
import itertools
from pprint import pprint
import argparse
parser = argparse.ArgumentParser(description='learning curves')
parser.add_argument('--I', type=int, default=1, help='number of units in parallel comput')
parser.add_argument('--i', type=int, default=0, help='thread id in parallel comput')
args = parser.parse_args()


import histogram_in_batch as histogram_in_batch

mini=-1.5 #-0.5
maxi=1.5
bins=50
histogram_in_batch = histogram_in_batch.histogram_in_batch(bins=bins, mini=mini, maxi=maxi)#.to(device)


#arr_txt = glob.glob('./output/validity/*.pt')
arr_txt = glob.glob('./output/log_fid/*.npz')


i=args.i
I=args.I

summary_txt=''

for txt in arr_txt[i::I]: #for txt in arr_txt:

	basename = os.path.splitext(os.path.basename(txt))[0]
	filename = txt
	log_id = basename
	
	print(log_id)
	
	if os.path.isfile(filename):
		print('{}'.format(filename))
		
		#
		# load data
		#

		log = np.load(filename,allow_pickle=True)

		epoch=log['epoch']
		fid=log['fid']

		plt.cla()
		fig,ax = plt.subplots()
		ax.plot(epoch,fid)

		print(epoch)
		print(fid)
		
		ax.tick_params(labelsize = 16)
		ax.set_xlabel('epoch', fontsize=20 )
		ax.set_ylabel('FID', fontsize=20 )

		ax.set_xlim( (0, 100) ) 
		ax.set_ylim( (1, 1000) ) 
		ax.set_yscale('log')
			
		filename='./output/img/fid,{}.png'.format(log_id)
		fig.savefig(filename, bbox_inches='tight')
		print(filename)


		
		
