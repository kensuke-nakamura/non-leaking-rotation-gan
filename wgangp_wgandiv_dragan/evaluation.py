
import numpy as np
import os.path
import matplotlib
from matplotlib.ticker import FormatStrFormatter
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize']=100000

import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import os
import glob
from os.path import basename
from math import pow as power

from scipy import stats

import pylab
import time

import itertools
from pprint import pprint

#http://akuederle.com/matplotlib-zoomed-up-inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


if not os.path.exists('output/img'):  os.makedirs('output/img')


import argparse
parser = argparse.ArgumentParser(description='learning curves')
parser.add_argument('--I', type=int, default=1, help='number of units in parallel comput')
parser.add_argument('--i', type=int, default=0, help='thread id in parallel comput')
args = parser.parse_args()


i=args.i
I=args.I


def initText(filename):
	txtfile=open(filename,'wt')
	txtfile.close();




indir = './output/fid_is'

if True and i == 0:
	arr_txt = glob.glob(indir+'/*.npz')

	outfile4='./output/evaluations_ave.txt'
	initText(outfile4)
	stat4_txt=''


	# Find the unique filenames (w/o trial number)
	ulist=[]
	for txt in arr_txt: #for txt in arr_txt:
		uname,tail=txt.split(',tr,')
		ulist.append(uname)
	ulist=set(ulist)
	nu=len(ulist)

	# Index by the unique filename
	index={}
	for idx,uname in enumerate(ulist):
		index[uname]=idx

	# Merge list
	mlist=[]
	for u in ulist:
		mlist.append([])
	#
	for txt in arr_txt: #for txt in arr_txt:
		uname,tail=txt.split(',tr,')
		if uname:
			mlist[index[uname]].append(txt)


	for uname in list(ulist)[i::I]: #for uname in ulist: # Unique condition

		fid=[]
		inception_score=[]

		for txt in mlist[index[uname]]: # Files to be merged within the condition
 
			basename = os.path.splitext(os.path.basename(txt))[0]
			filename = indir+'/{}.npz'.format(basename)

			log = np.load(filename)

			fid.append(log['fid'])
			inception_score.append(log['inception_score'])
			 
			#label=txt.split('.npz')[0].split(',')[-1]  #txt.split(',')[-1].split('.')[-2]

		# Filename to output
		basename = os.path.splitext(os.path.basename(uname+'.dummy'))[0] #+ ',' + label

		ntrials  = len(fid)

		fid_mean = np.mean(fid)
		fid_min = np.min(fid)
		is_mean  = np.mean(inception_score) 
		is_max  = np.max(inception_score) 

		print(ntrials,basename)
		stat4_txt += '{},ntrial,{},fid,{},min-fid,{},is,{},max-is,{}\n'.format(basename,ntrials,fid_mean,fid_min,is_mean,is_max)


	txtfile=open(outfile4,'at')
	txtfile.write(stat4_txt)
	txtfile.close();






# ---------------------------------------------------------------------------
# Step 2: 
#     Given evaluation results in ./evaluation/txt/, 
#     concanate them into ./ev
# ---------------------------------------------------------------------------

file_evaluation_results = './output/best_FID_IS.txt'
path_evaluation_txt     = './output/best_FID/'
print('Concanate evaluated results > ', file_evaluation_results)

with open(file_evaluation_results, 'w') as outfile:
	arr_txt = glob.glob(path_evaluation_txt+'*.txt')
	for idx,txt in enumerate(arr_txt):
		with open(txt) as infile:
			for line in infile:
				outfile.write(line)






