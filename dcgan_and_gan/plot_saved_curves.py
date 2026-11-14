

import numpy as np
import os
import glob
from scipy import stats

from utility_visualize.visualize import *



def plot_IPR(file_input, file_output):
	log            = np.load(file_input, allow_pickle=True)
	precision      = log['precision']
	recall         = log['recall']
	density        = log['density']
	coverage       = log['coverage']
	plot_curve_errorbar([precision,recall,density,coverage], errs=None, \
			labels=['precision', 'recall', 'density','coverage'], \
			colors=['red','blue','limegreen','y','magenta'], \
			linewidths=[2,2,2,2], 
			horizontal_line=0.6,
			filename=file_output, ylim=[-0.05,1.05])
	print('{} -> {}'.format(file_input, file_output))			


def plot_D_outputs(file_input, file_output):
	log            = np.load(file_input, allow_pickle=True)
	epoch          = log['epoch']
	D_outputs_real = log['D_outputs_real']
	D_outputs_fake = log['D_outputs_fake']
	FID            = log['FID']
	if 'MNIST' in file_input: ylim=[-1,1]
	else:                     ylim=[-3,3]
	plot_curve_D_outputs(epoch,D_outputs_real,D_outputs_fake,FID,ylim=ylim,filename=file_output)
	print('{} -> {}'.format(file_input, file_output))


def plot_overfitting_r(file_input, file_output):
	log            = np.load(file_input, allow_pickle=True)
	r_v            = log['r_v']
	r_t            = log['r_t']
	plot_curve_errorbar([r_v, r_t], errs=None, labels=['r_v','r_t'], alphas=[0.60,0.60], filename=file_output[0], ylim=[-1.1,1.1])
	plot_curve_errorbar([r_v], colors=['red'],  errs=None, labels=['r_v'],               filename=file_output[1], ylim=[-1.1,1.1])
	plot_curve_errorbar([r_t], colors=['blue'], errs=None, labels=['r_t'],               filename=file_output[2], ylim=[-1.1,1.1])
				
	print('{} -> {}'.format(file_input, file_output[0]))

def plot_curves(file_input, file_output, ylim=[0,3]):
	log    = np.load(file_input, allow_pickle=True)
	means  = log['means']
	stds   = log['stds']
	labels = log['labels']
	plot_curve_errorbar(values=means, errs=stds, labels=labels, ylim=ylim, filename=file_output)
	print('{} -> {}'.format(file_input, file_output))

'''
def plot_FID(file_input, file_output):
	log        = np.load(file_input, allow_pickle=True)
	means      = log['means']
	stds       = log['stds']
	labels     = log['labels']
	x_interval = log['x_interval']
	ylim       = log['ylim']
	grid       = log['grid']
	plot_curve_errorbar(values=means, errs=stds, labels=labels, linewidths=[2], x_interval=x_interval, ylim=[10,40], grid=grid, filename=file_output)
	print('{} -> {}'.format(file_input, file_output))
'''
def plot_FID(file_input, file_output1, file_output2):
	log        = np.load(file_input, allow_pickle=True)
	means      = log['means']
	#stds       = log['stds']
	labels     = log['labels']
	#x_interval = log['x_interval']
	ylim       = log['ylim']
	grid       = log['grid']
	# two FIDs between raw datas; and between augmented datas.
	plot_curve_errorbar(values=means, errs=None, labels=labels, 
			linewidths=[1,4], 
			alphas=[1,0.5], 
			ylim=[0,100], grid=grid, filename=file_output1)
	# two FIDs between raw datas; and between augmented datas.
	plot_curve_errorbar(values=means[0], errs=None, labels=labels[0], 
			linewidths=[2], 
			alphas=[1], 
			ylim=[0,100], grid=grid, filename=file_output2)
	print('{} -> {}'.format(file_input, file_output1))
	print('{} -> {}'.format(file_input, file_output2))

# aa,bb,loss --> aa,bb
def remove_sub_extension(string):
	return ','.join( string.split(',')[:-1] )

if __name__=='__main__':

	Draw_Each_Cuves         = True
	Draw_Average_loss       = True
	Draw_Average_prediction = True
	Draw_Average_FID        = True

	dir_input   = './result/plot/'
	dir_average = './result/plot/average/'
	dir_output  = './result/figure/'
	os.makedirs(dir_output, exist_ok=True)
	os.makedirs(dir_average,exist_ok=True)

	#-----------------------------------------------------------------------------------------
	# Step 1: Draw curves of each individual result
	#-----------------------------------------------------------------------------------------	
	if Draw_Each_Cuves:
		arr_txt = glob.glob(dir_input+'*,loss.npz')
		for txt in arr_txt:
			# Extract the file name as:
			#  aaa/bbb/ddd,loss.npz --> ddd,loss.npz --> ddd
			file_id = os.path.splitext(os.path.basename(txt))[0]
			file_id = remove_sub_extension(file_id)
			print(file_id)
			#file_id='default'
			plot_curves(file_input=dir_input+file_id+',loss.npz',       file_output=dir_output+file_id+',loss.png')
			plot_curves(file_input=dir_input+file_id+',prediction.npz', file_output=dir_output+file_id+',prediction.png')
			plot_FID(   file_input=dir_input+file_id+',FID.npz',        file_output1=dir_output+file_id+',FID.png', file_output2=dir_output+file_id+',FID_raw.png', )

			plot_D_outputs(   file_input=dir_input+file_id+',D_outputs.npz',  file_output=dir_output+file_id+',D_outputs.png')
			plot_IPR( file_input=dir_input+file_id+',IPR.npz',          file_output=dir_output+file_id+',IPR.png' )
			plot_IPR( file_input=dir_input+file_id+',IPR_trans.npz',    file_output=dir_output+file_id+',IPR_trans.png' )

			plot_IPR( file_input=dir_input+file_id+',IPR_valid.npz',          file_output=dir_output+file_id+',IPR_valid.png' )

			plot_overfitting_r( file_input=dir_input+file_id+',overfit_r.npz', \
				file_output=[dir_output+file_id+',overfit_r.png', dir_output+file_id+',overfit_rv.png', dir_output+file_id+',overfit_rt.png', ])

	#-----------------------------------------------------------------------------------------
	# Step 2: Draw curves of each individual result
	#-----------------------------------------------------------------------------------------	
	
	# loss
	if Draw_Average_loss:
		#arr_txt = glob.glob(dir_input+'*,loss.npz')
		arr_txt = glob.glob(dir_input+'*,tr,*,loss.npz')

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

		#print(ulist)
		#print(nu)
		#print(index)

		#for uname in list(ulist)[i::I]: #for uname in ulist: # Unique condition
		for uname in list(ulist): #for uname in ulist: # Unique condition
			# Compute averge
			print(uname)
			means =[]
			stds  =[]
			labels=[]
			for txt in mlist[index[uname]]: # Files to be merged within the condition
				#basename = os.path.splitext(os.path.basename(txt))[0]
				#filename = dir_input + '{}.npz'.format(basename)
				filename = txt
				if os.path.isfile(filename):
					print('{}'.format(filename))
				log = np.load(filename)
		
				means.append(  log['means']  )
				stds.append(   log['stds']   )
				labels.append( log['labels'] )			
			# Compute average
			trim=0
			means  = stats.trim_mean(np.array(means),trim,axis=0) 
			stds   = stats.trim_mean(np.array(stds),trim,axis=0) 
			labels = labels[0]
			# Output file
			basename = os.path.basename(uname)
			filename = dir_average + basename + ',ave.npz'
			np.savez(filename, means=means, stds=stds, labels=labels)
			# Draw
			file_input  = dir_average + basename + ',ave.npz'
			file_output = dir_output + basename + ',ave,loss.png'
			plot_curves(file_input=file_input, file_output=file_output)

			print(filename)


	
	# prediction
	if Draw_Average_prediction:
		arr_txt = glob.glob(dir_input+'*,prediction.npz')

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

		#print(ulist)
		#print(nu)
		#print(index)

		#for uname in list(ulist)[i::I]: #for uname in ulist: # Unique condition
		for uname in list(ulist): #for uname in ulist: # Unique condition
			# Compute averge
			print(uname)
			means =[]
			stds  =[]
			labels=[]
			for txt in mlist[index[uname]]: # Files to be merged within the condition
				#basename = os.path.splitext(os.path.basename(txt))[0]
				#filename = dir_input + '{}.npz'.format(basename)
				filename = txt
				if os.path.isfile(filename):
					print('{}'.format(filename))
				log = np.load(filename)
		
				means.append(  log['means']  )
				stds.append(   log['stds']   )
				labels.append( log['labels'] )			
			# Compute average
			trim=0
			means  = stats.trim_mean(np.array(means),trim,axis=0) 
			stds   = stats.trim_mean(np.array(stds),trim,axis=0) 
			labels = labels[0]
			# Output file
			basename = os.path.basename(uname)
			filename = dir_average + basename + ',ave.npz'
			np.savez(filename, means=means, stds=stds, labels=labels)
			# Draw
			file_input  = dir_average + basename + ',ave.npz'
			file_output = dir_output + basename + ',ave,prediction.png'
			plot_curves(file_input=file_input, file_output=file_output, ylim=[-3,3])

			print(filename)



	# FID
	if Draw_Average_FID:
		arr_txt = glob.glob(dir_input+'*,FID.npz')

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

		#print(ulist)
		#print(nu)
		#print(index)

		#for uname in list(ulist)[i::I]: #for uname in ulist: # Unique condition
		for uname in list(ulist): #for uname in ulist: # Unique condition
			# Compute averge
			print(uname)
			means =[]
			#stds  =[]
			labels=[]
			#x_interval=[]
			ylim      =[]
			grid      =[]
			for txt in mlist[index[uname]]: # Files to be merged within the condition
				#basename = os.path.splitext(os.path.basename(txt))[0]
				#filename = dir_input + '{}.npz'.format(basename)
				filename = txt
				if os.path.isfile(filename):
					print('{}'.format(filename))
				log = np.load(filename)
		
				means.append(      log['means']  )
				#stds.append(       log['stds']   )
				labels.append(     log['labels'] )	
				#x_interval.append( log['x_interval'] )
				ylim.append(       log['ylim']   )
				grid.append(       log['grid']   )
		

			# note:
			# avearge curve of FID can cause an error
			# where epochs have different number of values.
			# Consider 'Draw_Average_FID = False'

			# Compute average
			trim=0
			means      = stats.trim_mean(np.array(means),trim,axis=0) 
			#stds       = stats.trim_mean(np.array(stds),trim,axis=0) 
			labels     = labels[0]
			#x_interval = x_interval[0]
			ylim       = ylim[0]
			grid       = grid[0]
			# Output file
			basename = os.path.basename(uname)
			filename = dir_average + basename + ',ave.npz'
			#np.savez(filename, means=means, stds=stds, labels=labels, x_interval=x_interval, ylim=ylim, grid=grid)
			#np.savez(filename, means=means, stds=None, labels=labels, x_interval=x_interval, ylim=ylim, grid=grid)
			np.savez(filename, means=means, stds=None, labels=labels, x_interval=None, ylim=ylim, grid=grid)
			# Draw
			file_input  = dir_average + basename + ',ave.npz'
			file_output1 = dir_output + basename + ',ave,FID.png'
			file_output2 = dir_output + basename + ',ave,FID_raw.png'
			plot_FID(file_input=file_input, file_output1=file_output1, file_output2=file_output2)

			print(filename)




