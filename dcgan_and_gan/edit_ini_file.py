

# -------------------------------------------------------------------------------------------------------------------------------
#
# Option file editor ver 21.11.05:
#   Given an init file './option/default.ini' with arguments like --Optimize.generator__optimize.generator.learning_rate_initial 0.01,
#   this script creates a new option file in './result/option/'.
#
# Usage:
#
#   Options are shown by
#   
#   > python3 edit_ini_file.py './option/default.ini'
#   
#
#   Options are changed and saved as
#
#   > python3 edit_ini_file.py './option/default.ini' --dir_output './result/option/' --new_option_name edited.ini --Optimize.generator__learning_rate_initial 0.01 --Dataset.real__name Celeba
#
# -------------------------------------------------------------------------------------------------------------------------------
import configparser
import argparse
import sys
import os
from option import abstract_of_option


if len(sys.argv) == 1: 
	print('error: input file is required like')
	print('python3 edit_ini_file.py ./option/default.ini')
	exit(-1)

elif len(sys.argv) == 2: 

	if sys.argv[1][:2] == '--':
		print('error: input file is required followed by options like')
		print('python3 edit_ini_file.py ./option/default.ini --Optimize__size_batch 100')
		exit(-1)

	file_option_input = sys.argv[1]
	display_options   = True

elif len(sys.argv) > 2: 

	if sys.argv[1][:2] == '--':
		print('error: input file is required followed by options like')
		print('python3 edit_ini_file.py ./option/default.ini --Optimize__size_batch 100')
		exit(-1)

	file_option_input = sys.argv[1]
	display_options   = False


# We will call the argparser using the 3rd and later items
argv_for_argparser = sys.argv[2:]

# Over-write a new option file if the same file exists.
over_write        = True


# -------------------------------------------------------------------------------------------
# Input and constants
# -------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------
# Step 1: read the default option as configparser
# -------------------------------------------------------------------------------------------
config = configparser.ConfigParser()
config.read(file_option_input)


# -------------------------------------------------------------------------------------------
# Step 2: convert the configparser to argparse
# -------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser('option')
for section in config.sections():
	for item,value in config.items(section):
		arg_key = '--' + section + '__' + item
		parser.add_argument(arg_key,  type=str, default=value, help=section+item)
# additional arguments
parser.add_argument('--new_option_name', type=str, default=None,   help='new option file')
parser.add_argument('--dir_output', type=str, default='./result/option/',   help='new option file')
opt = parser.parse_args(argv_for_argparser)

os.makedirs(opt.dir_output, exist_ok=True)

# -------------------------------------------------------------------------------------------
# Step 3: update the configparser using the argparse
# -------------------------------------------------------------------------------------------
for arg in vars(opt):
	if len(arg.split('__')) == 1:
		# we skip the --output argument
		pass
	else:
		section,item = arg.split('__')
		value        = getattr(opt, arg)
		if display_options:
			print('\t\t --{}__{} {}'.format(section, item, value) )
		#if value != config.get(section,item):
		#	print('(updated) \t --{}__{} {} -> {}'.format(section,item,config.get(section,item),value) )
		#
		#else:	print('\t\t --{}__{} {}'.format(section, item, value) )
		config.set(section,item,value)

if display_options:
	print('Options can be edited as')
	print('python3 edit_ini_file.py {} --dir_output ./result/option/ --Dataset.real__name celeba'.format(file_option_input))


# ---------------------------------------------------------------
# save the edit option file
# ---------------------------------------------------------------
# A new option name is created automatically if not specified.
if opt.new_option_name is None: file_option_output = opt.dir_output + abstract_of_option(config) + '.ini'  
else:                           file_option_output = opt.dir_output + opt.new_option_name

if os.path.exists(file_option_output) and over_write==False:
	#print(' Already exists: ', file_option_output)
	pass
else:
	with open(file_option_output, 'w') as configfile:   
		config.write(configfile)
	#print('saved', file_option_input, ' -> ', file_option_output)

# return value from python script to shell script 
# https://stackoverflow.com/questions/34171568/return-value-from-python-script-to-shell-script
if not display_options:
	print(file_option_output)
