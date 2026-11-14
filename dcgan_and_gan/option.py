import configparser
import json

def read_config(file_option):
	
	config = configparser.ConfigParser()
	config.read(file_option)

	config_dataset_real = {
		'name'              : config.get('Dataset.real', 'name'),
		'channel'           : int(config.get('Dataset.real', 'channel')),
		'height'            : int(config.get('Dataset.real', 'height')),
		'width'             : int(config.get('Dataset.real', 'width')),
		'normalize'         : config.get('Dataset.real', 'normalize'), 
		'normalize_value1'  : float(config.get('Dataset.real', 'normalize.value1')),
		'normalize_value2'  : float(config.get('Dataset.real', 'normalize.value2')), 
		'size_data'         : config.get('Dataset.real', 'size_data'),
	} 

	config_dataset_fake = {
		'channel'       : int(config.get('Dataset.fake', 'channel')),
		'height'        : int(config.get('Dataset.fake', 'height')),
		'width'         : int(config.get('Dataset.fake', 'width')),
		'length_latent' : int(config.get('Dataset.fake', 'length_latent')), 
	} 


	config_transform = {
		'probability'   : float(config.get('Transform', 'probability')),
		'option'        : config.get('Transform', 'option').split(','),
		#'use_condition' : bool(config.getboolean('Transform', 'use_condition')),
		'use_condition' : config.get('Transform', 'use_condition'),
	}

	config_network = {
		'generator'     : config.get('Network', 'generator'),
		'discriminator' : config.get('Network', 'discriminator') 
	}

	config_optimize = {
		'algorithm'      : config.get('Optimize', 'algorithm'),
		'size_batch'     : int(config.get('Optimize', 'size_batch')),
		'number_epoch'   : int(config.get('Optimize', 'number_epoch')),
		'trial_id_number': int(config.get('Optimize', 'trial_id_number')),
	}

	config_optimize_generator = {
		'loss'                          : config.get('Optimize.generator', 'loss'),
		'weight_total_variation'        : float(config.get('Optimize.generator', 'weight_total_variation')),
		'weight_decay'                  : float(config.get('Optimize.generator', 'weight_decay')),
		'momentum'                      : float(config.get('Optimize.generator', 'momentum')),
		'learning_rate_initial'         : float(config.get('Optimize.generator', 'learning_rate_initial')),
		'learning_rate_middle'          : float(config.get('Optimize.generator', 'learning_rate_middle')),
		'learning_rate_final'           : float(config.get('Optimize.generator', 'learning_rate_final')),
		'learning_rate_slope'           : float(config.get('Optimize.generator', 'learning_rate_slope')),
		'learning_rate_ratio_division'  : float(config.get('Optimize.generator', 'learning_rate_ratio_division'))
	}

	config_optimize_discriminator = {
		'loss'                          : config.get('Optimize.discriminator', 'loss'),
		'gradient_regularization'       : float(config.get('Optimize.discriminator', 'gradient_regularization')),
		'weight_decay'                  : float(config.get('Optimize.discriminator', 'weight_decay')),
		'momentum'                      : float(config.get('Optimize.discriminator', 'momentum')),
		'learning_rate_initial'         : float(config.get('Optimize.discriminator', 'learning_rate_initial')),
		'learning_rate_middle'          : float(config.get('Optimize.discriminator', 'learning_rate_middle')),
		'learning_rate_final'           : float(config.get('Optimize.discriminator', 'learning_rate_final')),
		'learning_rate_slope'           : float(config.get('Optimize.discriminator', 'learning_rate_slope')),
		'learning_rate_ratio_division'  : float(config.get('Optimize.discriminator', 'learning_rate_ratio_division'))
	}


	return config_dataset_real, config_dataset_fake, config_transform,  \
		config_network,     config_optimize,     config_optimize_generator, \
		config_optimize_discriminator, \
		config




def abstract_network_name(name_to_be_shorten, label='Discriminator'):

	if label == 'Discriminator': net = 'D'
	else:                        net = 'G'

	name = name_to_be_shorten
	split_name = name.split('_')

	if split_name[0] == label: # Discriminator_Convolution_Stride_Leaky_Relu_32
		new_name = ''.join([c[:2] for c in name.split('_')])

	elif split_name[0] == 'GAN':
		new_name = ''.join([c[:2] for c in name.split('_')])

	else: # DCGAN_Discriminator32_wo_Sigmoid

		if label+'32' in split_name[1]:   new_name = split_name[0] + '_' + net + '_32'
		else:                             new_name = split_name[0] + '_' + net

		for s in split_name:
			if 'Bottleneck' in s: new_name = new_name + '_bneck' + s[len('Bottleneck'):]

	return new_name


def abstract_of_option(config):

	def str_without_leading_zero( string ):
		return string.lstrip('0')
		#value = float( value )
		#return ("%f" % value ).lstrip('0')
	
	#config = configparser.ConfigParser()
	#config.read(file_option)

	
	# --------------------------------------------------------------------------------------------------------
	# make abbreviated label of each option
	# --------------------------------------------------------------------------------------------------------
	# transformation
	label_options           = config.get('Transform', 'option').split(',')
	label_options           = '-'.join(label_options)
	#
	label_transforms        = 'P,{:0.2f},'.format( float(config.get('Transform', 'probability')) )
	label_transforms        += label_options
	#if bool(config.getboolean('Transform', 'use_condition')):  label_transforms += ',use_cond'
	#else:                                                      label_transforms += ',no_cond'
	#label_transforms        += ',cond,' + config.get('Transform', 'use_condition')

	# networks
	#label_model_discriminator = ''.join([c[:2] + c[-2:] for c in config.get('Network', 'generator').split('_')])
	#label_model_generator     = ''.join([c[:2] + c[-2:] for c in config.get('Network', 'discriminator').split('_')])
	label_model_discriminator = abstract_network_name( config.get('Network', 'discriminator'), 'Discriminator')
	#label_model_generator     = abstract_network_name( config.get('Network', 'generator'),     'Generator')
	label_model_generator     = abstract_network_name( config.get('Network', 'generator'),     'Generator')

	# learning-rate
	#label_optim_discriminator = 'D,' 
	#label_optim_discriminator += str_without_leading_zero( config.get('Optimize.discriminator', 'learning_rate_initial') ) + ','
	#label_optim_discriminator += str_without_leading_zero( config.get('Optimize.discriminator', 'learning_rate_initial') ) + '_'
	#label_optim_discriminator += str_without_leading_zero( config.get('Optimize.discriminator', 'learning_rate_middle')  ) + '_'
	#label_optim_discriminator += str_without_leading_zero( config.get('Optimize.discriminator', 'learning_rate_final')   ) + ','
	#label_optim_discriminator += 'gr,' + config.get('Optimize.discriminator', 'gradient_regularization')


	#label_optim_generator = 'G,' 
	#label_optim_generator += str_without_leading_zero( config.get('Optimize.generator', 'learning_rate_initial') ) #+ ','
	#label_optim_generator += str_without_leading_zero( config.get('Optimize.generator', 'learning_rate_initial') ) + '_'
	#label_optim_generator += str_without_leading_zero( config.get('Optimize.generator', 'learning_rate_middle')  )+ '_'
	#label_optim_generator += str_without_leading_zero( config.get('Optimize.generator', 'learning_rate_final')   )+ ','
	#label_optim_generator += 'tv,' + config.get('Optimize.generator', 'weight_total_variation')


	label_optim = 'lrD,' 
	label_optim += str_without_leading_zero( config.get('Optimize.discriminator', 'learning_rate_initial') ) + ','
	label_optim += 'lrG,' 
	label_optim += str_without_leading_zero( config.get('Optimize.generator', 'learning_rate_initial') ) + ','
	label_optim += 'gr,' + config.get('Optimize.discriminator', 'gradient_regularization')



	# --------------------------------------------------------------------------------------------------------
	# abstract of option
	# --------------------------------------------------------------------------------------------------------
	abstract=[]

	abstract.append( config.get('Dataset.real', 'name') ) 
	abstract.append( config.get('Dataset.real', 'size_data')  ) 
	abstract.append( label_transforms ) 
	abstract.append( config.get('Optimize', 'algorithm') )
	abstract.append( label_optim )
	#abstract.append( label_optim_discriminator )
	#abstract.append( label_optim_generator )
	#abstract.append( 'B,'  + config.get('Optimize', 'Optimize.size_batch') )
	abstract.append( label_model_discriminator )
	abstract.append( label_model_generator )

	abstract.append( 'nz,' + config.get('Dataset.fake', 'length_latent') )
	abstract.append( 'e,' + config.get('Optimize', 'number_epoch') )
	abstract.append( 'tr,' + config.get('Optimize', 'trial_id_number') )

	abstract = ','.join(abstract)

	#print(file_option, ' -> ', abstract)

	return abstract


if __name__ == '__main__':

	config = configparser.ConfigParser()
	config.read('./option/default.ini')
	abstract = abstract_of_option(config)
	print(abstract)

	#abstract_label_of_option('./option/211021.2.ini')
	#abstract_label_of_option('./option/211021.3.ini')


