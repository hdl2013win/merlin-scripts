
import cPickle
import gzip
import os, sys, errno
import time
import math

print sys.path
sys.path.append("../../../src")

import subprocess
import socket # only for socket.getfqdn()

#  numpy & theano imports need to be done in this order (only for some numpy installations, not sure why)
import numpy
#import gnumpy as gnp
# we need to explicitly import this in some cases, not sure why this doesn't get imported with numpy itself
import numpy.distutils.__config__
# and only after that can we import theano 
import theano

from utils.providers import ListDataProvider

from frontend.label_normalisation import HTSLabelNormalisation, XMLLabelNormalisation
from frontend.silence_remover import SilenceRemover
from frontend.silence_remover import trim_silence
from frontend.min_max_norm import MinMaxNormalisation
from frontend.acoustic_composition import AcousticComposition
from frontend.parameter_generation import ParameterGeneration
from frontend.mean_variance_norm import MeanVarianceNorm

# the new class for label composition and normalisation
from frontend.label_composer import LabelComposer
from frontend.label_modifier import HTSLabelModification
from frontend.merge_features import MergeFeat
#from frontend.mlpg_fast import MLParameterGenerationFast

#from frontend.mlpg_fast_layer import MLParameterGenerationFastLayer


import configuration
from models.deep_rnn import DeepRecurrentNetwork

from utils.compute_distortion import DistortionComputation, IndividualDistortionComp
from utils.generate import generate_wav
from utils.learn_rates import ExpDecreaseLearningRate

from io_funcs.binary_io import  BinaryIOCollection

#import matplotlib.pyplot as plt
# our custom logging class that can also plot
#from logplot.logging_plotting import LoggerPlotter, MultipleTimeSeriesPlot, SingleWeightMatrixPlot
from logplot.logging_plotting import LoggerPlotter, MultipleSeriesPlot, SingleWeightMatrixPlot
import logging # as logging
import logging.config
import StringIO


def extract_file_id_list(file_list):
    file_id_list = []
    for file_name in file_list:
        file_id = os.path.basename(os.path.splitext(file_name)[0])
        file_id_list.append(file_id)

    return  file_id_list

def read_file_list(file_name):

    logger = logging.getLogger("read_file_list")

    file_lists = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_lists.append(line)
    fid.close()

    logger.debug('Read file list from %s' % file_name)
    return  file_lists


def make_output_file_list(out_dir, in_file_lists):
    out_file_lists = []

    for in_file_name in in_file_lists:
        file_id = os.path.basename(in_file_name)
        out_file_name = out_dir + '/' + file_id
        out_file_lists.append(out_file_name)

    return  out_file_lists

def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_name_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_name_list.append(file_name)

    return  file_name_list

    
def visualize_dnn(dnn):

    plotlogger = logging.getLogger("plotting")

	# reference activation weights in layers
    W = list(); layer_name = list()
    for i in xrange(len(dnn.params)):
        aa = dnn.params[i].get_value(borrow=True).T
        print   aa.shape, aa.size
        if aa.size > aa.shape[0]:
        	W.append(aa)
        	layer_name.append(dnn.params[i].name)
        	
    ## plot activation weights including input and output
    layer_num = len(W)		
    for i_layer in xrange(layer_num):
		fig_name = 'Activation weights W' + str(i_layer) + '_' + layer_name[i_layer]
		fig_title = 'Activation weights of W' + str(i_layer)
		xlabel = 'Neuron index of hidden layer ' + str(i_layer)
		ylabel = 'Neuron index of hidden layer ' + str(i_layer+1)
		if i_layer == 0:
			xlabel = 'Input feature index'
		if i_layer == layer_num-1:
			ylabel = 'Output feature index'
		logger.create_plot(fig_name, SingleWeightMatrixPlot)
		plotlogger.add_plot_point(fig_name, fig_name, W[i_layer])
		plotlogger.save_plot(fig_name, title=fig_name, xlabel=xlabel, ylabel=ylabel)


def load_covariance(var_file_dict, out_dimension_dict): 
    var = {}
    io_funcs = BinaryIOCollection()
    for feature_name in var_file_dict.keys():
        var_values, dimension = io_funcs.load_binary_file_frame(var_file_dict[feature_name], 1)

        var_values = numpy.reshape(var_values, (out_dimension_dict[feature_name], 1))

        var[feature_name] = var_values

    return  var


def dnn_generation_yuhao(valid_file_list, dnn_model, n_ins, n_outs, out_file_list):
	logger = logging.getLogger("dnn_generation")
	logger.debug('Starting dnn_generation')

	plotlogger = logging.getLogger("plotting")
	file_number = len(valid_file_list)

	for i in xrange(file_number):  #file_number
		logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
		fid_lab = open(valid_file_list[i], 'rb')
		features = numpy.fromfile(fid_lab, dtype=numpy.float32)
		fid_lab.close()
		features = features[:(n_ins * (features.size / n_ins))]
		test_set_x = features.reshape((-1, n_ins))

	#yuhao
	#t0 = time.time()
		#predicted_parameter = dnn_model.parameter_prediction(test_set_x)
	#logger.info("dry predict time : %4f -------------" % (time.time()-t0))
	#t1 = time.time()
		#predicted_parameter = dnn_model.parameter_prediction(test_set_x)
	#logger.info("real predict time : %4f -------------" % (time.time()-t1))

		predicted_parameter = dnn_model.parameter_prediction(test_set_x)

		### write to cmp file
		predicted_parameter = numpy.array(predicted_parameter, 'float32')
		temp_parameter = predicted_parameter
		fid = open(out_file_list[i], 'wb')
		predicted_parameter.tofile(fid)
		logger.debug('saved to %s' % out_file_list[i])
		fid.close()

def main_function_synth(cfg, dnn_model):    

	# get a logger for this main function
	logger = logging.getLogger("main")

	# get another logger to handle plotting duties
	plotlogger = logging.getLogger("plotting")

	# later, we might do this via a handler that is created, attached and configured
	# using the standard config mechanism of the logging module
	# but for now we need to do it manually
	plotlogger.set_plot_path(cfg.plot_dir)

	#### parameter setting########
	hidden_layer_size = cfg.hyper_params['hidden_layer_size']


	####prepare environment

	try:
		file_id_list = read_file_list(cfg.file_id_scp)
		logger.debug('Loaded file id list from %s' % cfg.file_id_scp)
	except IOError:
		# this means that open(...) threw an error
		logger.critical('Could not load file id list from %s' % cfg.file_id_scp)
		raise

	###total file number including training, development, and testing
	total_file_number = len(file_id_list)

	data_dir = cfg.data_dir

	nn_cmp_dir       = os.path.join(data_dir, 'nn' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))
	nn_cmp_norm_dir   = os.path.join(data_dir, 'nn_norm'  + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))

	#model_dir = os.path.join(cfg.work_dir, 'nnets_model')
	gen_dir   = os.path.join(cfg.work_dir, 'gen')    

	in_file_list_dict = {}

	for feature_name in cfg.in_dir_dict.keys():
		in_file_list_dict[feature_name] = prepare_file_path_list(file_id_list, cfg.in_dir_dict[feature_name], cfg.file_extension_dict[feature_name], False)

	nn_cmp_file_list         = prepare_file_path_list(file_id_list, nn_cmp_dir, cfg.cmp_ext)
	nn_cmp_norm_file_list    = prepare_file_path_list(file_id_list, nn_cmp_norm_dir, cfg.cmp_ext)

	###normalisation information
	norm_info_file = os.path.join(data_dir, 'norm_info' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim) + '_' + cfg.output_feature_normalisation + '.dat')

	### normalise input full context label
	# currently supporting two different forms of lingustic features
	# later, we should generalise this 

	if cfg.label_style == 'HTS':
		label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name, add_frame_features=cfg.add_frame_features, subphone_feats=cfg.subphone_feats)
		add_feat_dim = sum(cfg.additional_features.values())
		lab_dim = label_normaliser.dimension + add_feat_dim + cfg.appended_input_dim
		logger.info('Input label dimension is %d' % lab_dim)
		suffix=str(lab_dim)
	# no longer supported - use new "composed" style labels instead
	elif cfg.label_style == 'composed':
		# label_normaliser = XMLLabelNormalisation(xpath_file_name=cfg.xpath_file_name)
		suffix='composed'
	dnn_generation
	if cfg.process_labels_in_work_dir:
		label_data_dir = cfg.work_dir
	else:
		label_data_dir = data_dir

	# the number can be removed
	binary_label_dir      = os.path.join(label_data_dir, 'binary_label_'+str(label_normaliser.dimension))
	nn_label_dir          = os.path.join(label_data_dir, 'nn_no_silence_lab_'+suffix)
	nn_label_norm_dir     = os.path.join(label_data_dir, 'nn_no_silence_lab_norm_'+suffix)

	in_label_align_file_list = prepare_file_path_list(file_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)
	binary_label_file_list   = prepare_file_path_list(file_id_list, binary_label_dir, cfg.lab_ext)
	nn_label_file_list       = prepare_file_path_list(file_id_list, nn_label_dir, cfg.lab_ext)
	nn_label_norm_file_list  = prepare_file_path_list(file_id_list, nn_label_norm_dir, cfg.lab_ext)
	dur_file_list            = prepare_file_path_list(file_id_list, cfg.in_dur_dir, cfg.dur_ext)
	lf0_file_list            = prepare_file_path_list(file_id_list, cfg.in_lf0_dir, cfg.lf0_ext)

	# to do - sanity check the label dimension here?



	min_max_normaliser = None
	label_norm_file = 'label_norm_%s_%d.dat' %(cfg.label_style, lab_dim)
	label_norm_file = os.path.join(label_data_dir, label_norm_file)

	if cfg.GenTestList:
		try:
			test_id_list = read_file_list(cfg.test_id_scp)
			logger.debug('Loaded file id list from %s' % cfg.test_id_scp)
		except IOError:
			# this means that open(...) threw an error
			logger.critical('Could not load file id list from %s' % cfg.test_id_scp)
			raise

		in_label_align_file_list = prepare_file_path_list(test_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)
		binary_label_file_list   = prepare_file_path_list(test_id_list, binary_label_dir, cfg.lab_ext)
		nn_label_file_list       = prepare_file_path_list(test_id_list, nn_label_dir, cfg.lab_ext)
		nn_label_norm_file_list  = prepare_file_path_list(test_id_list, nn_label_norm_dir, cfg.lab_ext)

	if cfg.NORMLAB and (cfg.label_style == 'HTS'):
		# simple HTS labels 
		logger.info('preparing label data (input) using standard HTS style labels')
		label_normaliser.perform_normalisation(in_label_align_file_list, binary_label_file_list, label_type=cfg.label_type)

		if cfg.additional_features:
			out_feat_dir  = os.path.join(data_dir, 'binary_label_'+suffix)
			out_feat_file_list = prepare_file_path_list(file_id_list, out_feat_dir, cfg.lab_ext)
			in_dim = label_normaliser.dimension
			for new_feature, new_feature_dim in cfg.additional_features.iteritems():
				new_feat_dir  = os.path.join(data_dir, new_feature)
				new_feat_file_list = prepare_file_path_list(file_id_list, new_feat_dir, '.'+new_feature)

				merger = MergeFeat(lab_dim = in_dim, feat_dim = new_feature_dim) 
				merger.merge_data(binary_label_file_list, new_feat_file_list, out_feat_file_list)
				in_dim += new_feature_dim

				binary_label_file_list = out_feat_file_list

		remover = SilenceRemover(n_cmp = lab_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type, remove_frame_features = cfg.add_frame_features, subphone_feats = cfg.subphone_feats)
		remover.remove_silence(binary_label_file_list, in_label_align_file_list, nn_label_file_list)

		min_max_normaliser = MinMaxNormalisation(feature_dimension = lab_dim, min_value = 0.01, max_value = 0.99)
		###use only training data to find min-max information, then apply on the whole dataset
		if cfg.GenTestList:
			min_max_normaliser.load_min_max_values(label_norm_file)
		else:
			min_max_normaliser.find_min_max_values(nn_label_file_list[0:cfg.train_file_number])
		### enforce silence such that the normalization runs without removing silence: only for final synthesis
		if cfg.GenTestList and cfg.enforce_silence:
			min_max_normaliser.normalise_data(binary_label_file_list, nn_label_norm_file_list)
		else:
			min_max_normaliser.normalise_data(nn_label_file_list, nn_label_norm_file_list)


	if cfg.NORMLAB and (cfg.label_style == 'composed'):
		# new flexible label preprocessor

		logger.info('preparing label data (input) using "composed" style labels')
		label_composer = LabelComposer()
		label_composer.load_label_configuration(cfg.label_config_file)

		logger.info('Loaded label configuration')
		# logger.info('%s' % label_composer.configuration.labels )

		lab_dim=label_composer.compute_label_dimension()
		logger.info('label dimension will be %d' % lab_dim)

		if cfg.precompile_xpaths:
			label_composer.precompile_xpaths()

		# there are now a set of parallel input label files (e.g, one set of HTS and another set of Ossian trees)
		# create all the lists of these, ready to pass to the label composer

		in_label_align_file_list = {}
		for label_style, label_style_required in label_composer.label_styles.iteritems():
			if label_style_required:
				logger.info('labels of style %s are required - constructing file paths for them' % label_style)
				if label_style == 'xpath':
					in_label_align_file_list['xpath'] = prepare_file_path_list(file_id_list, cfg.xpath_label_align_dir, cfg.utt_ext, False)
				elif label_style == 'hts':
					in_label_align_file_list['hts'] = prepare_file_path_list(file_id_list, cfg.hts_label_align_dir, cfg.lab_ext, False)
				else:
					logger.critical('unsupported label style %s specified in label configuration' % label_style)
					raise Exception

			# now iterate through the files, one at a time, constructing the labels for them 
			num_files=len(file_id_list)
			logger.info('the label styles required are %s' % label_composer.label_styles)

			for i in xrange(num_files):
				logger.info('making input label features for %4d of %4d' % (i+1,num_files))

				# iterate through the required label styles and open each corresponding label file

				# a dictionary of file descriptors, pointing at the required files
				required_labels={}

				for label_style, label_style_required in label_composer.label_styles.iteritems():

					# the files will be a parallel set of files for a single utterance
					# e.g., the XML tree and an HTS label file
					if label_style_required:
						required_labels[label_style] = open(in_label_align_file_list[label_style][i] , 'r')
						logger.debug(' opening label file %s' % in_label_align_file_list[label_style][i])

				logger.debug('label styles with open files: %s' % required_labels)
				label_composer.make_labels(required_labels,out_file_name=binary_label_file_list[i],fill_missing_values=cfg.fill_missing_values,iterate_over_frames=cfg.iterate_over_frames)

				# now close all opened files
				for fd in required_labels.itervalues():
					fd.close()


		# silence removal
		if cfg.remove_silence_using_binary_labels:
			silence_feature = 0 ## use first feature in label -- hardcoded for now
			logger.info('Silence removal from label using silence feature: %s'%(label_composer.configuration.labels[silence_feature]))
			logger.info('Silence will be removed from CMP files in same way')
			## Binary labels have 2 roles: both the thing trimmed and the instructions for trimming: 
			trim_silence(binary_label_file_list, nn_label_file_list, lab_dim, \
								binary_label_file_list, lab_dim, silence_feature)
		else:
			logger.info('No silence removal done')
			# start from the labels we have just produced, not trimmed versions
			nn_label_file_list = binary_label_file_list

		min_max_normaliser = MinMaxNormalisation(feature_dimension = lab_dim, min_value = 0.01, max_value = 0.99)
		###use only training data to find min-max information, then apply on the whole dataset
		min_max_normaliser.find_min_max_values(nn_label_file_list[0:cfg.train_file_number])
		min_max_normaliser.normalise_data(nn_label_file_list, nn_label_norm_file_list)

	if min_max_normaliser != None and not cfg.GenTestList:
		### save label normalisation information for unseen testing labels
		label_min_vector = min_max_normaliser.min_vector
		label_max_vector = min_max_normaliser.max_vector
		label_norm_info = numpy.concatenate((label_min_vector, label_max_vector), axis=0)

		label_norm_info = numpy.array(label_norm_info, 'float32')
		fid = open(label_norm_file, 'wb')
		label_norm_info.tofile(fid)
		fid.close()
		logger.info('saved %s vectors to %s' %(label_min_vector.size, label_norm_file))


	### make output duration data
	if cfg.MAKEDUR:
		logger.info('creating duration (output) features')
		label_type = cfg.label_type
		feature_type = cfg.dur_feature_type
		label_normaliser.prepare_dur_data(in_label_align_file_list, dur_file_list, label_type, feature_type)


	### make output acoustic data
	if cfg.MAKECMP:
		logger.info('creating acoustic (output) features')
		delta_win = cfg.delta_win #[-0.5, 0.0, 0.5]
		acc_win = cfg.acc_win     #[1.0, -2.0, 1.0]

		acoustic_worker = AcousticComposition(delta_win = delta_win, acc_win = acc_win)
		if 'dur' in cfg.in_dir_dict.keys() and cfg.AcousticModel:
			acoustic_worker.make_equal_frames(dur_file_list, lf0_file_list, cfg.in_dimension_dict)
		acoustic_worker.prepare_nn_data(in_file_list_dict, nn_cmp_file_list, cfg.in_dimension_dict, cfg.out_dimension_dict)

		if cfg.remove_silence_using_binary_labels:
			## do this to get lab_dim:
			label_composer = LabelComposer()
			label_composer.load_label_configuration(cfg.label_config_file)
			lab_dim=label_composer.compute_label_dimension()

			silence_feature = 0 ## use first feature in label -- hardcoded for now
			logger.info('Silence removal from CMP using binary label file') 

			## overwrite the untrimmed audio with the trimmed version:
			trim_silence(nn_cmp_file_list, nn_cmp_file_list, cfg.cmp_dim, 
								binary_label_file_list, lab_dim, silence_feature)

		else: ## back off to previous method using HTS labels:
			remover = SilenceRemover(n_cmp = cfg.cmp_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type, remove_frame_features = cfg.add_frame_features, subphone_feats = cfg.subphone_feats)
			remover.remove_silence(nn_cmp_file_list[0:cfg.train_file_number+cfg.valid_file_number], 
								   in_label_align_file_list[0:cfg.train_file_number+cfg.valid_file_number], 
								   nn_cmp_file_list[0:cfg.train_file_number+cfg.valid_file_number]) # save to itself

	### save acoustic normalisation information for normalising the features back
	var_dir   = os.path.join(data_dir, 'var')
	if not os.path.exists(var_dir):
		os.makedirs(var_dir)

	var_file_dict = {}
	for feature_name in cfg.out_dimension_dict.keys():
		var_file_dict[feature_name] = os.path.join(var_dir, feature_name + '_' + str(cfg.out_dimension_dict[feature_name]))

	### normalise output acoustic data
	if cfg.NORMCMP:
		logger.info('normalising acoustic (output) features using method %s' % cfg.output_feature_normalisation)
		cmp_norm_info = None
		if cfg.output_feature_normalisation == 'MVN':
			normaliser = MeanVarianceNorm(feature_dimension=cfg.cmp_dim)
			###calculate mean and std vectors on the training data, and apply on the whole dataset
			global_mean_vector = normaliser.compute_mean(nn_cmp_file_list[0:cfg.train_file_number], 0, cfg.cmp_dim)
			global_std_vector = normaliser.compute_std(nn_cmp_file_list[0:cfg.train_file_number], global_mean_vector, 0, cfg.cmp_dim)

			normaliser.feature_normalisation(nn_cmp_file_list[0:cfg.train_file_number+cfg.valid_file_number], 
											 nn_cmp_norm_file_list[0:cfg.train_file_number+cfg.valid_file_number])
			cmp_norm_info = numpy.concatenate((global_mean_vector, global_std_vector), axis=0)

		elif cfg.output_feature_normalisation == 'MINMAX':        
			min_max_normaliser = MinMaxNormalisation(feature_dimension = cfg.cmp_dim)
			global_mean_vector = min_max_normaliser.compute_mean(nn_cmp_file_list[0:cfg.train_file_number])
			global_std_vector = min_max_normaliser.compute_std(nn_cmp_file_list[0:cfg.train_file_number], global_mean_vector)

			min_max_normaliser = MinMaxNormalisation(feature_dimension = cfg.cmp_dim, min_value = 0.01, max_value = 0.99)
			min_max_normaliser.find_min_max_values(nn_cmp_file_list[0:cfg.train_file_number])
			min_max_normaliser.normalise_data(nn_cmp_file_list, nn_cmp_norm_file_list)

			cmp_min_vector = min_max_normaliser.min_vector
			cmp_max_vector = min_max_normaliser.max_vector
			cmp_norm_info = numpy.concatenate((cmp_min_vector, cmp_max_vector), axis=0)

		else:
			logger.critical('Normalisation type %s is not supported!\n' %(cfg.output_feature_normalisation))
			raise

		cmp_norm_info = numpy.array(cmp_norm_info, 'float32')
		fid = open(norm_info_file, 'wb')
		cmp_norm_info.tofile(fid)
		fid.close()
		logger.info('saved %s vectors to %s' %(cfg.output_feature_normalisation, norm_info_file))

		feature_index = 0
		for feature_name in cfg.out_dimension_dict.keys():
			feature_std_vector = numpy.array(global_std_vector[:,feature_index:feature_index+cfg.out_dimension_dict[feature_name]], 'float32')

			fid = open(var_file_dict[feature_name], 'w')
			feature_var_vector = feature_std_vector**2
			feature_var_vector.tofile(fid)
			fid.close()

			logger.info('saved %s variance vector to %s' %(feature_name, var_file_dict[feature_name]))

			feature_index += cfg.out_dimension_dict[feature_name]

	train_x_file_list = nn_label_norm_file_list[0:cfg.train_file_number]
	train_y_file_list = nn_cmp_norm_file_list[0:cfg.train_file_number]
	valid_x_file_list = nn_label_norm_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number]    
	valid_y_file_list = nn_cmp_norm_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number]
	test_x_file_list  = nn_label_norm_file_list[cfg.train_file_number+cfg.valid_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]    
	test_y_file_list  = nn_cmp_norm_file_list[cfg.train_file_number+cfg.valid_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

	### generate parameters from DNN
	temp_dir_name = '%s_%s_%d_%d_%d_%d_%d_%d_%d' \
					%(cfg.combined_model_name, cfg.combined_feature_name, int(cfg.do_post_filtering), \
					  cfg.train_file_number, lab_dim, cfg.cmp_dim, \
					  len(hidden_layer_size), hidden_layer_size[0], hidden_layer_size[-1])
	gen_dir = os.path.join(gen_dir, temp_dir_name)

	gen_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
	test_x_file_list  = nn_label_norm_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]    

	if cfg.GenTestList:
		gen_file_id_list = test_id_list
		test_x_file_list = nn_label_norm_file_list
		### comment the below line if you don't want the files in a separate folder
		gen_dir = cfg.test_synth_dir

	if cfg.DNNGEN:
		logger.info('generating from DNN')

		try:
			os.makedirs(gen_dir)
		except OSError as e:
			if e.errno == errno.EEXIST:
				# not an error - just means directory already exists
				pass
			else:
				logger.critical('Failed to create generation directory %s' % gen_dir)
				logger.critical(' OS error was: %s' % e.strerror)
				raise

		gen_file_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.cmp_ext)
		#dnn_generation(test_x_file_list, nnets_file_name, lab_dim, cfg.cmp_dim, gen_file_list)
		dnn_generation_yuhao(test_x_file_list, dnn_model, lab_dim, cfg.cmp_dim, gen_file_list)


		logger.debug('denormalising generated output using method %s' % cfg.output_feature_normalisation)

		fid = open(norm_info_file, 'rb')
		cmp_min_max = numpy.fromfile(fid, dtype=numpy.float32)
		fid.close()
		cmp_min_max = cmp_min_max.reshape((2, -1))
		cmp_min_vector = cmp_min_max[0, ] 
		cmp_max_vector = cmp_min_max[1, ]

		if cfg.output_feature_normalisation == 'MVN':
			denormaliser = MeanVarianceNorm(feature_dimension = cfg.cmp_dim)
			denormaliser.feature_denormalisation(gen_file_list, gen_file_list, cmp_min_vector, cmp_max_vector)

		elif cfg.output_feature_normalisation == 'MINMAX':
			denormaliser = MinMaxNormalisation(cfg.cmp_dim, min_value = 0.01, max_value = 0.99, min_vector = cmp_min_vector, max_vector = cmp_max_vector)
			denormaliser.denormalise_data(gen_file_list, gen_file_list)
		else:
			logger.critical('denormalising method %s is not supported!\n' %(cfg.output_feature_normalisation))
			raise

		if cfg.AcousticModel:
			##perform MLPG to smooth parameter trajectory
			## lf0 is included, the output features much have vuv. 
			generator = ParameterGeneration(gen_wav_features = cfg.gen_wav_features, enforce_silence = cfg.enforce_silence)
			generator.acoustic_decomposition(gen_file_list, cfg.cmp_dim, cfg.out_dimension_dict, cfg.file_extension_dict, var_file_dict, do_MLPG=cfg.do_MLPG, cfg=cfg)    

		if cfg.DurationModel:
			### Perform duration normalization(min. state dur set to 1) ### 
			gen_dur_list   = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.dur_ext)
			gen_label_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.lab_ext)
			in_gen_label_align_file_list = prepare_file_path_list(gen_file_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)

			generator = ParameterGeneration(gen_wav_features = cfg.gen_wav_features)
			generator.duration_decomposition(gen_file_list, cfg.cmp_dim, cfg.out_dimension_dict, cfg.file_extension_dict)

			label_modifier = HTSLabelModification(silence_pattern = cfg.silence_pattern, label_type = cfg.label_type)
			label_modifier.modify_duration_labels(in_gen_label_align_file_list, gen_dur_list, gen_label_list)


	### generate wav
	if cfg.GENWAV:
		logger.info('reconstructing waveform(s)')
		generate_wav(gen_dir, gen_file_id_list, cfg)     # generated speech
	#    	generate_wav(nn_cmp_dir, gen_file_id_list, cfg)  # reference copy synthesis speech

	### setting back to original conditions before calculating objective scores ###
	if cfg.GenTestList:
		in_label_align_file_list = prepare_file_path_list(file_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)
		binary_label_file_list   = prepare_file_path_list(file_id_list, binary_label_dir, cfg.lab_ext)
		gen_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

	### evaluation: RMSE and CORR for duration       
	if cfg.CALMCD and cfg.DurationModel:
		logger.info('calculating MCD')

		ref_data_dir = os.path.join(data_dir, 'ref_data')

		ref_dur_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.dur_ext)

		in_gen_label_align_file_list = in_label_align_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
		calculator = IndividualDistortionComp()

		valid_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number]
		test_file_id_list  = file_id_list[cfg.train_file_number+cfg.valid_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

		if cfg.remove_silence_using_binary_labels:
			untrimmed_reference_data = in_file_list_dict['dur'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
			trim_silence(untrimmed_reference_data, ref_dur_list, cfg.dur_dim, \
								untrimmed_test_labels, lab_dim, silence_feature)
		else:
			remover = SilenceRemover(n_cmp = cfg.dur_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type, remove_frame_features = cfg.add_frame_features)
			remover.remove_silence(in_file_list_dict['dur'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_dur_list)

		valid_dur_rmse, valid_dur_corr = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.dur_ext, cfg.dur_dim)
		test_dur_rmse, test_dur_corr = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.dur_ext, cfg.dur_dim)

		logger.info('Develop: DNN -- RMSE: %.3f frames/phoneme; CORR: %.3f; ' \
					%(valid_dur_rmse, valid_dur_corr))
		logger.info('Test: DNN -- RMSE: %.3f frames/phoneme; CORR: %.3f; ' \
					%(test_dur_rmse, test_dur_corr))

	### evaluation: calculate distortion        
	if cfg.CALMCD and cfg.AcousticModel:
		logger.info('calculating MCD')

		ref_data_dir = os.path.join(data_dir, 'ref_data')

		ref_mgc_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.mgc_ext)
		ref_bap_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.bap_ext)
		ref_lf0_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.lf0_ext)

		in_gen_label_align_file_list = in_label_align_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
		calculator = IndividualDistortionComp()

		spectral_distortion = 0.0
		bap_mse             = 0.0
		f0_mse              = 0.0
		vuv_error           = 0.0

		valid_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number]
		test_file_id_list  = file_id_list[cfg.train_file_number+cfg.valid_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

		if cfg.remove_silence_using_binary_labels:
			## get lab_dim:
			label_composer = LabelComposer()
			label_composer.load_label_configuration(cfg.label_config_file)
			lab_dim=label_composer.compute_label_dimension()

			## use first feature in label -- hardcoded for now
			silence_feature = 0

			## Use these to trim silence:
			untrimmed_test_labels = binary_label_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]    


		if cfg.in_dimension_dict.has_key('mgc'):
			if cfg.remove_silence_using_binary_labels:
				untrimmed_reference_data = in_file_list_dict['mgc'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
				trim_silence(untrimmed_reference_data, ref_mgc_list, cfg.mgc_dim, \
									untrimmed_test_labels, lab_dim, silence_feature)
			else:
				remover = SilenceRemover(n_cmp = cfg.mgc_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
				remover.remove_silence(in_file_list_dict['mgc'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_mgc_list)
			valid_spectral_distortion = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.mgc_ext, cfg.mgc_dim)
			test_spectral_distortion  = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.mgc_ext, cfg.mgc_dim)
			valid_spectral_distortion *= (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD
			test_spectral_distortion  *= (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD


		if cfg.in_dimension_dict.has_key('bap'):
			if cfg.remove_silence_using_binary_labels:
				untrimmed_reference_data = in_file_list_dict['bap'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
				trim_silence(untrimmed_reference_data, ref_bap_list, cfg.bap_dim, \
									untrimmed_test_labels, lab_dim, silence_feature)
			else:
				remover = SilenceRemover(n_cmp = cfg.bap_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
				remover.remove_silence(in_file_list_dict['bap'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_bap_list)
			valid_bap_mse        = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.bap_ext, cfg.bap_dim)
			test_bap_mse         = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.bap_ext, cfg.bap_dim)
			valid_bap_mse = valid_bap_mse / 10.0    ##Cassia's bap is computed from 10*log|S(w)|. if use HTS/SPTK style, do the same as MGC
			test_bap_mse  = test_bap_mse / 10.0    ##Cassia's bap is computed from 10*log|S(w)|. if use HTS/SPTK style, do the same as MGC

		if cfg.in_dimension_dict.has_key('lf0'):
			if cfg.remove_silence_using_binary_labels:
				untrimmed_reference_data = in_file_list_dict['lf0'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
				trim_silence(untrimmed_reference_data, ref_lf0_list, cfg.lf0_dim, \
									untrimmed_test_labels, lab_dim, silence_feature)
			else:
				remover = SilenceRemover(n_cmp = cfg.lf0_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
				remover.remove_silence(in_file_list_dict['lf0'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_lf0_list)
			valid_f0_mse, valid_f0_corr, valid_vuv
	#    if gnp._boardId is not None:
	#        import gpu_lock
	#        gpu_lock.free_lock(gnp._boardId)_error   = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.lf0_ext, cfg.lf0_dim)
			test_f0_mse , test_f0_corr, test_vuv_error    = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.lf0_ext, cfg.lf0_dim)

		logger.info('Develop: DNN -- MCD: %.3f dB; BAP: %.3f dB; F0:- RMSE: %.3f Hz; CORR: %.3f; VUV: %.3f%%' \
					%(valid_spectral_distortion, valid_bap_mse, valid_f0_mse, valid_f0_corr, valid_vuv_error*100.))
		logger.info('Test   : DNN -- MCD: %.3f dB; BAP: %.3f dB; F0:- RMSE: %.3f Hz; CORR: %.3f; VUV: %.3f%%' \
					%(test_spectral_distortion , test_bap_mse , test_f0_mse , test_f0_corr, test_vuv_error*100.))

def throw_main_log_message():
	# get a logger for this main function
	logger = logging.getLogger("main")
	
	logger.info('Installation information:')
	logger.info('  Merlin directory: '+os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
	logger.info('  PATH:')
	env_PATHs = os.getenv('PATH')
	if env_PATHs:
		env_PATHs = env_PATHs.split(':')
		for p in env_PATHs:
			if len(p)>0: logger.info('      '+p)
	logger.info('  LD_LIBRARY_PATH:')
	env_LD_LIBRARY_PATHs = os.getenv('LD_LIBRARY_PATH')
	if env_LD_LIBRARY_PATHs:
		env_LD_LIBRARY_PATHs = env_LD_LIBRARY_PATHs.split(':')
		for p in env_LD_LIBRARY_PATHs:
			if len(p)>0: logger.info('      '+p)
	logger.info('  Python version: '+sys.version.replace('\n',''))
	logger.info('    PYTHONPATH:')
	env_PYTHONPATHs = os.getenv('PYTHONPATH')
	if env_PYTHONPATHs:
		env_PYTHONPATHs = env_PYTHONPATHs.split(':')
		for p in env_PYTHONPATHs:
			if len(p)>0:
				logger.info('      '+p)
	logger.info('  Numpy version: '+numpy.version.version)
	logger.info('  Theano version: '+theano.version.version)
	logger.info('    THEANO_FLAGS: '+os.getenv('THEANO_FLAGS'))
	logger.info('    device: '+theano.config.device)

	# Check for the presence of git
	ret = os.system('git status > /dev/null')
	if ret==0:
		logger.info('  Git is available in the working directory:')
		git_describe = subprocess.Popen(['git', 'describe', '--tags', '--always'], stdout=subprocess.PIPE).communicate()[0][:-1]
		logger.info('    Merlin version: '+git_describe)
		git_branch = subprocess.Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=subprocess.PIPE).communicate()[0][:-1]
		logger.info('    branch: '+git_branch)
		git_diff = subprocess.Popen(['git', 'diff', '--name-status'], stdout=subprocess.PIPE).communicate()[0]
		git_diff = git_diff.replace('\t',' ').split('\n')
		logger.info('    diff to Merlin version:')
		for filediff in git_diff:
			if len(filediff)>0: logger.info('      '+filediff)
		#logger.info('      (all diffs logged in '+os.path.basename(cfg.log_file)+'.gitdiff'+')')  # comment by yuhao
		#os.system('git diff > '+cfg.log_file+'.gitdiff') # comment by yuhao

	logger.info('Execution information:')
	logger.info('  HOSTNAME: '+socket.getfqdn())
	logger.info('  USER: '+os.getenv('USER'))
	logger.info('  PID: '+str(os.getpid()))
	PBS_JOBID = os.getenv('PBS_JOBID')
	if PBS_JOBID:
		logger.info('  PBS_JOBID: '+PBS_JOBID)
		


def load_nnets_models(cfg_list):

	nnets_model_list = []
	for i, cfg in enumerate(cfg_list): #load two nnets_models into memory		
		model_dir = os.path.join(cfg.work_dir, 'nnets_model')
		hidden_layer_size = cfg.hyper_params['hidden_layer_size']
		combined_model_arch = str(len(hidden_layer_size))
		for hid_size in hidden_layer_size:
			combined_model_arch += '_' + str(hid_size)
		label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name, add_frame_features=cfg.add_frame_features, subphone_feats=cfg.subphone_feats)
		add_feat_dim = sum(cfg.additional_features.values())
		lab_dim = label_normaliser.dimension + add_feat_dim + cfg.appended_input_dim

		nnets_file_name = '%s/%s_%s_%d_%s_%d.%d.train.%d.%f.rnn.model' \
					  %(model_dir, cfg.combined_model_name, cfg.combined_feature_name, int(cfg.multistream_switch), 
						combined_model_arch, lab_dim, cfg.cmp_dim, cfg.train_file_number, cfg.hyper_params['learning_rate'])

		nnets_model = cPickle.load(open(nnets_file_name, 'rb'))
		print nnets_file_name
		print "__________________________"
		nnets_model_list.append(nnets_model)
	assert(len(nnets_model_list)==2)
	return nnets_model_list


def TTS_labeling(voice_name, TTS_text):
	logger = logging.getLogger("main")
	testDir="./experiments/" + voice_name + "/test_synthesis"
	assert(os.path.isdir(testDir))
	
	subprocess.call(["rm", "-rf", "./"+testDir + "/gen-lab/*"])
	subprocess.call(["rm", "-rf", "./"+testDir + "/prompt-lab/*"])
	subprocess.call(["./experiments/"+ voice_name + "/TTS-Front/main"])
	subprocess.call(["mv", testDir+"/prompt-lab/TTS.lab", testDir + "/prompt-lab/test_synthesis.lab"])
	return NULL


def env_setup(voice_name, global_config_file):
	logger = logging.getLogger("set_env")
	
	logger.info("simulating merlin/src/setup_env.sh...")
	# src/setup_env.sh
	os.environ["PYTHONBIN"] = "python"
	# Basic Theano flags
	os.environ["MERLIN_THEANO_FLAGS"] = "cuda.root=/opt/6.5.19,floatX=float32,on_unused_input=ignore"
	
	logger.info("Running on CPU")
	os.environ["THEANO_FLAGS"] = os.environ["MERLIN_THEANO_FLAGS"]
	
	logger.info("building two python cfg object...")
	config_file1 = "conf/test_dur_synth_" + voice_name + ".conf"
	config_file2 = "conf/test_synth_" + voice_name + ".conf"
	config_file_list = [config_file1, config_file2]
	
	cfg_list = []
	for i in range(2): 
		config_file = config_file_list[i]
		#cfg = configuration.cfg # create an configuration instance
		cfg = configuration.configuration.configuration() # create an configuration instance
		cfg.configure(config_file)
		cfg_list.append(cfg)
	assert(len(cfg_list)==2)
	assert(id(cfg_list[0]) != id(cfg_list[1])) # check pointers are different
	return cfg_list

	
def env_setdown(global_config_file):
	subprocess.call(["./scripts/remove_intermediate_files.sh", global_config_file])

if __name__ == '__main__':
	# set up logging to use our custom class
	logging.setLoggerClass(LoggerPlotter)
	logger = logging.getLogger("main")
	
	
	voice_name = "ht-500-data"
	global_config_file="./conf/global_settings.cfg"
	
	logger.info("Step 0: loading two NN model into memory...")
	cfg_list = env_setup(voice_name, global_config_file)# two configure python object for each NN
	nnets_model_list = load_nnets_models(cfg_list)#  two NN in memory
	
	logger.info("Step 1: creating label files from text...")
	# TODO add python-C API 
	
	TTS_labeling(voice_name, txt)
	
	logger.info("Step 2: synthesizing speech...")
	#throw_main_log_message()
	for i in range(2):
		cfg = cfg_list[i]
		nnets_model = nnets_model_list[i]
		main_function_synth(cfg, nnets_model)

	logger.info("Step 3: deleting intermediate synthesis files...")
	env_setdown(global_config_file)
	
	logger.info("synthesized audio files are in: experiments/" + voice_name + "test_synthesis/wav")
	
	sys.exit(0)
