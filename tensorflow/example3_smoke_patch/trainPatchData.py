import sys,os
examplePath = os.path.dirname(sys.argv[0]) + '/'
if( len(examplePath) == 1 ): examplePath = ''
sys.path.append(examplePath + "../tools")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

import shutil, sys, time, traceback
from manta import *

from MyCnnGraph import * # custom module, build the cnn graph
import datasets # custom module, load data

denFlag = 1
curlFlag = 2

def run(dirPath, nowtstr, SEED, DIM, saved_model, PATCH_SIZE,
	BATCH_SIZE, TEST_SZ, test_rate, wei_decay, st_lr,
	data_dirs, dataFlag, trainRange,
	flog):
	#====== random seeds =======================================
	np.random.seed(SEED)
	tf.set_random_seed(SEED)

	#====== parameters' definition ==================================
	# cnn structure, HARD coded, could make it more flexible
	BASEP_SIZE = PATCH_SIZE
	SAIMEFLAG  = 1  # 1 shared weight, 0, seperate weights
	DCFLAG     = 0  # 1, use decision layer, 0 use distance functions
	# -C-R-(P)-C-R-(P)-...-F-
	# conv, relu, pooling, conv, relu, pooling, ... fully connected..
	CNN_KNLS_N = [5,5,5,3]   # 4 conv layers, kernel size array
	CNN_KNLS_P = [1,1,1,1]   # 4 conv layers, kernel stride array
	CNN_MAPS_N = [4,8,16,32] # 4 conv layers, feature maps array
	POOL_AFT_C = [1,0,1,0]   # after each conv layers, whether there is a pooling layer
	FC_INNER_N = [128*(DIM-1)] # after all conv layers, fully connected array
	DC_OUTER_N = [FC_INNER_N[-1]*2, 1] # at last, a decision layer, when DCFLAG == 1; 
	NoPadding  = True
	# cnn training parameters
	if(dataFlag == denFlag):
		tmp_predir= ['file_den', 'file_den']
		dataPair  =  [['DenG'], ['BasG']]
		VecFLAG = 0
		scaleF = True
		offsetF = True
		offC = 0.5
	elif(dataFlag == curlFlag):
		dataPair  =  [['CurG'], ['BcuG']] 
		tmp_predir= ['file_curl', 'file_curl']
		VecFLAG = (DIM - 2)*2 # curl in 2D is scalar(VecFLAG, 0), curl in 3D is vector(VecFLAG, 2)
		scaleF = False
		offsetF = False
		offC = 0.0
	else: # other setups, not supported yet
		print("implement other settings please!")
		#  dataPair: original data pair [ [high-res], [low-res] ], 
		#           e.g., [['DenG'], ['BasG']], [['CurG'], ['BcuG']]
		#           or even, [['VelG','DenG'], ['BvlG','BasG']], 
		# 2xn,n is all entry # [['VelG','DenG'], ['BvlG','BasG']], [['DenG'], ['BasG']]
		# tmp_predir: place to save temporary npz files (pack all pairs together)
		tmp_predir= ['file_den', 'file_den']
		dataPair  =  [['DenG'], ['BasG']]
		VecFLAG = 0
		scaleF = True
		offsetF = True
		offC = 0.5

	# build the graph in tensorflow
	sess      = tf.InteractiveSession()
	cnnInst   = MyCnnGraph( sess, DIM, SAIMEFLAG, DCFLAG, PATCH_SIZE, \
		BASEP_SIZE, NoPadding, wei_decay, st_lr, \
		CNN_KNLS_N, CNN_KNLS_P, CNN_MAPS_N, POOL_AFT_C, \
		FC_INNER_N, DC_OUTER_N, VecFLAG )
	
	sess.run(tf.global_variables_initializer()) # init from random
	# load pre-trained model, if existed
	if ( not ( len(saved_model) == 0)): 
		print("Model %s will be loaded..." % saved_model)
		try:
			cnnInst.loadModel(saved_model)
		except Exception as e:
			print("Model doesn't fit. Try loading one branch from model.")
			cnnInst.loadOneBranchModel(saved_model)
		
	cnnInst.createSumWritter(dirPath)
	sdf_datas = []
	sdf_dataNs = []
	trainCounter = []

	for dataDir,dPre in zip(data_dirs, tmp_predir):
		dataN = 0
		if( not os.path.exists(dataDir+dPre) ):
			os.makedirs(dataDir+dPre)
		if( os.path.isfile('%s/%s/libdata_%d.npz' % (dataDir, dPre, PATCH_SIZE)) ):
			curpath = dataDir + dPre + "/"
			datafiles = [name for name in os.listdir(curpath) if (os.path.isfile(curpath + name) and name.startswith('data_%d_'% PATCH_SIZE))]
			dataN = len(datafiles)
		else:
			dataN = datasets.pack_lib_index(dataDir,dPre, dataPair, PATCH_SIZE)
			
		sdf_dataNs.append( dataN )
		trainCounter.append( 0 )
		sdf_datas.append(datasets.read_packed_data_sets(dataDir, dPre+'/data_%d'% PATCH_SIZE, 0, test_rate, \
			scaleF, offsetF, offC) )

	curid = 0
	data_num = len(sdf_datas)
	lr = st_lr
	i = 0

	#====== Logs ============================================
	if(flog):
		flog.write("Batch Size %d, Test rate %f, Variable Num %d, range %d\n"%\
			(BATCH_SIZE, test_rate, cnnInst.variable_counter, trainRange) )
		flog.write("step\ttrain loss\tnet loss\tmean_pos\tmean_neg\tmean_dis\tevalMean_p\tevalMean_n\tevalMeanDis\n")
		
	# log arrays
	logArray1 = []
	logArray2 = []
	logArray3 = []
	logArray4 = []
	logArray5 = []
	logArray6 = []
	logArray7 = []
	logArray8 = []

	try: # Ctrl+C can stop traning, the last model is still saved
		#====== Training Loop ======================================
		for i in range(trainRange):
			#if( (i+1) % (10*batchRange) == 0 ): lr = lr * 0.8
			if(data_num > 0): curid = i % data_num #curid = (i%1000) % data_num
			sdf_data = sdf_datas[curid]
			
			batch = sdf_data.train.next_labelled_batch(BATCH_SIZE, [1, -1])
			train_dict = {cnnInst.base_grid:batch[0], cnnInst.high_grid: batch[1],\
				cnnInst.y_label: batch[2], cnnInst.learning_rate: lr}
			# outputs
			if (i%200 == 0) or i == (trainRange-1):
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
		
				[train_loss,train_net_mean, train_p_mean, train_n_mean] = \
					sess.run( [cnnInst.loss, cnnInst.net_loss_mean, cnnInst.net_p_mean, cnnInst.net_n_mean], \
					feed_dict = train_dict, options=run_options, run_metadata=run_metadata)
				print("s%d, lr %f, train loss: %g, net loss : %g,\nmean f_s, p:%g, n:%g, dis: %g"%(
					i, lr, train_loss, train_net_mean, train_p_mean, train_n_mean, train_p_mean - train_n_mean))
				
				cnnInst.train_writer.add_run_metadata(run_metadata, 'step%d' % i)
				
				fig_sumlist = []
				fig_sumlist.append( tf.Summary.Value(tag="train/loss_total", simple_value=float(train_loss)) )
				fig_sumlist.append( tf.Summary.Value(tag="train/loss_net", simple_value=float(train_net_mean)) )
				fig_sumlist.append( tf.Summary.Value(tag="train/train_p", simple_value=float(train_p_mean)))
				fig_sumlist.append( tf.Summary.Value(tag="train/train_n", simple_value=float(train_n_mean)))
				fig_sumlist.append( tf.Summary.Value(tag="train/train_dif", simple_value=float(train_p_mean - train_n_mean)))
				
				test_P_batch = sdf_data.test.next_labelled_batch(TEST_SZ, [1,1])
				test_N_batch = sdf_data.test.next_labelled_batch(TEST_SZ, [-1,-1])
				e_p_dict = {cnnInst.base_grid: test_P_batch[0], cnnInst.high_grid: test_P_batch[1]} #
				e_n_dict = {cnnInst.base_grid: test_N_batch[0], cnnInst.high_grid: test_N_batch[1]} #
				[evpDiff, histsump] = sess.run( [cnnInst.netOutputMean, cnnInst.hist_sum_p] , feed_dict= e_p_dict )
				[evnDiff, histsumn] = sess.run( [cnnInst.netOutputMean, cnnInst.hist_sum_n] , feed_dict= e_n_dict )
				print("evaluation, mean : P:%g, N:%g, Dis:%g"%(evpDiff,evnDiff, evpDiff - evnDiff) )
				fig_sumlist.append(tf.Summary.Value(tag="test/test_p", simple_value=float(evpDiff)))
				fig_sumlist.append(tf.Summary.Value(tag="test/test_n", simple_value=float(evnDiff)))
				fig_sumlist.append(tf.Summary.Value(tag="test/test_dis", simple_value=float(evpDiff - evnDiff)))
				for fig_sum in fig_sumlist:
					cnnInst.train_writer.add_summary( tf.Summary(value=[fig_sum]), i )
				
				if(flog):
					flog.write("%d\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n"%(\
						i, train_loss, train_net_mean, train_p_mean, train_n_mean, train_p_mean-train_n_mean, evpDiff, evnDiff,\
						evpDiff - evnDiff) )
								
				cnnInst.histSummaryAdd(i)
				cnnInst.train_writer.add_summary(histsump, i)
				cnnInst.train_writer.add_summary(histsumn, i)
				logArray1.append(train_loss)
				logArray2.append(train_net_mean)
				logArray3.append(evpDiff - evnDiff)
				logArray4.append(train_p_mean - train_n_mean)
				logArray5.append(train_p_mean)
				logArray6.append(train_n_mean)
				logArray7.append(evpDiff)
				logArray8.append(evnDiff)
			# data training
			cnnInst.train_step.run(feed_dict = train_dict)
			
			#if (i%10000 == 0) or i == (trainRange-1):
			#	cnnInst.saveModel( dirPath+"bk", "bk" )
			if ( sdf_dataNs[curid] > 1 and sdf_data.train._epochs_completed >= 60): 
				# train 60 times and move to next, 60 is HARD coded...
				newone = (sdf_datas[curid].train._dataID+1)%sdf_dataNs[curid]
				if( newone == 0): trainCounter[curid] += sdf_data.train._epochs_completed
				sdf_datas[curid] = datasets.read_packed_data_sets(data_dirs[curid], \
					tmp_predir[curid]+'/data_%d'%PATCH_SIZE, newone, test_rate, scaleF, offsetF, offC)
				
				if(flog):
					flog.write( "# data %d moved to next one %d, with %d data"%\
						(curid, newone,sdf_datas[curid].train._num_examples) )
				print("# data %d moved to next one %d, with %d data"%(curid, newone,sdf_datas[curid].train._num_examples))
				
	except Exception as e:
		print("Unexpected error %d:"%i)
		exc_type, exc_value, exc_traceback = sys.exc_info()
		fname = os.path.split(exc_traceback.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, sys.exc_info()[2].tb_lineno)
		print ("*** print_exception:")
		traceback.print_exception(exc_type, exc_value, exc_traceback,limit=2, file=sys.stdout)
		
	except (KeyboardInterrupt, SystemExit):
		print("User Keyboard Stop at %d:"%i)
		
	#====== Export Log, Model, End =============================
	if(flog):
		for curid in range(data_num):
			trainCounter[curid] += sdf_datas[curid].train._epochs_completed
			flog.write("Repeat times: %d\n"%trainCounter[curid])
			print("Repeat times: %d"%trainCounter[curid])
			
	cnnInst.saveModel( dirPath, nowtstr )
	fig, axarr = plt.subplots(2, 2, figsize=(8, 8))
	axarr[0,0].plot(logArray1)
	axarr[0,1].plot(logArray2)
	axarr[1,1].plot(logArray4, color = 'blue')
	axarr[1,1].plot(logArray3, linestyle = '--', color = 'green')
	axarr[1,0].plot(logArray5, color = 'blue')
	axarr[1,0].plot(logArray6, color = 'green')
	axarr[1,0].plot(logArray7, linestyle = '--', color = 'red')
	axarr[1,0].plot(logArray8, linestyle = '--', color = 'orange')

	axarr[0,0].set_title('Total train loss')
	axarr[0,1].set_title('train net loss (1->0)')
	axarr[1,1].set_title('p-n, (0->2)')
	axarr[1,0].set_title('p/n mean, (0->1/0->-1)')

	plt.tight_layout()
	plt.show()
	fig.savefig("%strend_%s.png" % (dirPath, nowtstr))
	cnnInst.sumWritterFlush()		
	cnnInst.sessClose()
	
def main():
	# set params  ----------------------------------------------------------------------#
	setDebugLevel(0)
	nowtstr = time.strftime('%Y%m%d_%H%M%S')
	dirPath = '%s../data/_trainlog_%s/' % ( examplePath,nowtstr)
	
	kwargs = {
		"dirPath":dirPath, # output folder
		"nowtstr":nowtstr, # postfix for the name of log files
		"SEED":42, # random seed
		"DIM":2, # simulation dimension
		"saved_model": "",# empty: a new training
		# a path, continued training, for e.g., saved_model 
		#  = "../data/_log_20170622_143004/model_20170622_143004.ckpt"
		
		"PATCH_SIZE": 36, # 
		"BATCH_SIZE" : int(80), # training parameter batch size
		"TEST_SZ" : int(40),# training parameter test data batch size
		"test_rate" : 0.1, # training parameter test data percentage
		"wei_decay" : 1.0, # training parameter part of the loss function, the weight of networks' weights decay term
		"st_lr" : float( 1e-3 ),# training parameter learning rate
		"dataFlag" : denFlag, # density -> dataFlag = 1, curl -> dataFlag = 2
		"trainRange" : 6000, # 300000 for good result, 6000 for shorter test
		
		# need special handling for this
		"data_dirs" : examplePath + '../data/patch_1000/'
		# data_dirs, training patch datasets, can have more than one path [path1, path2, path3,...]
	}
	
	# update with cmd params
	if( len (sys.argv) % 2 == 1 ):
		totalpar = int( (len (sys.argv)- 1) / 2 )
		for parI in range(totalpar):
			try:
				if(sys.argv[parI*2+2] == "False" and \
					type(kwargs[str(sys.argv[parI*2+1])]).__name__ == 'bool'):
					kwargs[str(sys.argv[parI*2+1])] = False
				else:
					kwargs[str(sys.argv[parI*2+1])] = \
						type(kwargs[str(sys.argv[parI*2+1])])(sys.argv[parI*2+2])
			except KeyError:
				print("Unknown parameter %s with value %s is ignored" \
					%(sys.argv[parI*2+1], sys.argv[parI*2+2]) )
				continue
			except ValueError:
				print("Parameter %s is ignored due to a strange input %s." \
					%(sys.argv[parI*2+1], sys.argv[parI*2+2]) )
				continue
	else:
		print("Parameters are all ignored... More information can be found in ReadMe.md" )

	dirPath = kwargs["dirPath"]
	
	# some fixed params handling
	# normalize
	kwargs["BATCH_SIZE"] = int( kwargs["BATCH_SIZE"] / (int(kwargs["DIM"]) - 1) )
	kwargs["TEST_SZ"] = int ( kwargs["TEST_SZ"] / (int(kwargs["DIM"]) - 1) )
	kwargs["st_lr"] = kwargs["st_lr"] / float(kwargs["BATCH_SIZE"])
	tmplist = kwargs["data_dirs"].split(",")
	kwargs["data_dirs"] = []
	strn = len(tmplist)
	for strs in tmplist:
		curstr = strs.replace("'", "")
		curstr = curstr.replace('"', '')
		curstr = curstr.strip()
		kwargs["data_dirs"].append(curstr)
	print(kwargs)
	
	#====== back up first =======================================	
	if not os.path.exists(dirPath): os.mkdir(dirPath)
	if not os.path.exists('%sboard/'%dirPath): os.mkdir('%sboard/'%dirPath)
	shutil.copyfile( sys.argv[0], '%sscene_%s.py' % (dirPath, nowtstr) )
	shutil.copyfile( examplePath+'MyCnnGraph.py', '%sMyCnnGraph_%s.py'% (dirPath, nowtstr) )
	with open('%sscene_%s.py' % (dirPath, nowtstr), "a") as myfile:
		myfile.write("\n#%s\n" % str(kwargs) )
	# output logs
	kwargs["flog"] = open('%slog_%s.txt' % (dirPath, nowtstr), 'w') # log file, 0 to disable
    # training
	run(**kwargs)
	if(kwargs["flog"]): kwargs["flog"].close()
	# end

if __name__ == '__main__':
    main()