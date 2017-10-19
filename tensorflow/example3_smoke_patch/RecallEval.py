import sys,os
examplePath = os.path.dirname(sys.argv[0]) + '/'
sys.path.append(examplePath + "../tools")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

import shutil, time
from manta import *

from MyCnnGraph import * # custom module, build the cnn graph
import datasets # custom module, load data
denFlag = 1
curlFlag = 2

def run(dirPath, nowtstr, SEED, DIM, saved_model, color, PATCH_SIZE, data_dirs, dataFlag, desdataPath, flog):
	#====== random seeds =======================================
	np.random.seed(SEED)
	tf.set_random_seed(SEED)

	#====== global definition ==================================
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
	BATCH_SIZE = int(80 / (DIM - 1)) # actually not used
	wei_decay  = 1.0 # actually not used
	st_lr      = 1e-3 / float(BATCH_SIZE) # actually not used

	# cnn training parameters
	if(dataFlag == denFlag):
		libpath = "file_den"
		dataPair  =  [['DenG'], ['BasG']]
		VecFLAG = 0
		scaleF = True
		offsetF = True
		offC = 0.5
	elif(dataFlag == curlFlag):
		dataPair  =  [['CurG'], ['BcuG']] 
		libpath = "file_curl"
		VecFLAG = (DIM - 2)*2 # curl in 2D is scalar(VecFLAG, 0), curl in 3D is vector(VecFLAG, 2)
		scaleF = False
		offsetF = False
		offC = 0.0
	else: # other setups, not supported yet
		print("implement other settings please!")
		# dataPair: original data pair [ [high-res], [low-res] ], 
		#           e.g., [['DenG'], ['BasG']], [['CurG'], ['BcuG']]
		#           or even, [['VelG','DenG'], ['BvlG','BasG']], 
		# 2xn,n is all entry # [['VelG','DenG'], ['BvlG','BasG']], [['DenG'], ['BasG']]
		# libpath: place to save temporary npz files (pack all pairs together)
		libpath = "file_den"
		dataPair  =  [['DenG'], ['BasG']]
		VecFLAG = 0
		scaleF = True
		offsetF = True
		offC = 0.5
		

	# build the graph in tensorflow
	sess      = tf.InteractiveSession()
	cnnInst   = MyCnnGraph( sess, DIM, SAIMEFLAG, DCFLAG, PATCH_SIZE, BASEP_SIZE, NoPadding, wei_decay, st_lr, \
			CNN_KNLS_N, CNN_KNLS_P, CNN_MAPS_N, POOL_AFT_C, FC_INNER_N, DC_OUTER_N, VecFLAG )

	# load pre-trained model
	sess.run(tf.global_variables_initializer())
	try:
		cnnInst.loadModel(saved_model)
	except Exception as e:
		print("Model doesn't fit. Try loading one branch from model.")
		cnnInst.loadOneBranchModel(saved_model)

	for data_I in range( len(data_dirs) ):
		loadD = data_dirs[data_I]
		if(flog): flog.write("log, %s\n"%(loadD))
		if( not os.path.exists(loadD+libpath) ):
			os.makedirs(loadD+libpath)
		if( not os.path.isfile('%s/%s/libdata_%d.npz' % (loadD, libpath, PATCH_SIZE)) ):
			dataN = datasets.pack_lib_index(loadD,libpath, dataPair, PATCH_SIZE)
			
		cnndata = np.load('%s/%s/libdata_%d.npz'% (loadD, libpath, PATCH_SIZE))
		desdata_path = [ '%s/_%d_%s_%s%d.npz'% (desdataPath, data_I, libpath, 'libdataD_' ,PATCH_SIZE),\
			'%s/_%d_%s_%s%d.npz'% (desdataPath, data_I, libpath, 'libdataCD_' ,PATCH_SIZE)]
		totalnum = cnndata['id'].shape[0]
		norms = [] # save hi_res and low_res descriptors
		for desI in range(2):
			if( os.path.isfile(desdata_path[desI]) ):
				norms.append( np.load(desdata_path[desI])['des'] )
			else:
				desHolder = []
				desLen = []
				inputBufferList = []
				print("")
				for dataI in range (totalnum):
					if( dataI % int(totalnum/100) == 0 ):
						sys.stdout.write('\r')
						sys.stdout.write("percent %2.2f" % (dataI*100.0/totalnum))
						sys.stdout.flush()
					curpid = cnndata['id'][dataI]
					# todo here support concatenate, as in datasets.py
					gd_filepath = '%s%s/P%02d/P%02d_%03d.uni'%(loadD,dataPair[desI][0],int(curpid/50), \
						int(curpid%50), int(cnndata['fr'][dataI]))
					head, content = datasets.readUni_fixX(gd_filepath, PATCH_SIZE)
					inputBufferList.append(content)
				
					if(len(inputBufferList) >= 60 or (dataI == totalnum-1) ):
						cnnBuffer = np.nan_to_num(inputBufferList)
						if(dataI == 1):
							run_dict = {cnnInst.base_grid: cnnBuffer}
							desDataBuffer = sess.run( cnnInst.l_branch_out , feed_dict= run_dict ) #sp,realNum x
						else:
							run_dict = {cnnInst.high_grid: cnnBuffer}
							desDataBuffer = sess.run( cnnInst.r_branch_out , feed_dict= run_dict ) #sp,realNum x
						del cnnBuffer
									
						for i in range(desDataBuffer.shape[0]):
							normLen = np.linalg.norm(desDataBuffer[i])
							desLen.append(normLen)
							desHolder.append(desDataBuffer[i] / normLen)
						
						del desDataBuffer	
						del inputBufferList
						inputBufferList = []
			
				desLen = np.array(desLen, dtype=np.float32)
				desHolder = np.array(desHolder, dtype=np.float32)
				print("")
				print (str(desHolder.shape) + " descriptors generated in file %s"%desdata_path[desI])
				np.savez_compressed(desdata_path[desI], des=desHolder, len=desLen)
				norms.append( desHolder )

		cnnInst.sessClose()
		totalnum = norms[0].shape[0]
		# draw precision graph
		posDis = np.sum( np.square(norms[0] - norms[1]), axis=1 )
		distanceMap = np.zeros([totalnum, totalnum], dtype='f')
		countRank = np.ones([totalnum], dtype='f')
		print("")
		for desi in range(totalnum):
			if( desi % int(totalnum/100) == 0 ):
				sys.stdout.write('\r')
				sys.stdout.write("working on example %2.2f%%" % (desi*100.0/totalnum))
				sys.stdout.flush()
			for desj in range(totalnum):
				distanceMap[desi][desj] = np.sum( np.square(norms[0][desi] - norms[1][desj]) )
				if(distanceMap[desi][desj] < posDis[desi]):
					countRank[desi] = countRank[desi] + 1
		print("")
		
		if(flog):
			flog.write("ranks\n")
			for rank in countRank:
				flog.write("%d\n"%rank)
			
			flog.write("\n\n\nCounts\ttrue Pos N\tTrue Pos Rate\n")
			flog.write("0\t0\t0\n")

		graphPosX = []
		graphPosY = []
		truePositive = 0
		curCount = 1
		for countI in sorted(countRank):
			if countI > curCount:
				graphPosX.append(curCount)
				percent = 100.0 * truePositive/totalnum
				if(flog): flog.write("%d\t%d\t%f\n" %(curCount, truePositive, percent))
				graphPosY.append(percent)
				curCount = countI
			
			truePositive = truePositive + 1
		# last one
		graphPosX.append(curCount)
		percent = 100.0 * truePositive/totalnum
		if(flog): flog.write("%d\t%d\t%f\n" %(curCount, truePositive, percent))
		graphPosY.append(percent)
		#curCount = countI

		fig = plt.figure(figsize=(6,6))
		plt.plot(graphPosX, graphPosY, c=color)
		plt.ylim( (0,100) )
		plt.xlim( (0,totalnum) )
		plt.show()
		fig.savefig("%sTruePosN_%s.png" % (dirPath, nowtstr))
		plt.close()
		
	if(flog): flog.close()
	
def main():
	# set params  ----------------------------------------------------------------------#
	setDebugLevel(0)
	nowtstr = time.strftime('%Y%m%d_%H%M%S')
	dirPath = '%s../data/_evaltlog_%s/' % ( examplePath,nowtstr)
		
	kwargs = {
		"dirPath":dirPath,# output folder
		"nowtstr":nowtstr,# postfix for the name of log files
		"SEED":42,# random seed
		"DIM":2,# simulation dimension
		"saved_model": examplePath + '/models/model_2D/den_cnn.ckpt',
		# must be a valid path, for e.g., saved_model 
		#  = "../data/_log_20170622_143004/model_20170622_143004.ckpt"
		
		"color" : 'b', # use blue color for the recall curve
		"dataFlag" : denFlag, # density -> dataFlag = 1, curl -> dataFlag = 2
		"PATCH_SIZE": 36,
		
		# need special handling for these
		"data_dirs" : examplePath + '../data/patch_1002/',
		# data_dirs, can have more than one path [path1, path2, path3,...]
		"desdataPath" : dirPath # output path for descriptors
		# if descriptors' npz file already exist, descriptor calculation will be skipped.
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
				print("Unknown parameter %s with value %s is ignored" %(sys.argv[parI*2+1], sys.argv[parI*2+2]) )
				continue
			except ValueError:
				print("Parameter %s is ignored due to a strange input %s." %(sys.argv[parI*2+1], sys.argv[parI*2+2]) )
				continue
	dirPath = kwargs["dirPath"]
	
	# some fixed params handling		
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
	if not os.path.exists(dirPath): os.makedirs(dirPath)
	shutil.copyfile( sys.argv[0], '%sscene_%s.py' % (dirPath, nowtstr) )
	shutil.copyfile( examplePath+'MyCnnGraph.py', '%sMyCnnGraph_%s.py'% (dirPath, nowtstr) )
	with open('%sscene_%s.py' % (dirPath, nowtstr), "a") as myfile:
		myfile.write("\n#%s\n" % str(kwargs) )
	# output logs
	kwargs["flog"] = open('%slog_%s.txt' % (dirPath, nowtstr), 'w') # log file, 0 to disable
	# training
	run(**kwargs)
	if(kwargs["flog"]): kwargs["flog"].close()

if __name__ == '__main__':
    main()
