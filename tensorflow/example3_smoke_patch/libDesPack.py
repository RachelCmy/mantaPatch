import sys,os
examplePath = os.path.dirname(sys.argv[0]) + '/'
if( len(examplePath) == 1 ): examplePath = ''
sys.path.append(examplePath + "../tools")

import numpy as np
import tensorflow as tf

import shutil, sys, time
from manta import *

from MyCnnGraph import * # custom module, build the cnn graph
import datasets # custom module, load data

denFlag = 1
curlFlag = 2

def run(SEED, DIM, PATCH_SIZE, dataFlag, GenCoarseDes, data_dirs, saved_model, desdataPath, updateLib):
	#====== random seeds =======================================
	np.random.seed(SEED)
	tf.set_random_seed(SEED)

	#====== global definition ==================================
	if(dataFlag == denFlag):
		libpath = "file_den"
		libGrid = "/DenG/"
		VecFLAG = 0
		despath = 'libdataD_'
		if(GenCoarseDes):
			despath = 'libdataCD_'
			libGrid = "/BasG/"
	elif(dataFlag == curlFlag):
		libpath = "file_curl"
		libGrid = "/CurG/"
		VecFLAG = (DIM - 2)*2 # curl in 2D is scalar(VecFLAG, 0), curl in 3D is vector(VecFLAG, 2)
		despath = 'libdataD_'
		if(GenCoarseDes):
			despath = 'libdataCD_'
			libGrid = "/BcuG/"
	else: # define others
		libpath = "file_den"
		libGrid = "/DenG/"
		VecFLAG = 0
		despath = 'libdataD_'
		if(GenCoarseDes):
			despath = 'libdataCD_'
			libGrid = "/BasG/"
	# hard coded cnn structure
	CNN_KNLS_N = [5,5,5,3]
	CNN_KNLS_P = [1,1,1,1] #stride in conv
	CNN_MAPS_N = [4,8,16,32]
	POOL_AFT_C = [1,0,1,0]
	FC_INNER_N = [128*(DIM-1)]
	DC_OUTER_N = [FC_INNER_N[-1]*2, 1]

	sess      = tf.InteractiveSession()
	cnnInst   = MyCnnGraph( sess, DIM, 1, 0, PATCH_SIZE, PATCH_SIZE, True, 1.0, 1e-3 / float(60), \
		CNN_KNLS_N, CNN_KNLS_P, CNN_MAPS_N, POOL_AFT_C, FC_INNER_N, DC_OUTER_N, VecFLAG )
	sess.run(tf.global_variables_initializer())
	try:
		cnnInst.loadModel(saved_model)
	except Exception as e:
		print("Model doesn't fit. Try loading one branch from model.")
		cnnInst.loadOneBranchModel(saved_model)

	for data_I in range( len(data_dirs) ):
		loadD = data_dirs[data_I]
		if( not os.path.exists(loadD+libpath) ):
			os.makedirs(loadD+libpath)
		if( not os.path.isfile('%s/%s/libdata_%d.npz' % (loadD, libpath, PATCH_SIZE)) ):
			dataN = datasets.pack_lib_index(loadD,libpath, [], PATCH_SIZE)
			
		cnndata = np.load('%s/%s/libdata_%d.npz'% (loadD, libpath, PATCH_SIZE))
		totalnum = cnndata['id'].shape[0]
		
		desHolder = []
		desLen = []
		inputBufferList = []
		
		print(totalnum)
		for dataI in range (totalnum):
			if( dataI % int(totalnum/100) == 0 ):
				sys.stdout.write('\r')
				sys.stdout.write("percent %2.2f" % (dataI*100.0/totalnum))
				sys.stdout.flush()
			curpid = cnndata['id'][dataI]
			gd_filepath = '%s%sP%02d/P%02d_%03d.uni'%(loadD,libGrid,int(curpid/50), \
				int(curpid%50), int(cnndata['fr'][dataI]))
			head, content = datasets.readUni_fixX(gd_filepath, PATCH_SIZE)
			inputBufferList.append(content)
			
			if(len(inputBufferList) >= 60 or (dataI == totalnum-1) ):
				cnnBuffer = np.nan_to_num(inputBufferList)
				if(GenCoarseDes):
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
				del content
				del desDataBuffer
				del inputBufferList
				inputBufferList = []
		
		desLen = np.array(desLen, dtype=np.float32)
		desHolder = np.array(desHolder, dtype=np.float32)
		
		print (str(desHolder.shape) + " descriptors generated in file %s/_%d_%s_%s%d.npz"% (desdataPath, data_I, libpath, despath,PATCH_SIZE))
		np.savez_compressed('%s/_%d_%s_%s%d.npz'% (desdataPath, data_I, libpath, despath,PATCH_SIZE), des=desHolder, len=desLen)
		if(updateLib):
			print ("Copy file %s/_%d_%s_%s%d.npz to folder %s"% \
				(desdataPath, data_I, libpath, despath,PATCH_SIZE, (loadD+libpath)))
			shutil.copyfile( "%s/_%d_%s_%s%d.npz"% (desdataPath, data_I, libpath, despath,PATCH_SIZE),\
				"%s/%s/%s%d.npz"%(loadD,libpath, despath,PATCH_SIZE) )
		
	cnnInst.sessClose()
	
def main():
	# set params  ----------------------------------------------------------------------#
	setDebugLevel(0)# 1, full log output for debugging, 0 ignore
	nowtstr = time.strftime('%Y%m%d_%H%M%S')
	dirPath = '%s../data/_deslog_%s/' % ( examplePath,nowtstr)
	kwargs = {
		"SEED":42,# random seed
		"DIM":2, # simulation dimension
		"PATCH_SIZE": 36,  # CNN input patch size
		"GenCoarseDes" : False, # True: work on coarse only, False: work on high-res only
		"updateLib" : True, # if True, the extracted descriptor file will be copied to the repository
		"desdataPath" : dirPath, # output folder
		"dataFlag" : denFlag, # density -> dataFlag = 1, curl -> dataFlag = 2
		"saved_model" : examplePath + '/models/model_2D/den_cnn.ckpt',
		# the path of the trained model, for e.g. (3D)
		#  = examplePath + '/models/model_3D/den_cnn.ckpt',
		# Note that density CNN should be used to pack density descriptors (denFlag)
		# and curl CNN should be used to pack curl descriptors (curlFlag)
				
		# need special handling for this
		"data_dirs" : examplePath + '../data/patch_1000/',
		# data_dirs, can have more than one path [path1, path2, path3,...]
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

	dirPath = kwargs["desdataPath"]
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
	# training
	run(**kwargs)
	# end

if __name__ == '__main__':
    main()