#
# Simple example scene for a 3D simulation with Patches
# Simulation of a buoyant soke density plume with open boundaries at top
# tracking patches, synthesizing with matched repository patches
import sys,os,time
examplePath = os.path.dirname(sys.argv[0]) + '/'
if( len(examplePath) == 1 ): examplePath = ''
sys.path.append(examplePath + "../tools")

import shutil, datetime
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize

from manta import *
from MyCnnGraph import * # custom module, build the cnn graph
import patch_repo as pr
import uniio

def run(dim, SynCurlFlag, anticipatP, dirPath, SEED, data_dirs, den_model, curl_model,PATCH_SIZE, LOGs ):
	steps = 160
	#====== random seeds =======================================
	np.random.seed(SEED)
	tf.set_random_seed(SEED)
	
	sess = tf.InteractiveSession()
	# cnn settings
	data_gnames= [ 'density', 'curl' ] # can be 'velocity-density' as one entry
	des_weight = [0.8660254, 0.5] # norm(des_weight) should be 1
	des_models = [den_model,curl_model]
	matchEmax = 0.8
	holdEmax = 1.2

	BASEP_SIZE = PATCH_SIZE
	DES_LENGTH = 128*(3-1)

	datasetnum = len(data_dirs)
	cnn_num    = len(des_models)
	cnnInstList = [] # build two cnn graph, one for density, one for curl of vel
	for cnnI in range(cnn_num):
		cnnInstList.append(\
			MyCnnGraph( sess, 3, 1, 0, PATCH_SIZE, BASEP_SIZE,\
				True, 1e-3, 1e-5, [5,5,5,3], [1,1,1,1], [4,8,16,32], [1,0,1,0], \
				[DES_LENGTH], [DES_LENGTH*2, 1], (cnnI)*2, 0.0, 0.7 ))
		cnnInstList[cnnI].loadModel(des_models[cnnI]) # load trained model

	lamda = 0.01
	fadT = 30
	aftF = fadT*2 # enough to fade in and fade out
	preF = 0
	if(anticipatP): # fully visible when applying
		preF = fadT
		aftF = fadT
	patchRep = pr.PatchRepo(data_dirs, ['file_den/libdata','file_curl/libdata'], PATCH_SIZE, preF, aftF, des_weight)

	# solver params
	res = 50
	gs = vec3(res,res * 1.5,res)
	s = Solver(name='main', gridSize = gs, dim=3)
	s.timestep = 0.25
	timings = Timings()

	# prepare grids
	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	density = s.create(RealGrid)
	pressure = s.create(RealGrid)

	# patch grids
	xlgs = vec3(res * 4,res * 6,res * 4)
	xl = Solver(name='xl', gridSize = xlgs, dim=3)
	hiDen = xl.create(RealGrid)
	patchDen = xl.create(RealGrid)
	weiG = xl.create(RealGrid)
	vec3weiG = xl.create(MACGrid)
	patchDen1 = xl.create(RealGrid)

	bWidth=1
	flags.initDomain(boundaryWidth=bWidth) 
	flags.fillGrid()

	setOpenBound(flags, bWidth,'Y',FlagOutflow|FlagEmpty) 

	if (GUI):
		gui = Gui()
		gui.show( True ) 
		#gui.pause()

	upz = vec3(0, 0.05, 0)
	cpos = vec3(0.5,0.1,0.5)
	source = s.create(Cylinder, center=gs*cpos, radius=res*0.15, z=gs*upz)
	noise = s.create(NoiseField, fixedSeed=265, loadFromFile=True)
	noise.posScale = vec3(20)
	noise.clamp = True
	noise.clampNeg = 0
	noise.clampPos = 2
	noise.valScale = 1
	noise.valOffset = 0.075
	noise.timeAnim = 0.3

	baseR = 12.0
	pp = s.create(PatchSynSystem, subdiv = 2, baseRes = baseR, jN = 6, anticipate = anticipatP)
	pp.saveLocalPerCellAcceleration( accSZ = vec3(BASEP_SIZE,BASEP_SIZE,1) ) 
	# acceleration for packing numpy arrays

	den_shape = [-1, BASEP_SIZE, BASEP_SIZE, BASEP_SIZE, 1]
	vel_shape = [-1, BASEP_SIZE, BASEP_SIZE, BASEP_SIZE, 3]
	initfadW = 0.0
	if(anticipatP): initfadW = 1.0


	#main loop
	for t in range(0, steps):
		print('Frame %d' % (t))
		
		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2) 
		advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2, openBounds=True, boundaryWidth=bWidth)
		resetOutflow(flags=flags,real=density) 

		setWallBcs(flags=flags, vel=vel)    
		addBuoyancy(density=density, vel=vel, gravity=vec3(0,-8e-3,0), flags=flags)

		solvePressure(flags=flags, vel=vel, pressure=pressure)
		setWallBcs(flags=flags, vel=vel)    
		densityInflow( flags=flags, density=density, noise=noise, shape=source, scale=1, sigma=0.5 )
	
		# projectPpmFull( density, dirPath + 'den_%04d.ppm'% (t), 0, 1.7 )
		if( anticipatP and t >= 40 ): # have to save for backward anticipation
			density.save( dirPath + 'den_%04d.uni'% (t))
			vel.save( dirPath + 'vel_%04d.uni'% (t))
	
		if( t >= 60): # patch operations
			pp.AdvectWithControl(lamda = lamda, flags = flags,vel = vel, integrationMode=IntRK4, scaleLen = True)
			# sample new patches with cube cages
			pp.sampleCandidates( denG = density, samWidth = 16.0, weiThresh = 0.1, occThresh = 0.5)
			pp.initCandidateCage( denG = density )
			pp.addCandidatePatch()
			pp.initNewPatchInfo(initfadW) # initfadW
			# pack local regions for descriptor calculation
			patchN = pp.pySize()
			realNum = 0
			if(patchN > 0):
				patdic2 = np.intc([-1] * patchN)
				denMin = np.array([0.0] * patchN, dtype = np.float32)
				denMax = np.array([1.0] * patchN, dtype = np.float32)
				for cnnI in range(cnn_num):
					cnnBuffer = 0
					if (data_gnames[cnnI]==('velocity')):
						patVGrids = np.array([0.0] * (patchN*BASEP_SIZE*BASEP_SIZE*BASEP_SIZE*3), dtype = np.float32)
						realNum = pp.saveLocalPatchNumpyMAC(vel, patVGrids, patdic2, vec3(BASEP_SIZE,BASEP_SIZE,BASEP_SIZE))
						if(realNum > 0): cnnBuffer = (patVGrids.reshape(vel_shape))[:realNum] # nx36x36x36x3
						del patVGrids
					elif (data_gnames[cnnI]==('curl')):
						patCGrids = np.array([0.0] * (patchN*BASEP_SIZE*BASEP_SIZE*BASEP_SIZE*3), dtype = np.float32)
						realNum = pp.saveLocalPatchNumpyCurl(vel, patCGrids, patdic2, vec3(BASEP_SIZE,BASEP_SIZE,BASEP_SIZE))
						if(realNum > 0): cnnBuffer = (patCGrids.reshape(vel_shape))[:realNum] # nx36x36x36x3 for 3d
						del patCGrids
					elif (data_gnames[cnnI]==('density')):
						patNGrids = np.array([0.0] * (patchN*BASEP_SIZE*BASEP_SIZE*BASEP_SIZE*1), dtype = np.float32)
						realNum = pp.saveLocalPatchNumpyReal(density, patNGrids, patdic2, vec3(BASEP_SIZE,BASEP_SIZE,BASEP_SIZE))
						if(realNum > 0): cnnBuffer = (patNGrids.reshape(den_shape))[:realNum] # nx36x36x36x1
						patNGrids = (patNGrids.reshape([patchN, -1]))[:realNum]
						denMin = np.maximum( np.amin(patNGrids, axis=1), 0) # for scaling...
						denMax = np.maximum( np.amax(patNGrids, axis=1), 0) # for scaling...
						del patNGrids
					if(realNum > 0):
						cnnBuffer = np.nan_to_num(cnnBuffer)
						run_dict = {cnnInstList[cnnI].base_grid: cnnBuffer}
						desDataBuffer = sess.run( cnnInstList[cnnI].l_branch_out , feed_dict= run_dict ) #sp,realNum x FC_INNER_N[-1]
						normDes = normalize(desDataBuffer , axis=1, norm='l2') # shape, realNum x FC_INNER_N[-1]
						if( cnnI == 0): tarDes = np.array(normDes * des_weight[0], dtype=np.float32)
						else: tarDes = np.concatenate([tarDes, normDes*des_weight[cnnI]],axis = len(tarDes.shape)-1)
						del normDes, desDataBuffer,cnnBuffer

			if(realNum > 0):
				old_matchList = np.array([0]*patchN, dtype=np.intc)
				denMinL = np.array([0.0]*patchN, dtype=np.float32)
				denMaxL = np.array([1.0]*patchN, dtype=np.float32)
				new_matchError = np.array([0.0]*patchN, dtype=np.float32)
				pp.getMatchList( getFadOut = True, matchList = old_matchList, tarMin = denMinL, tarMax = denMaxL)
				newmatchList = np.copy(old_matchList)
				timeOutList = patchRep.getNextMatchError(newmatchList, new_matchError, \
					tarDes, patdic2, matchEmax, holdEmax)
				# load repo patches
				patSynGrids = []
				patSynDict  = []
				pnum = 0
				for pi in range(patchN):
					if(newmatchList[pi] >= 0):
						repoPath = patchRep.getPatchPath(newmatchList[pi])
						pHead, pCont = uniio.readUni(repoPath)
						#scale according to min max, denMin, denMax
						gi = patdic2[pi]
						if(gi >= 0 and old_matchList[pi] < 0):
							pmin = np.amin(pCont)
							pmax = np.amax(pCont)
							scalefactor = (pmax - pmin)
							if(scalefactor < 0.01):
								pmax = pmax + 0.005
								pmin = pmin - 0.005
							scalefactor = (denMax[gi] - denMin[gi]) / (pmax - pmin)
							denMinL[pi] = denMin[gi] - scalefactor * pmin
							denMaxL[pi] = scalefactor
						#pCont = (pCont - pmin) / scalefactor * (denMaxL[pi] - denMinL[pi]) + denMinL[pi]
						pCont = pCont * denMaxL[pi] + denMinL[pi]
						
						patSynGrids.append(pCont)
						patSynDict.append(pnum)
						pnum = pnum + 1
					else:
						patSynDict.append(-1)
				patSynGrids = np.array(patSynGrids, dtype = np.float32)
				patSynDict  = np.intc(patSynDict)
                
				pp.setMatchList( newmatchList, new_matchError, denMinL, denMaxL )
				#pp.getMatchList( getFadOut = True, tarMin = denMinL, tarMax = denMaxL)
				pp.removeBad(80.0, timeOutList, density)
				pp.updateFading(1.0/fadT)
				if(anticipatP): pp.anticipateStore(t)
				
				# acceleration for synthesis functions, should be called right before synthesis functions
				pp.synPerCellAcceleration( tarSZ = xlgs )
				pp.patchSynthesisReal( tarG = patchDen1, patchSZ = vec3(pHead['dimX'],pHead['dimY'],pHead['dimZ']), \
					patchGrids = patSynGrids, patchDict = patSynDict, withSpaceW = True, weigG = weiG)
				if( anticipatP): # save for synthesizing continuely
					patchDen1.save( dirPath+'Pden_%04d.uni'% (t) )
					weiG.save(dirPath+'Pwei_%04d.uni'% (t))
				# scale and merge with base
				synthesisScale(patchDen1, weiG, density)
				if(anticipatP):	projectPpmFull( patchDen1, dirPath+'PdenPre_%04d.ppm'% (t), 0, 1.7 )
				else: projectPpmFull( patchDen1, dirPath+'Pden_%04d.ppm'% (t), 0, 1.7 )
				if(LOGs): pp.printLog(dirPath+'Plog_%04d.log'% (t))
				pp.updateParts(compress = True) # simply increasing lifet, remove PNEW flags
			projectPpmFull( density, dirPath+'base_%04d.ppm'% (t), 0, 1.7 )
		#timings.display()
		s.step()

	pp.clearParts()
	if(anticipatP):# patch anticipation
		projectPpmFull( patchDen1, dirPath+'Pden_%04d.ppm'% (t-1), 0, 1.7 )
		t = t - 2
		while ( t >= 40 ):
			print('Anticipating frame %d' % (t))
			density.load( dirPath + 'den_%04d.uni'% (t))
			vel.load( dirPath + 'vel_%04d.uni'% (t + 1))
			if( t >= 60 ):
				patchDen1.load( dirPath+'Pden_%04d.uni'% (t) )
				weiG.load(dirPath+'Pwei_%04d.uni'% (t))
			else:
				patchDen1.setConst(0.0)
				weiG.setConst(0.0)
				
			vel.multConst( vec3(-1.0, -1.0, -1.0 ) )
			pp.anticipateAdd( t + 1.0 )
			patchN = pp.pySize()
			if(patchN <= 0):
				t = t - 1
				continue
			pp.AdvectWithControl(lamda = lamda, flags = flags,vel = vel, integrationMode=IntRK4 )
			pp.updateFading(-1.0/fadT)
			pp.removeBad(maxDefE = 9999999.0, den = density)
			matchList = np.array([0]*patchN, dtype=np.intc)
			denMinL = np.array([0.0]*patchN, dtype=np.float32)
			denMaxL = np.array([1.0]*patchN, dtype=np.float32)
			pp.getMatchList( getFadOut = True, matchList = matchList,tarMin = denMinL, tarMax = denMaxL)
			matchList = matchList - 1 # fading in
			pp.setMatchList( matchList)
			patSynGrids = [] # load repo patches
			patSynDict  = []
			pnum = 0
			for pi in range(patchN):
				if(matchList[pi] >= 0):
					repoPath = patchRep.getPatchPath(matchList[pi])
					pHead, pCont = uniio.readUni(repoPath)
					pCont = pCont * denMaxL[pi] + denMinL[pi]
					patSynGrids.append(pCont)
					patSynDict.append(pnum)
					pnum = pnum + 1
				else:
					patSynDict.append(-1)
			patSynGrids = np.array(patSynGrids, dtype = np.float32)
			patSynDict  = np.intc(patSynDict)

			pp.synPerCellAcceleration( tarSZ = xlgs )
			pp.patchSynthesisReal( tarG = patchDen1, patchSZ = vec3(pHead['dimX'],pHead['dimY'],pHead['dimZ']), \
				patchGrids = patSynGrids, patchDict = patSynDict, withSpaceW = True, weigG = weiG, clear = False)
			if(LOGs): pp.printLog(dirPath+'Plog_back_%04d.log'% (t))
			synthesisScale(patchDen1, weiG, density)
			projectPpmFull( patchDen1, dirPath+'Pden_%04d.ppm'% (t), 0, 1.7 )
			pp.updateParts(compress = True) # simply increasing lifet, remove PNEW flags
			t = t - 1
		
def main():
	# set params  ----------------------------------------------------------------------#
	setDebugLevel(0)
	nowtstr = time.strftime('%Y%m%d_%H%M%S')
	dirPath = '%s../data/_simlog_%s/' % ( examplePath,nowtstr )
	
	kwargs = {
		"dim":3, # dimension
		"SynCurlFlag" : False, # synthesis curl field as well, only for visualization, not used in simulation
		"anticipatP" : True, # do anticipation instead of normal fading in
		"dirPath":dirPath, # output folder
		"SEED":42, # random seed
		"den_model": examplePath + '/models/model_3D/den_cnn.ckpt', 
        # trained density model, should be the same as the repository
		# must be a valid path, for e.g., saved_model 
		#  = "../data/_log_20170622_143004/model_20170622_143004.ckpt"
		"curl_model": examplePath + '/models/model_3D/curl_cnn.ckpt',
        # the curl model
		"PATCH_SIZE": 36, # the CNN patch size
		
		# need special handling for these
		"data_dirs" : "%s, %s" %(
			examplePath+'mantaPatchData/Data_72_1/3D', examplePath+'mantaPatchData/Data_72_2/3D'
			),
		# data_dirs, can have more than one path [path1, path2, path3,...]
		# data_dirs is the repository path list
		"LOGs" : False
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
	dirPath = kwargs["dirPath"]
	if not os.path.exists(dirPath): os.makedirs(dirPath)
	shutil.copyfile( sys.argv[0], '%sscene_%s.py' % (dirPath, nowtstr) )
	shutil.copyfile( examplePath+'MyCnnGraph.py', '%sMyCnnGraph_%s.py'% (dirPath, nowtstr) )
	with open('%sscene_%s.py' % (dirPath, nowtstr), "a") as myfile:
		myfile.write("\n#%s\n" % str(kwargs) )
	# training
	run(**kwargs)

if __name__ == '__main__':
	main()
