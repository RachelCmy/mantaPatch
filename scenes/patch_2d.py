#
# Simple example scene for a 2D simulation with Patches
# Simulation of a buoyant soke density plume with open boundaries at top
# tracking patches, visualized with chessboard
#

from manta import *
import numpy as np
import os

savePatchUni   = False
savePatchNumpy = False

# solver params
res = 100
gs = vec3(res,res * 1.5,1)
s = Solver(name='main', gridSize = gs, dim=2)
s.timestep = 0.25
timings = Timings()

# prepare grids
flags = s.create(FlagGrid)
vel = s.create(MACGrid)
density = s.create(RealGrid)
pressure = s.create(RealGrid)

# patch grids
weiG = s.create(RealGrid)
vec3weiG = s.create(MACGrid)
patchDen = s.create(RealGrid)
patchVel = s.create(MACGrid)
patchDen1 = s.create(RealGrid)

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

pp = s.create(PatchAdvSys, subdiv = 2, baseRes = 18.0, jN = 1)
pp.saveLocalPerCellAcceleration( accSZ = vec3(18,18,1) ) 
# accSZ, the size of the acceleration grid that only need to build once, larger size is preferred

# 3 pat texture examples using numpy array
texID = 2 # 0, const texture, 1, chess board, 2, gray value (inversed) of png image
if(texID == 0):# const texture
	patlist = np.array([1.0] * (36*36), dtype = np.float32)
	texSZ = vec3(36,36,1)
elif(texID == 1): # chess board
	texW = 36
	patlist = np.zeros((texW, texW), dtype = np.float32)
	texSZ = vec3(texW,texW,1)
	for ti in range(texW):
		for tj in range(texW):
			if( (int(5*ti / texW)%2) == (int(5*tj / texW)%2) ):
				patlist[ ti, tj ] = 1.0
			else:
				patlist[ ti, tj ] = 0.2
elif(texID == 2 ): # from an uni/png file
	import matplotlib.image as mpimg
	def rgb2gray(rgb): return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

	img = mpimg.imread('../resources/windings.png')
	patlist = 1.0 - np.array(rgb2gray(img), dtype = np.float32)
	texSZ = vec3(patlist.shape[0],patlist.shape[1],1)

if(savePatchUni):
	# create all necessary directories first!
	if not os.path.exists('../data/'):
		os.makedirs('../data/')
	if not os.path.exists('../data/DenG/'):
		os.makedirs('../data/DenG/')
	if not os.path.exists('../data/CurG/'):
		os.makedirs('../data/CurG/')
	if not os.path.exists('../data/VelG/'):
		os.makedirs('../data/VelG/')
#main loop
for t in range(0, 160):
	mantaMsg('\nFrame %i' % (s.frame))
		
	advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2) 
	advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2, openBounds=True, boundaryWidth=bWidth)
	resetOutflow(flags=flags,real=density) 

	setWallBcs(flags=flags, vel=vel)    
	addBuoyancy(density=density, vel=vel, gravity=vec3(0,-8e-3,0), flags=flags)

	solvePressure(flags=flags, vel=vel, pressure=pressure)
	setWallBcs(flags=flags, vel=vel)    
	densityInflow( flags=flags, density=density, noise=noise, shape=source, scale=1, sigma=0.5 )
	
	# projectPpmFull( density, '../data/den_%04d.ppm'% (t), 0, 0.7 )
	# density.save('../data/den_%04d.uni'% (t))
	# vel.save('../data/vel_%04d.uni'% (t))
	
	if( t >= 60): # patch operations
		pp.AdvectWithControl(lamda = 0.02, flags = flags,vel = vel, integrationMode=IntRK4 )
		if( t == 60):
			pp.sampleCandidates( denG = density, samWidth = 25.0, weiThresh = 2.0, occThresh = 0.5)
			pp.initCandidateCage( denG = density )
			pp.addCandidatePatch()
		
		# acceleration for synthesis functions, 
		# useful when synthesis high resolution or synthesis multiple times
		# tarSZ should be the target resolution we want to synthesis. 
		# acceleration grid is always in size of pp.getParent().getGridSize()
		if(savePatchUni or savePatchNumpy): # multiple times
			pp.synPerCellAcceleration( tarSZ = gs )
		
		patchN = pp.pySize()
		# np.intc is either int64 (NPY_LONG) or int32(NPY_INT)
		# tested with 64bit machine and 64bit mantaflow, todo test with 32 bit...
		patdic = np.intc([0] * patchN) # all patches use the first texture grid in patlist
		
		# texture synthesis
		pp.patchSynthesisReal( tarG = patchDen, patchSZ = texSZ, patchGrids = patlist, \
			patchDict = patdic, withSpaceW = False, weigG = weiG)
		weiG.clamp(1.0, 100000.0)
		patchDen.safeDivide(weiG)
		
		weiG.copyFrom(density)
		weiG.multConst(0.4)
		patchDen.add(weiG)
		
		del patdic
		# projectPpmFull( patchDen, '../data/patDen_%04d.ppm'% (t), 0, 1.0 )
		if(savePatchUni):
			# create all necessary directories first!
			pp.saveLocalPatchGridReal(density, '../data/DenG', False, 1e-5, True)
			pp.saveLocalPatchMACCurl(vel, '../data/CurG', False, 1e-5, True)
			pp.saveLocalPatchMACGrid(vel, '../data/VelG', False, 1e-5, True)
		if(savePatchNumpy):
			# save patch data to numpy, and apply to same region
			patdic2 = np.intc([0] * patchN) 
			patGrids = np.array([0.0] * (18*18*patchN), dtype = np.float32)
			pp.saveLocalPatchNumpyReal(density, patGrids, patdic2)
			# pp.saveLocalPatchNumpyCurl(vel, patGrids, patdic2)
			pp.patchSynthesisReal( tarG = patchDen1, patchSZ = vec3(18,18,1), patchGrids = patGrids, \
				patchDict = patdic2, withSpaceW = False, weigG = weiG)
			weiG.clamp(1.0, 100000.0)
			patchDen1.safeDivide(weiG)
			
			patdic3 = np.intc([0] * patchN) 
			patGrids2 = np.array([0.0] * (18*18*patchN*3), dtype = np.float32)
			pp.saveLocalPatchNumpyMAC(vel, patGrids2, patdic3)
			pp.patchSynthesisMAC( tarG = patchVel, patchSZ = vec3(18,18,1), patchGrids = patGrids2, \
				patchDict = patdic3, withSpaceW = False, weigG = weiG)
			weiG.clamp(1.0, 100000.0)
			copyRealToVec3(weiG, weiG, weiG, vec3weiG)
			patchVel.safeDivide(vec3weiG)
			
			del patdic2
			del patdic3
			del patGrids
			del patGrids2
			
		pp.updateParts() # simply increasing lifet, todo fading
	#timings.display()
	s.step()
