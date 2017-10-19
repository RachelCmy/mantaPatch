#
# Simple example scene for a 3D simulation with Patches
# Simulation of a buoyant soke density plume with open boundaries at top
# tracking patches, visualized with meshes
#

from manta import *
import numpy as np

savePatchUni   = False
savePatchNumpy = False

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
weiG = s.create(RealGrid)
vec3weiG = s.create(MACGrid)
patchDen = s.create(RealGrid) # apply const patch
patchVel = s.create(MACGrid)  # apply patch local data back, validation test
patchDen1 = s.create(RealGrid)# apply patch local data back, validation test

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

pp = s.create(PatchAdvSys, subdiv = 2, baseRes = 6.0, jN = 1)
pp.saveLocalPerCellAcceleration( accSZ = vec3(6,6,6) ) 
# accSZ, the size of the acceleration grid that only need to build once, larger size is preferred

mesh = s.create(Mesh)
patlist = np.array([1.0] * (9*9*9), dtype = np.float32)

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
	
	# projectPpmFull( density, '../data/den_%04d.ppm'% (t), 0, 1.7 )
	# density.save('../data/den_%04d.uni'% (t))
	# vel.save('../data/vel_%04d.uni'% (t))
	
	if( t >= 60): # patch operations
		pp.AdvectWithControl(lamda = 0.01, flags = flags,vel = vel, integrationMode=IntRK4, scaleLen = True)
		if( t == 60):
			pp.sampleCandidates( denG = density, samWidth = 16.0, weiThresh = 2.0, occThresh = 0.5)
			pp.initCandidateCage( denG = density )
			pp.addCandidatePatch()
		pp.meshView(mesh)
		# acceleration for synthesis functions, 
		# useful when synthesis high resolution or synthesis multiple times
		# tarSZ should be the target resolution we want to synthesis. 
		# acceleration grid is always in size of pp.getParent().getGridSize()
		if(savePatchUni or savePatchNumpy): # multiple times
			pp.synPerCellAcceleration( tarSZ = gs )
		patchN = pp.pySize()
		patdic = np.intc([0] * patchN)

		# constant synthesis
		pp.patchSynthesisReal( tarG = patchDen, patchSZ = vec3(9, 9, 9), patchGrids = patlist, \
			patchDict = patdic, withSpaceW = False, weigG = weiG)
		weiG.clamp(1.0, 100000.0)
		patchDen.safeDivide(weiG)
		
		del patdic
		# projectPpmFull( patchDen, '../data/patDen_%04d.ppm'% (t), 0, 1.7 )
		if(savePatchUni):
			# create all necessary directories first!
			pp.saveLocalPatchGridReal(density, '../data/patDen', False, 1e-5, True)
			pp.saveLocalPatchMACCurl(vel, '../data/patCurl', False, 1e-5, True)
			pp.saveLocalPatchMACGrid(vel, '../data/patVel', False, 1e-5, True)
		if(savePatchNumpy):
			# save patch data to numpy, and apply to same region
			patdic2 = np.intc([0] * patchN) 
			patGrids = np.array([0.0] * (18*18*patchN), dtype = np.float32)
			pp.saveLocalPatchNumpyReal(density, patGrids, patdic2)
			pp.patchSynthesisReal( tarG = patchDen1, patchSZ = vec3(6,6,6), patchGrids = patGrids, \
				patchDict = patdic2, withSpaceW = False, weigG = weiG)
			weiG.clamp(1.0, 100000.0)
			patchDen1.safeDivide(weiG)
			
			patdic3 = np.intc([0] * patchN) 
			patGrids2 = np.array([0.0] * (6*6*6*patchN*3), dtype = np.float32)
			pp.saveLocalPatchNumpyMAC(vel, patGrids2, patdic3)
			pp.patchSynthesisMAC( tarG = patchVel, patchSZ = vec3(6,6,6), patchGrids = patGrids2, \
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
