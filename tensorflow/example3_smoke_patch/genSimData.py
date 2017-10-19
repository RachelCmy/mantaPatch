#******************************************************************************
#
# Double sim data generation
#
#******************************************************************************
from manta import *
import os, shutil, math, sys, time
import numpy as np

def run(steps, savedata, saveppm, simNo, npSeed, showGui, basePath, res, dim, offset, resetN, scaleFactor, buoy, c_timestep, doOpen, doCoarse):
	sm_gs = vec3(res,res,res)
	xl_gs = sm_gs * float(scaleFactor)
	if (dim==2):  xl_gs.z = sm_gs.z = 1  # 2D
	
	xl_buoy = buoy * vec3(1./scaleFactor)
	velOffset    = vec3(0.)
	xl_velOffset = vec3(0.)

	sm = Solver(name='smaller', gridSize = sm_gs, dim=dim)
	sm.timestep = c_timestep

	xl = Solver(name='larger', gridSize = xl_gs, dim=dim)
	xl.timestep = c_timestep

	timings = Timings()

	# Insta params  -----------------------------------------------------------------------#
	bgt     = 60 # the starting frame to record patch data

	# Simulation Grids  -------------------------------------------------------------------#
	flags    = sm.create(FlagGrid)
	vel      = sm.create(MACGrid)
	velTmp   = sm.create(MACGrid)
	density  = sm.create(RealGrid)
	pressure = sm.create(RealGrid)

	xl_flags   = xl.create(FlagGrid)
	xl_vel     = xl.create(MACGrid)
	xl_velTmp  = xl.create(MACGrid)
	xl_blurvel = xl.create(MACGrid)
	xl_density = xl.create(RealGrid)
	xl_blurden = xl.create(RealGrid)
	xl_pressure= xl.create(RealGrid)

	# open boundaries
	bWidth=1
	flags.initDomain(boundaryWidth=bWidth)
	flags.fillGrid()
	xl_flags.initDomain(boundaryWidth=bWidth)
	xl_flags.fillGrid()


	if doOpen:
		setOpenBound(flags,    bWidth,'yY',FlagOutflow|FlagEmpty) 
		setOpenBound(xl_flags, bWidth,'yY',FlagOutflow|FlagEmpty) 

	# inflow sources ----------------------------------------------------------------------#
	if(npSeed>0): np.random.seed(npSeed) # -1 means really random!!

	# init random density
	noise    = []
	sources  = []

	noiseN = 12
	nseeds = np.random.randint(10000,size=noiseN)

	cpos = vec3(0.5,0.5,0.5)

	randoms = np.random.rand(noiseN, 8)
	for nI in range(noiseN):
		noise.append( sm.create(NoiseField, fixedSeed= int(nseeds[nI]), loadFromFile=True) )
		noise[nI].posScale = vec3( res * 0.1 * (randoms[nI][7] + 1) )
		noise[nI].clamp = True
		noise[nI].clampNeg = 0
		noise[nI].clampPos = 1.0
		noise[nI].valScale = 1.0
		noise[nI].valOffset = -0.01 # some gap
		noise[nI].timeAnim = 0.3
		noise[nI].posOffset = vec3(1.5)
		
		# random offsets
		coff = vec3(0.4) * (vec3( randoms[nI][0], randoms[nI][1], randoms[nI][2] ) - vec3(0.5))
		radius_rand = 0.035 + 0.035 * randoms[nI][3]
		upz = vec3(0.95)+ vec3(0.1) * vec3( randoms[nI][4], randoms[nI][5], randoms[nI][6] )
		if(dim == 2): 
			coff.z = 0.0
			upz.z = 1.0
		if( nI%2 == 0 ):
			sources.append(xl.create(Cylinder, center=xl_gs*(cpos+coff), radius=xl_gs.x*radius_rand, \
				z=xl_gs*radius_rand*upz))
		else:
			sources.append(xl.create(Sphere, center=xl_gs*(cpos+coff), radius=xl_gs.x*radius_rand, scale=upz))
			
		print (nI, "centre", xl_gs*(cpos+coff), "radius", xl_gs.x*radius_rand, "other", upz )
		
		densityInflow( flags=xl_flags, density=xl_density, noise=noise[nI], shape=sources[nI], scale=1.0, sigma=1.0 )

	# init random velocity
	Vrandom = np.random.rand(3)
	v1pos = vec3(0.7 + 0.4 *(Vrandom[0] - 0.5) ) #range(0.5,0.9) 
	v2pos = vec3(0.3 + 0.4 *(Vrandom[1] - 0.5) ) #range(0.1,0.5)
	vtheta = Vrandom[2] * math.pi * 0.5
	velInflow = 0.04 * vec3(math.sin(vtheta), math.cos(vtheta), 0)

	if(dim == 2):
		v1pos.z = v2pos.z = 0.5
		xl_sourcV1 = xl.create(Sphere, center=xl_gs*v1pos, radius=xl_gs.x*0.1, scale=vec3(1))
		xl_sourcV2 = xl.create(Sphere, center=xl_gs*v2pos, radius=xl_gs.x*0.1, scale=vec3(1))
		xl_sourcV1.applyToGrid( grid=xl_vel , value=(-velInflow*float(xl_gs.x)) )
		xl_sourcV2.applyToGrid( grid=xl_vel , value=( velInflow*float(xl_gs.x)) )
	elif(dim == 3):
		VrandomMore = np.random.rand(3)
		vtheta2 = VrandomMore[0] * math.pi * 0.5
		vtheta3 = VrandomMore[1] * math.pi * 0.25
		vtheta4 = VrandomMore[2] * math.pi * 0.25
		for dz in [1,2,3,7,8,9]: #range(1,10,1):
			v1pos.z = v2pos.z = (0.1*dz)
			vtheta_xy = vtheta *(1.0 - 0.1*dz ) + vtheta2 * (0.1*dz)
			vtheta_z  = vtheta3 *(1.0 - 0.1*dz ) + vtheta4 * (0.1*dz)
			velInflow = 0.04 * vec3( math.cos(vtheta_z) * math.sin(vtheta_xy), math.cos(vtheta_z) * math.cos(vtheta_xy),  math.sin(vtheta_z))
			xl_sourcV1 = xl.create(Sphere, center=xl_gs*v1pos, radius=xl_gs.x*0.1, scale=vec3(1))
			xl_sourcV2 = xl.create(Sphere, center=xl_gs*v2pos, radius=xl_gs.x*0.1, scale=vec3(1))
			xl_sourcV1.applyToGrid( grid=xl_vel , value=(-velInflow*float(xl_gs.x)) )
			xl_sourcV2.applyToGrid( grid=xl_vel , value=( velInflow*float(xl_gs.x)) )
	
	if(doCoarse):
		blurSig = float(scaleFactor) / 3.544908 # 3.544908 = 2 * sqrt( PI )
		#xl_blurden.copyFrom( xl_density )
		blurRealGrid( xl_density, xl_blurden, sigm = blurSig)
		interpolateGrid( target=density, source=xl_blurden )

		#xl_blurvel.copyFrom( xl_vel )
		blurMacGrid( xl_vel, xl_blurvel, sigm = blurSig)
		interpolateMACGrid( target=vel, source=xl_blurvel )
		vel.multConst( vec3(1./scaleFactor) )

	printBuildInfo()

	# Setup UI ---------------------------------------------------------------------#
	if (showGui and GUI):
		gui=Gui()
		gui.show()
		gui.pause()

	t = 0
	tcnt = 0

	if savedata:
		folderNo = simNo
		pathaddition = 'sim_%04d/' % folderNo
		while os.path.exists(basePath + pathaddition):
			folderNo += 1
			pathaddition = 'sim_%04d/' % folderNo

		simPath = basePath + pathaddition
		print("Using output dir '%s'" % simPath) 
		simNo = folderNo
		os.makedirs(simPath)

	# main loop --------------------------------------------------------------------#
	while t < steps+offset:
		curt = t * sm.timestep
		sys.stdout.write( "Current time t: " + str(curt) +" \n" )
		
		newCentre = calcCenterOfMass(xl_density)
		xl_velOffset = xl_gs*float(0.5) - newCentre
		xl_velOffset = xl_velOffset * (1./ xl.timestep)
		velOffset = xl_velOffset * (1./ float(scaleFactor))
		
		#velOffset = xl_velOffset = vec3(0.0) # re-centering off
		if(dim == 2): xl_velOffset.z = velOffset.z = 0.0
		
		# high res fluid
		advectSemiLagrange(flags=xl_flags, vel=xl_velTmp, grid=xl_vel, order=2, openBounds=doOpen, boundaryWidth=bWidth)
		setWallBcs(flags=xl_flags, vel=xl_vel)
		addBuoyancy(density=xl_density, vel=xl_vel, gravity=buoy , flags=xl_flags)
		if 1 and ( t< offset ): 
			vorticityConfinement( vel=xl_vel, flags=xl_flags, strength=0.05 )
		solvePressure(flags=xl_flags, vel=xl_vel, pressure=xl_pressure ,  cgMaxIterFac=1.0, cgAccuracy=0.01 )
		setWallBcs(flags=xl_flags, vel=xl_vel)
		xl_velTmp.copyFrom( xl_vel )
		xl_velTmp.addConst( xl_velOffset )
		if( dim == 2 ):
			xl_vel.multConst( vec3(1.0,1.0,0.0) )
			xl_velTmp.multConst( vec3(1.0,1.0,0.0) )
		advectSemiLagrange(flags=xl_flags, vel=xl_velTmp, grid=xl_density, order=2, openBounds=doOpen, boundaryWidth=bWidth)
		xl_density.clamp(0.0, 2.0)

		# low res fluid, velocity
		if(doCoarse):
			if( t % resetN == 0) :
				xl_blurvel.copyFrom( xl_vel )
				blurMacGrid( xl_vel, xl_blurvel, sigm = blurSig)
				interpolateMACGrid( target=vel, source=xl_blurvel )
				vel.multConst( vec3(1./scaleFactor) )
			else:
				advectSemiLagrange(flags=flags, vel=velTmp, grid=vel, order=2, openBounds=doOpen, boundaryWidth=bWidth)
				setWallBcs(flags=flags, vel=vel)
				addBuoyancy(density=density, vel=vel, gravity=xl_buoy , flags=flags)
				if 1 and ( t< offset ): 
					vorticityConfinement( vel=vel, flags=flags, strength=0.05/scaleFactor )
				solvePressure(flags=flags, vel=vel, pressure=pressure , cgMaxIterFac=1.0, cgAccuracy=0.01 )
				setWallBcs(flags=flags, vel=vel)

			velTmp.copyFrom(vel)
			velTmp.addConst( velOffset )

			# low res fluid, density
			if( t % resetN == 0) :
				xl_blurden.copyFrom( xl_density )
				blurRealGrid( xl_density, xl_blurden, sigm = blurSig)
				interpolateGrid( target=density, source=xl_blurden )
			else:
				advectSemiLagrange(flags=flags, vel=velTmp, grid=density, order=2, openBounds=doOpen, boundaryWidth=bWidth)
				density.clamp(0.0, 2.0)

		# save low and high res
		# save all frames
		if savedata and tcnt>=offset:
			tf = tcnt-offset
			framePath = simPath + 'frame_%04d/' % tf
			os.makedirs(framePath)
			if(doCoarse):
				density.save(framePath + 'density_low_%04d_%04d.uni' % (simNo, tf))
				vel.save(framePath + 'vel_low_%04d_%04d.uni' % (simNo, tf))
			xl_density.save(framePath + 'density_high_%04d_%04d.uni' % (simNo, tf))
			xl_vel.save(framePath + 'vel_high_%04d_%04d.uni' % (simNo, tf))
			if(saveppm):
				wei = 1.0
				if(dim == 3): wei = 3.0
				projectPpmFull( xl_density, simPath + 'density_high_%04d_%04d.ppm' % (simNo, tf), 0, wei )
				if(doCoarse):
					projectPpmFull( density, simPath + 'density_low_%04d_%04d.ppm' % (simNo, tf), 0, wei )
		tcnt += 1

		sm.step()
		#gui.screenshot( 'outLibt1_%04d.png' % t )
		#timings.display()

		xl.step()
		t = t+1
	
	return simPath

def main():
	# run params  ----------------------------------------------------------------------#
	# Main params  ----------------------------------------------------------------------#
	# debugging
	#steps = 50       # shorter test
	#savedata = False # debug , dont write...
	#showGui  = 1
	#setDebugLevel(1)
	examplePath = os.path.dirname(sys.argv[0]) + '/'
	if( len(examplePath) == 1 ): examplePath = ''
	# Scene settings  ---------------------------------------------------------------------#
	setDebugLevel(0)

	kwargs = {
		"steps":200, # simulation length
		"savedata":True, # save simulation data, important!
		"saveppm":False, # images for visualization
		"simNo":1000, # starting ID, will increase automatically to avoid existed ones.
		"npSeed": -1, # random seed
		"showGui": 0, # Qt GUI
		"basePath" : examplePath + '../data/', # output folder
		"res" : 80, # simulation resolution, a square or a cube
		"dim" : 2, # dimention of the simulation
		"offset" : 20, # ignored first 20 frames
		"resetN" : 40, # reset the coarse one with the high-res one
		"scaleFactor" : 4, # factor between coarse resolution and fine resolution
		"buoy" : vec3(0,-1e-3,0), # bouyangcy forces
		"c_timestep" : 0.5, # timesteps
		"doOpen" : False, # open boundary or not
		"doCoarse" : True # if true, do a pair of simulations, else only do the high-res simulation
	}
	
	# cmd params, if only one, take the first one as npSeed
	if( len (sys.argv) == 2 ):
		try:
			kwargs["npSeed"] = int(sys.argv[1])
		except ValueError:
			print("Parameter npSeed is ignored due to a strange input %s." %(sys.argv[1]) )
			
	elif( len (sys.argv) % 2 == 1 ):
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
	else:
		print("Parameters are all ignored... More information can be found in ReadMe.md" )
				
	print(kwargs)
	sim_path = run(**kwargs)
	# log
	nowtstr = time.strftime('%Y%m%d_%H%M%S')
	file_path = 'scene_%s.py' % nowtstr
	shutil.copyfile( sys.argv[0], '%s%s' % (sim_path, file_path) )	
	with open(sim_path+file_path, "a") as myfile:
		myfile.write("\n#%s\n" % str(kwargs) )
	# end

if __name__ == '__main__':
    main()
