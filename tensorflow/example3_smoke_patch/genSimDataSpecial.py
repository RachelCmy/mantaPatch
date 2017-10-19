#******************************************************************************
#
# repo sim data generation, only has high-res
#
#******************************************************************************
from manta import *
import os, shutil, math, sys, time
import numpy as np

def run(steps, savedata, saveppm, simNo, showGui, basePath, dim, offset, buoy, c_timestep):
	scaleFactor = 4.0
	res = 108
	if (dim == 3): res   = 72
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


	setOpenBound(flags,    bWidth,'yY',FlagOutflow|FlagEmpty) 
	setOpenBound(xl_flags, bWidth,'yY',FlagOutflow|FlagEmpty) 

	# inflow sources ----------------------------------------------------------------------#
	np.random.seed(42)

	# init random density
	noises   = []
	sources  = []
	nseeds   = [265, 485, 672, 11, 143, 53, 320, 519, 84, 26, 398, 592]
	
	noiseN   = len(nseeds)
	cpos = vec3(0.5,0.5,0.5)
	noise_valoff = -0.01
	c_off = 0.4
	rad_rand = 0.035
	if( dim == 3 ):
		noise_valoff = -0.005
		c_off = 0.22
		rad_rand = 0.1
	randoms = np.random.rand(noiseN, 8)
	for nI in range(noiseN):
		noises.append( sm.create(NoiseField, fixedSeed=nseeds[nI], loadFromFile=True) )
		noises[nI].posScale = vec3( res * 0.1 * (randoms[nI][7] + 1) )
		noises[nI].clamp = True
		noises[nI].clampNeg = 0
		noises[nI].clampPos = 1.0
		noises[nI].valScale = 1.0
		noises[nI].valOffset = noise_valoff # some gap
		noises[nI].timeAnim = 0.3
		noises[nI].posOffset = vec3(1.5)
		
		# random offsets
		coff = vec3(c_off) * (vec3( randoms[nI][0], randoms[nI][1], randoms[nI][2] ) - vec3(0.5))
		radius_rand = rad_rand + 0.035 * randoms[nI][3]
		upz = vec3(0.95)+ vec3(0.1) * vec3( randoms[nI][4], randoms[nI][5], randoms[nI][6] )
		if(dim == 2): 
			coff.z = 0.0
			upz.z = 1.0
		if( nI%2 == 0 ):
			if(dim == 3):
				r = 0.95 + 0.1 * randoms[nI][4]
				theta = randoms[nI][5] * math.pi
				phi = randoms[nI][6] * 2.0 * math.pi
				upz = vec3( math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)) * r
			sources.append(xl.create(Cylinder, center=xl_gs*(cpos+coff), radius=xl_gs.x*radius_rand, \
				z=xl_gs*radius_rand*upz))
			print (nI, "Cylinder, centre", xl_gs*(cpos+coff), "radius", xl_gs.x*radius_rand, "upz", upz )
		else:
			sources.append(xl.create(Sphere, center=xl_gs*(cpos+coff), radius=xl_gs.x*radius_rand, scale=upz))
			print (nI, "Sphere, centre", xl_gs*(cpos+coff), "radius", xl_gs.x*radius_rand, "scale", upz )
		
		densityInflow( flags=xl_flags, density=xl_density, noise=noises[nI], shape=sources[nI], scale=1.0, sigma=1.0 )
	if(dim == 2):
		v1pos = vec3(0.6)
		v2pos = vec3(0.35)
		v1pos.z = v2pos.z = 0.5
		velInflow = vec3(0.04, 0.01, 0)
		xl_sourcV1 = xl.create(Sphere, center=xl_gs*v1pos, radius=xl_gs.x*0.1, scale=vec3(1))
		xl_sourcV2 = xl.create(Sphere, center=xl_gs*v2pos, radius=xl_gs.x*0.1, scale=vec3(1))
		xl_sourcV1.applyToGrid( grid=xl_vel , value=(-velInflow*float(xl_gs.x)) )
		xl_sourcV2.applyToGrid( grid=xl_vel , value=(velInflow*float(xl_gs.x)) )
	else:
		Vrandom = np.random.rand(3)
		vtheta = Vrandom[2] * math.pi * 0.5
		v1pos = vec3(0.7 + 0.2 *(Vrandom[0] - 0.5) ) 
		v2pos = vec3(0.3 + 0.2 *(Vrandom[1] - 0.5) ) 
		velInflow = 0.04 * vec3(math.sin(vtheta), math.cos(vtheta), 0)
		for dz in range(1,10,1):
			v1pos.z = v2pos.z = (0.1*dz)
			xl_sourcV1 = xl.create(Sphere, center=xl_gs*v1pos, radius=xl_gs.x*0.18, scale=vec3(1))
			xl_sourcV2 = xl.create(Sphere, center=xl_gs*v2pos, radius=xl_gs.x*0.18, scale=vec3(1))
			xl_sourcV1.applyToGrid( grid=xl_vel , value=(-velInflow*float(xl_gs.x)) )
			xl_sourcV2.applyToGrid( grid=xl_vel , value=(velInflow*float(xl_gs.x)) )

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
		advectSemiLagrange(flags=xl_flags, vel=xl_velTmp, grid=xl_vel, order=2, openBounds=True, boundaryWidth=bWidth)
		setWallBcs(flags=xl_flags, vel=xl_vel)
		addBuoyancy(density=xl_density, vel=xl_vel, gravity=buoy , flags=xl_flags)
		if 1 and ( t< 40 ): 
			vorticityConfinement( vel=xl_vel, flags=xl_flags, strength=0.05 )
		solvePressure(flags=xl_flags, vel=xl_vel, pressure=xl_pressure ,  cgMaxIterFac=1.0, cgAccuracy=0.01 )
		setWallBcs(flags=xl_flags, vel=xl_vel)
		xl_velTmp.copyFrom( xl_vel )
		xl_velTmp.addConst( xl_velOffset )
		if( dim == 2 ):
			xl_vel.multConst( vec3(1.0,1.0,0.0) )
			xl_velTmp.multConst( vec3(1.0,1.0,0.0) )
		advectSemiLagrange(flags=xl_flags, vel=xl_velTmp, grid=xl_density, order=2, openBounds=True, boundaryWidth=bWidth)
		xl_density.clamp(0.0, 2.0)

		# save low and high res
		# save all frames
		if savedata and tcnt>=offset:
			tf = tcnt-offset
			framePath = simPath + 'frame_%04d/' % tf
			os.makedirs(framePath)
			xl_density.save(framePath + 'density_high_%04d_%04d.uni' % (simNo, tf))
			xl_vel.save(framePath + 'vel_high_%04d_%04d.uni' % (simNo, tf))
			if(saveppm):
				projectPpmFull( xl_density, simPath + 'density_high_%04d_%04d.ppm' % (simNo, tf), 0, dim - 1.0 )
		tcnt += 1

		sm.step()
		#gui.screenshot( 'outLibt1_%04d.png' % t )
		#timings.display()

		xl.step()
		t = t+1
	
	return simPath

def main():
	# run params  ----------------------------------------------------------------------#
	examplePath = os.path.dirname(sys.argv[0]) + '/'
	if( len(examplePath) == 1 ): examplePath = ''
	# Scene settings  ---------------------------------------------------------------------#
	setDebugLevel(0)

	kwargs = {
		"steps":200, # simulation length
		"savedata":True, # save simulation data, important!
		"saveppm":False, # images for visualization
		"simNo":7000, # starting ID, will increase automatically to avoid existed ones.
		"showGui": 0, # Qt GUI
		"basePath" : examplePath + '../data/', # output folder
		"dim" : 2, # dimention of the simulation
		"offset" : 60, # ignored first 20 frames
		"buoy" : vec3(0,-1e-3,0), # bouyangcy forces
		"c_timestep" : 0.5, # timesteps
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
