import os, sys, gzip, shutil, time
examplePath = os.path.dirname(sys.argv[0]) + '/'
if( len(examplePath) == 1 ): examplePath = ''
sys.path.append(examplePath + "../tools")
from manta import *
import uniio # custom module, read uni file

def run(simNo = 1000, showGui = 0, basePath = examplePath + '../data/', PATCH_SIZE = 36, savePatchUni = True, savePatchPpm = False, doCoarse = True):
	
	pathaddition = 'sim_%04d/' % simNo

	if os.path.exists(basePath + pathaddition):
		outPath  = basePath + 'patch_%04d/' % simNo
		if not os.path.exists(outPath): 
			os.makedirs(outPath)
		else:
			input_var = input("Path exists! Overwrite(y/n)? : ")
			if(input_var != 'y' and input_var != 'Y'):
				return outPath
		
		if(savePatchUni):
			if not os.path.exists(outPath+"DenG/"): os.makedirs(outPath+"DenG/")
			if not os.path.exists(outPath+"CurG/"): os.makedirs(outPath+"CurG/")
			if(doCoarse):
				if not os.path.exists(outPath+"BasG/"): os.makedirs(outPath+"BasG/")
				if not os.path.exists(outPath+"BcuG/"): os.makedirs(outPath+"BcuG/")
		
		cur_step = 0
		# read setting from first density pair
		den_hi_path = basePath + pathaddition + 'frame_%04d/'%cur_step + \
			'density_high_%04d_%04d.uni' % (simNo, cur_step)
		den_lw_path = basePath + pathaddition + 'frame_%04d/'%cur_step + \
			'density_low_%04d_%04d.uni' % (simNo, cur_step)
		head = uniio.RU_read_header(gzip.open(den_hi_path, 'rb'))
		xl_gs = vec3( head['dimX'], head['dimY'], head['dimZ'] )
		if(doCoarse):
			head = uniio.RU_read_header(gzip.open(den_lw_path, 'rb'))
			gs = vec3( head['dimX'], head['dimY'], head['dimZ'] )
		dim = 3
		if( xl_gs.z == 1 and ( doCoarse == False or gs.z == 1)): dim = 2
		bWidth=1
		if(doCoarse):
			scaleFactor = xl_gs.x / gs.x
			sm = Solver(name='smaller', gridSize = gs, dim=dim)
			sm.timestep = 0.5 # todo get the right timestep
			flags    = sm.create(FlagGrid)
			vel      = sm.create(MACGrid)
			velTmp   = sm.create(MACGrid)
			density  = sm.create(RealGrid)
			flags.initDomain(boundaryWidth=bWidth) 
			flags.fillGrid()
			setOpenBound(flags, bWidth,'Y',FlagOutflow|FlagEmpty) # doesn't matter, always load grids
		xl = Solver(name='larger', gridSize = xl_gs, dim=dim)
		xl.timestep = 0.5 # todo get the right timestep
		
		xl_flags   = xl.create(FlagGrid)
		xl_vel     = xl.create(MACGrid)
		xl_velTmp  = xl.create(MACGrid)
		xl_density = xl.create(RealGrid)
				
		xl_flags.initDomain(boundaryWidth=bWidth) 
		xl_flags.fillGrid()
		setOpenBound(xl_flags, bWidth,'Y',FlagOutflow|FlagEmpty) 
		
		pp = xl.create(PatchAdvSys, subdiv = 2, baseRes = PATCH_SIZE, jN = 1)
		szz = PATCH_SIZE
		if(dim == 2): szz = 1
		pp.saveLocalPerCellAcceleration( accSZ = vec3(PATCH_SIZE,PATCH_SIZE,szz) )
		
		if (showGui and GUI):
			gui=Gui()
			gui.show()
			gui.pause()
			
		while os.path.isfile( den_hi_path ):
			mantaMsg('\nFrame %i' % (cur_step))
			vel_hi_path = basePath + pathaddition + 'frame_%04d/'%cur_step + \
				'vel_high_%04d_%04d.uni' % (simNo, cur_step)
			vel_lw_path = basePath + pathaddition + 'frame_%04d/'%cur_step + \
				'vel_low_%04d_%04d.uni' % (simNo, cur_step)
				
			xl_density.load(den_hi_path)
			xl_vel.load(vel_hi_path)
			newCentre = calcCenterOfMass(xl_density)
			xl_velOffset = xl_gs*float(0.5) - newCentre
			xl_velOffset = xl_velOffset * (1./ xl.timestep)
			# velocity with shifting
			xl_velTmp.copyFrom( xl_vel )
			xl_velTmp.addConst( xl_velOffset )
			if(doCoarse):
				density.load(den_lw_path)
				vel.load(vel_lw_path)
				velOffset = xl_velOffset * (1./ float(scaleFactor))
				velTmp.copyFrom(vel)
				velTmp.addConst( velOffset )
			
			pp.AdvectWithControl(lamda = 0.02 / float (dim - 1.0), flags = xl_flags, vel = xl_velTmp, \
				integrationMode=IntRK4, scaleLen = True )
			if(cur_step % 5 == 0 and pp.pySize() < 150 and cur_step < 100):
				if(doCoarse): # faster to sample on coarse one
					pp.sampleCandidates( denG = density, samWidth = (100./ float(scaleFactor)), \
						weiThresh = 3.0, occThresh = 0.5)
					pp.initCandidateCage( denG = density )
				else:
					pp.sampleCandidates( denG = xl_density, samWidth = 100.0, weiThresh = 3.0, \
						occThresh = 0.5)
					pp.initCandidateCage( denG = xl_density )
				pp.addCandidatePatch()	
			if(savePatchUni):
				if(doCoarse):
					pp.saveLocalPatchGridReal(density, outPath+'BasG/', False, 1e-5, savePatchPpm)
					pp.saveLocalPatchMACCurl(vel, outPath+'BcuG', False, 1e-5, savePatchPpm)
				
				pp.saveLocalPatchGridReal(xl_density, outPath+'DenG', False, 1e-5, savePatchPpm)
				pp.saveLocalPatchMACCurl(xl_vel, outPath+'CurG', False, 1e-5, savePatchPpm)
				
				#pp.saveLocalPatchMACGrid(xl_vel, '../data/VelG', False, 1e-5, savePatchPpm)
			pp.killBad(maxDefE = 9999999.0)
			pp.updateParts(compress = True) # simply increasing lifet, todo fading
			cur_step = cur_step + 1
			if(doCoarse): sm.step()
			xl.step()
			den_hi_path = basePath + pathaddition + 'frame_%04d/'%cur_step + \
				'density_high_%04d_%04d.uni' % (simNo, cur_step)
			den_lw_path = basePath + pathaddition + 'frame_%04d/'%cur_step + \
				'density_low_%04d_%04d.uni' % (simNo, cur_step)
		
		return outPath
	
	else:
		print("simulation folder not found! %s" % (basePath + pathaddition))
		return basePath # anything, whatever, if valid, then there will be a log
		
def main():
	# run params  ----------------------------------------------------------------------#
	setDebugLevel(0)
	kwargs = {
		"simNo":1000, # target simulation ID, if -1, check for all existed simulations
		"showGui": 0, # Qt GUI 
		"basePath" : examplePath + '../data/', # output folder
		"PATCH_SIZE" : 36, # patch size
		"savePatchPpm" : False, # images for visualization
		"savePatchUni" : True, # important! patch file!
		"doCoarse" : True, # if true, work on the pair of simulations, else only do the high-res simulation
	}
	# cmd params, if only one, take the first one as simNo
	if( len (sys.argv) == 2 ):
		try:
			kwargs["simNo"] = int(sys.argv[1])
		except ValueError:
			print("Parameter simNo is ignored due to a strange input %s." %(sys.argv[1]) )
			
	elif( len (sys.argv) % 2 == 1 ):
		totalpar = int( (len (sys.argv)- 1) / 2 )
		for parI in range(totalpar):
			try:
				if(sys.argv[parI*2+2] == "False" and \
					type(kwargs[str(sys.argv[parI*2+1])]).__name__ == 'bool'):
					kwargs[str(sys.argv[parI*2+1])] = False
				else:
					kwargs[str(sys.argv[parI*2+1])] = type(kwargs[str(sys.argv[parI*2+1])])(sys.argv[parI*2+2])
			except KeyError:
				print("Unknown parameter %s with value %s is ignored" %(sys.argv[parI*2+1], sys.argv[parI*2+2]) )
				continue
			except ValueError:
				print("Parameter %s is ignored due to a strange input %s." %(sys.argv[parI*2+1], sys.argv[parI*2+2]) )
				continue
	else:
		print("Parameters are all ignored... More information can be found in ReadMe.md" )
	sim_list = []
	
	if(kwargs["simNo"] != -1):
		sim_list.append(kwargs["simNo"])
	else:
		for root, dirs, files in os.walk(kwargs["basePath"]): 
			for d in sorted(dirs):
				if d.startswith("sim_")<=0: continue
				cursimNo = -1
				try:
					cursimNo = int(d[4:])
				except ValueError:
					continue
				pd = 'patch_%04d/' % cursimNo
				if(cursimNo > 0):
					if(not os.path.exists(os.path.join(root, pd))):
						sim_list.append(cursimNo)
					else:
						print("skip existed simNo %d" % cursimNo )
	
	for sim_id in sim_list:
		kwargs["simNo"] = sim_id
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
