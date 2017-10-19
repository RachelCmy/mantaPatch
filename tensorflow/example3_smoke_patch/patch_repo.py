import numpy as np
from pyflann import *

# build a kd-tree for repo patches according to their descriptors
class PatchRepo:
	def __init__(self, data_dirs, des_fnames, PATCH_SIZE, preF, aftF, des_weight):
		self.data_dirs = data_dirs
		self.datasetnum = len(data_dirs)
		# self.repoPatchIDList, size = PatchN * 3, x: libid, y, patchid, z frameID
		# self.repoPatchDesList, size = PatchN * desSizeSUM
		for dirI in range(self.datasetnum):
			cnndata = np.load('%s/%s_%d.npz'% (data_dirs[dirI], des_fnames[0], PATCH_SIZE) )
			libID = np.ones_like(cnndata['id']) * dirI
			curLibID = np.array( [libID, cnndata['id'], cnndata['fr']] , dtype=np.int_) # 3 * N
			if( dirI == 0):
				self.repoPatchIDList = curLibID.transpose()
			else:
				self.repoPatchIDList = np.concatenate([self.repoPatchIDList, curLibID.transpose()])
			cnn_num = len(des_fnames)
			for cnnI in range(cnn_num):
				libnormDes = np.load('%s/%sD_%d.npz'% (data_dirs[dirI], des_fnames[cnnI], PATCH_SIZE))['des']
				if( cnnI == 0):
					curLibDes = np.array(libnormDes, dtype=np.float32) * des_weight[cnnI]
				else:
					curLibDes = np.concatenate([curLibDes, libnormDes * des_weight[cnnI]], \
						axis = len(curLibDes.shape)-1)
			if( dirI == 0):
				self.repoPatchDesList = curLibDes
			else:
				self.repoPatchDesList = np.concatenate([self.repoPatchDesList, curLibDes])
		self.patchN = self.repoPatchDesList.shape[0]
		print("all des size = " + str(self.patchN))
		self.validID = [1] * (self.patchN)
		if(preF >= 1):
			prelID = -1
			prepID = -1
			count = 0
			for tID in range( 0, self.repoPatchIDList.shape[0], 1):
				if( prelID == self.repoPatchIDList[tID][0] and \
					prepID == self.repoPatchIDList[tID][1]):
					count = count + 1
				else:
					count = 0
				if(count <  preF):
					self.validID[tID] = 0
				prelID = self.repoPatchIDList[tID][0]
				prepID = self.repoPatchIDList[tID][1]
		if(aftF >= 1):
			prelID = -1
			prepID = -1
			count = 0
			for tID in range( self.repoPatchIDList.shape[0]-1, -1, -1):
				if( prelID == self.repoPatchIDList[tID][0] and \
					prepID == self.repoPatchIDList[tID][1]):
					count = count + 1
				else:
					count = 0
				if(count <  aftF):
					self.validID[tID] = 0
				#elif(count <  2*aftF): # even longer
				#	self.validID[tID] = 0.5 # not in tree, but ok for next frame
				prelID = self.repoPatchIDList[tID][0]
				prepID = self.repoPatchIDList[tID][1]
				
		self.TreeID = []
		self.TreeDes = []
		for tID in range( 0, self.repoPatchIDList.shape[0], 1):
			if(self.validID[tID] == 1):
				self.TreeID.append(tID)
				self.TreeDes.append(self.repoPatchDesList[tID])
		self.TreeDes = np.array(self.TreeDes)
		print("valid tree des size = " + str(self.TreeDes.shape))
		self.flann = FLANN()
		self.params = self.flann.build_index(self.TreeDes, algorithm="kdtree", trees = 4)
		
	def isValid( self, nmid ):
		return ( self.validID[nmid] == 1)
		
	def matchRepoPatches( self, newDes ):
		newN = newDes.shape[0]
		if(newN > 0):
			matchTreeID, matchError = self.flann.nn_index(newDes, 1, checks=self.params["checks"])
			for nid in range(newN):
				matchTreeID[nid] = self.TreeID[matchTreeID[nid]]
			return matchTreeID, matchError
		return [],[]
		
	def getPatchPath( self, matchID ):
		repoMID = self.repoPatchIDList[matchID]
		return self.data_dirs[repoMID[0]] + '/DenG/P%02d'%(repoMID[1]/50) + \
			'/P%02d'%(repoMID[1]%50) + '_%03d.uni'%(repoMID[2])
			
	def getMatchError( self, tarDes, mid):
		return np.sum( np.square( tarDes - self.repoPatchDesList[mid] ) )
		
	def getNextMatchError( self, matchList, matchError, tarDes, patdic, matchEmax, holdEmax):
		newDes = []
		newDict = []
		timeOutList = []
		patchN = patdic.shape[0]
		for pid in range(patchN):
			gid = patdic[pid]
			mid = matchList[pid]
			nmid = mid + 1
			if(mid == -1): # new ones
				newDes.append( tarDes[gid] )
				newDict.append( pid )
			elif(gid == -1): # is fading out, only move on
				matchList[pid] = nmid
			else: # check match quality here, has next one? still match?
				newE = np.sum( np.square( tarDes[gid] - self.repoPatchDesList[nmid] ) )
				matchList[pid] = nmid
				matchError[pid] = newE
				if( self.validID[nmid] == 0 or newE > holdEmax): timeOutList.append(pid)
		if( len(newDes) > 0 ):
			newDes = np.array(newDes, dtype=np.float32)
			matchTreeID, NmatchError= self.flann.nn_index(newDes,1,checks=self.params["checks"])
			for tid, merror, pid in zip(matchTreeID, NmatchError, newDict):
				if( merror < matchEmax and tid >= 0 and self.TreeID[tid] >= 0): # others remain -1
					mid = self.TreeID[tid]
					matchError[pid] = merror
					matchList[pid] = mid
		return np.int_(timeOutList)