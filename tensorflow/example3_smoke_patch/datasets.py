import numpy as np, os
import uniio
import gzip
from manta import *

def readUni_fixX(filename, len_x):
	with gzip.open(filename, 'rb') as bytestream:
		head = uniio.RU_read_header(bytestream)
		if( len_x <= 0 or int (len_x) == int(head['dimX']) ):
			content = uniio.RU_read_content(bytestream, head)
		else:
			gs1 = vec3(head['dimX'], head['dimY'], head['dimZ'])
			dim = 3
			if( gs1.z == 1): dim = 2
			factor = float(len_x) / float(head['dimX'])
			head['dimX'] = int(len_x)
			head['dimY'] = int(head['dimY']*factor)
			if (head['dimZ'] > 1): head['dimZ'] = int(head['dimZ']*factor)
			if (head['dimZ'] < 1): head['dimZ'] = 1
			gs2 = vec3( head['dimX'], head['dimY'], head['dimZ'])
			if (head['elementType'] == 2): 
				content = np.zeros( (head['dimX'], head['dimY'], head['dimZ'], 3) ,dtype="float32")
			else: 
				content = np.zeros( (head['dimX'], head['dimY'], head['dimZ']) ,dtype="float32")
			numpyGridResize(filename, content, gs1, gs2, head['elementType'] )
			content = np.nan_to_num(content)
			
		if (head['elementType'] == 2):
			if (head['dimZ'] == 1): #2D velocity has zero on Z!!
				content = content.reshape(head['dimX'], head['dimY'], 3, order='C')
				return head, np.array(content[:,:,:-1], dtype="float32")
			else:
				return head, content.reshape(head['dimX'], head['dimY'], head['dimZ'], 3, order='C')
		else:
			if (head['dimZ'] == 1):
				return head, content.reshape(head['dimX'], head['dimY'], 1, order='C')
			else:
				return head, content.reshape(head['dimX'], head['dimY'], head['dimZ'], 1, order='C')

# save the index dictionary, and the grid data (optional)
def pack_lib_index(dir, fname = 'file', des_dirPair = [['DenG'], ['BasG']], lenx = 36):
	# load ids
	print('Reading lib data in ' + dir)
	patIDs = [] # patch ID
	patFrs = [] # patch Frame Num
	patCont = [] # patch Data
	branchN = len(des_dirPair) # should be 2
	data_N = 0
	if(branchN > 0): data_N = len(des_dirPair[0])
	
	for branchI in range(branchN):
		patCont.append([])
		
	dirNum = 0
	onceFlag = True
	dataNum = 0
	tmpSize = 0
	while(1):
		rootDir = '%s%s/P%02d/'%(dir, 'DenG', dirNum)
		print('Reading data in P%02d' % dirNum)
		if not os.path.exists(rootDir): break
		
		for root, dirs, files in os.walk(rootDir): 
			for f in sorted(files):
				if f.endswith(".uni")<=0: continue
				filepath = os.path.join(root, f)
				frameID = int(f[4:7])
				patIDs.append(dirNum * 50 + int(f[1:3]))
				patFrs.append( frameID )
				
				for branchI in range(branchN):
					for dataI in range (data_N):
						hi_filepath = filepath.replace('DenG', des_dirPair[branchI][dataI])
						head, content = readUni_fixX(hi_filepath, lenx)
						
						if(dataI == 0):
							highContent = content
						else:
							caxis = len( content.shape )
							highContent = np.concatenate( (highContent, content) , axis = caxis-1 )
					if(data_N > 0):
						patCont[branchI].append(highContent)
						tmpSize = tmpSize + highContent.nbytes
						
				if( tmpSize > 1e9): # seperate files when larger than 1GB
					tmpSize = 0
					if( branchN == 2):
						perm = np.arange(len(patCont[0]))
						np.random.shuffle(perm)
						np.savez_compressed('%s/%s/data_%d_%d.npz' % (dir, fname, lenx, dataNum),\
							l=np.array(patCont[1], dtype=np.float32), \
							r1=np.array(patCont[0], dtype=np.float32), order=perm)
					#else:
					del patCont
					patCont = []
					for branchI in range(branchN):
						patCont.append([])
					dataNum = dataNum + 1
		dirNum += 1
		
	np.savez_compressed('%s/%s/libdata_%d.npz' % (dir, fname, lenx), id= np.array(patIDs, dtype=np.float32), \
		fr= np.array(patFrs, dtype=np.float32))
	if(data_N > 0):
		if(dataNum > 0 and tmpSize < 5e8): # merge with previous 
			predata = np.load('%s/%s/data_%d_%d.npz' % (dir, fname, lenx, dataNum-1))
			perm = np.arange(len(patCont[0]) + predata['order'].shape[0])
			np.random.shuffle(perm)
			contentl = np.concatenate( (predata['l'], np.array(patCont[1], dtype=np.float32)) )
			contentr = np.concatenate((predata['r1'], np.array(patCont[0], dtype=np.float32)) )
			np.savez_compressed('%s/%s/data_%d_%d.npz' % (dir, fname, lenx, dataNum-1),\
								l=contentl, r1=contentr, order=perm)
		else:	
			perm = np.arange(len(patCont[0]))
			np.random.shuffle(perm)
			np.savez_compressed('%s/%s/data_%d_%d.npz' % (dir, fname, lenx, dataNum),\
								l=np.array(patCont[1], dtype=np.float32), \
								r1=np.array(patCont[0], dtype=np.float32), order=perm)
		# order maybe useless, used to seperate test data and training data
	return dataNum

class DataSet(object):

	def shuffleData(self):
		self._index_in_epoch = int(0)
		self._train_order = np.arange(self._num_examples)
		self._neg_rand_perm = np.arange(self._num_examples)
		np.random.shuffle(self._train_order)
		np.random.shuffle(self._neg_rand_perm)
		niN = self._neg_rand_perm.shape[0]
		for ni in range( niN ): # make sure that neg_pair is not same
			if(self._neg_rand_perm[ni] == ni):
				self._neg_rand_perm[ni] = (ni + 10) % niN
				
	def pre_data_scale(self):
		dim = len(self._lwRess.shape) - 2
		for dataI in range (self._num_examples):
			arraymin = self._lwRess[dataI]
			arraymax = self._lwRess[dataI]
			for i in range(dim):
				arraymin = np.amin(arraymin, axis = 0)
				arraymax = np.amax(arraymax, axis = 0)
			if(arraymax - arraymin > 1e-6):
				self._lwRess[dataI] = self._lwRess[dataI] / (arraymax - arraymin)
			
			arraymin = self._hiRess[dataI]
			arraymax = self._hiRess[dataI]
			for i in range(dim):
				arraymin = np.amin(arraymin, axis = 0)
				arraymax = np.amax(arraymax, axis = 0)
			if(arraymax - arraymin > 1e-6):
				self._hiRess[dataI] = self._hiRess[dataI] / (arraymax - arraymin)
		
	def pre_data_recentre(self, newC):
		dim = len(self._lwRess.shape) - 2
		for dataI in range (self._num_examples):
			arraymin = self._lwRess[dataI] * 0.5
			arraymax = self._lwRess[dataI] * 0.5
			for i in range(dim):
				arraymin = np.amin(arraymin, axis = 0)
				arraymax = np.amax(arraymax, axis = 0)
			self._lwRess[dataI] = self._lwRess[dataI] - (arraymax + arraymin - newC)
			
			arraymin = self._hiRess[dataI] * 0.5
			arraymax = self._hiRess[dataI] * 0.5
			for i in range(dim):
				arraymin = np.amin(arraymin, axis = 0)
				arraymax = np.amax(arraymax, axis = 0)
			self._hiRess[dataI] = self._hiRess[dataI] - (arraymax + arraymin - newC)
				
	def __init__(self, lwRess, hiRess, dataID = 0, scaleFlag = False, offsetFlag = False, offC = 0.5):
		self._dataID = dataID
		self._epochs_completed = 0
		self._num_examples = int(lwRess.shape[0])
		self._lwRess = lwRess
		self._hiRess = hiRess
		if(scaleFlag):
			self.pre_data_scale()
		if(offsetFlag):
			self.pre_data_recentre(offC)
		self.shuffleData()
		
	def next_labelled_batch(self, batch_size, labels = [1, -1]):
		batch_size = int(batch_size)
		start = int(self._index_in_epoch)
		self._index_in_epoch += int(batch_size)
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			self.shuffleData()			
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = int(self._index_in_epoch)
		# rearrange the data from self.[start:end]
		highArray = self._train_order[start:end]
		lowList = self._lwRess[ highArray ]
		labelList = np.ones(batch_size,dtype=np.float)
		flagN = len(labels)
		for hi in range(batch_size):
			if(labels[(hi % flagN)] < 0):
				labelList[hi] = -1.0
				highArray[hi] = self._neg_rand_perm[ highArray[hi] ]
		highList = self._hiRess[ highArray ]
		
		return lowList,highList,labelList
		
class DataSets(object):
	pass
	
def read_packed_data_sets(SOURCE_PATH, fname = 'file/data', dataID = 0, test_rate = 0.2, \
	scaleFlag = False, offsetFlag = False, offC = 0.5):
	filepath = '%s%s_%d.npz' % (SOURCE_PATH, fname, dataID)
	data = np.load(filepath)
	print("%d input files succesfully read from %s." % (data['l'].shape[0], SOURCE_PATH))
	testSZ = int(data['l'].shape[0] * test_rate)
	data_sets = DataSets()
	if(testSZ > 0):# self._lwRess and self._hiRess is random
		trainDataPerm = data['order'][:-testSZ]
		testDataPerm = data['order'][-testSZ:]
		data_sets.train = DataSet(data['l'][trainDataPerm], data['r1'][trainDataPerm], dataID, scaleFlag, offsetFlag, offC)
		data_sets.test = DataSet(data['l'][testDataPerm], data['r1'][testDataPerm], dataID, scaleFlag, offsetFlag, offC)
	else: # self._lwRess and self._hiRess stay in order
		data_sets.train = DataSet(data['l'], data['r1'], dataID, scaleFlag, offsetFlag, offC)
		data_sets.test = data_sets.train
	return data_sets