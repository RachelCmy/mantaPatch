#
# my cnn graph class
#
# problem: use global graph, training, summary...
# only variable init, load, save are localized
import os, math, shutil, sys
import tensorflow as tf

g_nms_cnt = 0 # global name scope
#============ TF data methods wrapper definition ===========
# define simple tf access
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W, s = 1, pad = 0):
	if (pad == 0):
		return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='VALID')
	return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

def conv3d(x, W, s = 1, pad = 0):
	if (pad == 0):
		return tf.nn.conv3d(x, W, strides=[1, s, s, s, 1], padding='VALID')
	return tf.nn.conv3d(x, W, strides=[1, s, s, s, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
def max_pool3d_2x2(x):
	return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

def l2_distanceCal(des1, des2, rootFlag = False):
	l_des = tf.nn.l2_normalize( des1, 1)
	r_des = tf.nn.l2_normalize( des2, 1)
	if(rootFlag): # tf.square(l_des - r_des) is 0-4, or 0-2(with relu)
		return  tf.reduce_sum( tf.sqrt( tf.square(l_des - r_des) ), 1) # [0-1.414]
	else:
		return tf.reduce_sum(tf.square(l_des - r_des), 1) # [0-2]
		
def hinge_loss_l2Dis(l2_dis, labels, C1, C2):
	labeled_l2dis = tf.multiply(labels, l2_dis)
	hinge_offsets = tf.multiply(1-tf.maximum(labels,0), C2) - tf.multiply( tf.maximum(labels,0), C1 )
	return tf.maximum( (hinge_offsets + labeled_l2dis), 0.0 )

class MyCnnGraph:

	def _createWeiOneBranch(self, DIM, DCFLAG, GRIDSIZE, NoPadding,\
		CNN_KNLS_N, CNN_KNLS_P, CNN_MAPS_N, POOL_AFT_C, FC_INNER_N, DC_OUTER_N):
		var_counts = 0
		
		conv_weights 	= []
		conv_biases 	= []
		fc_weights  	= []
		fc_biases   	= []
		
		with tf.variable_scope("CreateWeis"):
			mpre = 1
			if( self.veldata == 1 ): mpre = 1 + DIM
			elif(self.veldata == 2 ): mpre = DIM
			elif(self.veldata == 3 ): mpre = 1 + DIM
			for k,m in zip(CNN_KNLS_N, CNN_MAPS_N):
				if (DIM == 2):
					var_counts += k*k*mpre*m + m
					conv_weights.append( weight_variable([k,k,mpre,m]) )
				elif (DIM == 3):
					var_counts += k*k*k*mpre*m + m
					conv_weights.append( weight_variable([k,k,k,mpre,m]) )
				conv_biases.append(bias_variable([m]))
				mpre = m
			mapwidth = GRIDSIZE
			for k,s,p in zip(CNN_KNLS_N, CNN_KNLS_P, POOL_AFT_C):
				if(NoPadding): mapwidth = mapwidth - (k-1)
				mapwidth = int( math.ceil( mapwidth / s ) )
				if(p > 0): mapwidth = int( math.ceil( mapwidth / 2.0 ) )
			if(len(CNN_MAPS_N)): 
				if (DIM == 2): mpre = mapwidth * mapwidth * CNN_MAPS_N[-1]
				elif (DIM == 3): mpre = mapwidth * mapwidth * mapwidth * CNN_MAPS_N[-1]
			else: 
				if (DIM == 2): mpre = mapwidth * mapwidth
				elif (DIM == 3): mpre = mapwidth * mapwidth * mapwidth
			
			for m in FC_INNER_N:
				var_counts += mpre * m + m
				fc_weights.append(weight_variable([mpre, m]))
				fc_biases.append( bias_variable  ([m]) )
				mpre = m
		
		#l_keep_prob = tf.placeholder(tf.float32)
		return conv_weights, conv_biases, fc_weights, fc_biases, var_counts
	
	def _connectOneBranch( self, DIM, grids, cw, cb, fw, fb, POOL_AFT_C, CNN_KNLS_P):
		#inputG = tf.minimum( tf.maximum(tf.zeros_like(grids), grids), tf.ones_like(grids) )
		inputG = grids # todo proper scale
				
		for w,b,p,s  in zip(cw, cb, POOL_AFT_C, CNN_KNLS_P):
			if (DIM == 2): conv = conv2d(inputG, w, s)
			elif (DIM == 3): conv = conv3d(inputG, w, s)
			relu = tf.nn.relu(tf.nn.bias_add(conv, b))
			if( p > 0 ):
				if (DIM == 2): pool = max_pool_2x2(relu)
				elif (DIM == 3): pool = max_pool3d_2x2(relu)
				inputG = pool
			else:
				inputG = relu
		pool_shape = inputG.get_shape().as_list()
		flatshape = pool_shape[1] * pool_shape[2] * pool_shape[3]
		if (DIM == 3): flatshape = flatshape * pool_shape[4]
		p_reshape = tf.reshape( inputG, [-1, flatshape] )
		for w,b in zip (fw, fb):
			fclayer = tf.nn.relu( tf.nn.bias_add( tf.matmul(p_reshape, w), b) )
			p_reshape = fclayer
			
		return p_reshape
	
	# generate all weights in init
	def __init__(self, tfSession, DIM, SAIMEFLAG, DCFLAG, PATCH_SIZE, BASEP_SIZE, NoPadding, wei_decay, learnrate, \
		CNN_KNLS_N, CNN_KNLS_P, CNN_MAPS_N, POOL_AFT_C, FC_INNER_N, DC_OUTER_N, vel_flag = 0, l_m = 0.0, r_m = 0.7):
		global g_nms_cnt
		self.nms_id = int(g_nms_cnt) # current name_scope id
		g_nms_cnt += 1
		print ("total cnns %d, current ID %d" % (g_nms_cnt, self.nms_id))
		self.sess = tfSession
		self.variable_counter = 0
		self.veldata = vel_flag
		
		with tf.name_scope('NM%d'%self.nms_id):
			self.y_label   = tf.placeholder(tf.float32, shape=[None], name='LabelInput')
			layers = 1
			if (self.veldata == 1): layers = 1 + DIM
			elif(self.veldata == 2): layers = DIM
			elif(self.veldata == 3): layers = 1 + DIM
			with tf.name_scope('leftBranch'):
				if (DIM == 2):
					self.base_grid = tf.placeholder(tf.float32, shape=[None, BASEP_SIZE, BASEP_SIZE, layers]) # goto left
				elif (DIM == 3):
					self.base_grid = tf.placeholder(tf.float32, shape=[None, BASEP_SIZE, BASEP_SIZE, BASEP_SIZE, layers]) 
					
				self.l_conv_weights, self.l_conv_biases, self.l_fc_weights, self.l_fc_biases, count1 = \
					self._createWeiOneBranch(DIM, DCFLAG, BASEP_SIZE, NoPadding, \
					CNN_KNLS_N, CNN_KNLS_P, CNN_MAPS_N, POOL_AFT_C, FC_INNER_N, DC_OUTER_N)
				self.variable_counter += count1
				self.l_branch_out = self._connectOneBranch( DIM, self.base_grid, self.l_conv_weights,\
					self.l_conv_biases, self.l_fc_weights, self.l_fc_biases, POOL_AFT_C, CNN_KNLS_P)
			with tf.name_scope('rightBranch'):
				if (DIM == 2):
					self.high_grid = tf.placeholder(tf.float32, shape=[None, PATCH_SIZE, PATCH_SIZE, layers]) # goto right
				elif (DIM == 3):
					self.high_grid = tf.placeholder(tf.float32, shape=[None, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, layers])
				
				if (SAIMEFLAG):
					self.r_conv_weights = 0
					self.r_conv_biases  = 0
					self.r_fc_weights   = 0
					self.r_fc_biases    = 0
					self.r_branch_out   = self._connectOneBranch( DIM, self.high_grid, self.l_conv_weights,\
						self.l_conv_biases, self.l_fc_weights, self.l_fc_biases, POOL_AFT_C, CNN_KNLS_P)
				else:
					self.r_conv_weights, self.r_conv_biases, self.r_fc_weights, self.r_fc_biases, count2 = \
						self._createWeiOneBranch(DIM, DCFLAG, PATCH_SIZE, NoPadding, \
						CNN_KNLS_N, CNN_KNLS_P, CNN_MAPS_N, POOL_AFT_C, FC_INNER_N, DC_OUTER_N)
					self.variable_counter += count2
					self.r_branch_out = self._connectOneBranch( DIM, self.high_grid, self.r_conv_weights,\
						self.r_conv_biases, self.r_fc_weights, self.r_fc_biases, POOL_AFT_C, CNN_KNLS_P)
			
			if (DCFLAG):
				with tf.name_scope('decisionLayer'):
					self.dc_weights = []
					self.dc_biases  = []
					mpre = 2 * FC_INNER_N[-1]
					for m in DC_OUTER_N:
						self.variable_counter += mpre * m + m
						self.dc_weights.append( weight_variable([mpre, m]) )
						self.dc_biases.append( bias_variable ([m]) )
					
					dcResult = tf.concat(axis=1, values=[self.l_branch_out, self.r_branch_out]) 
					for w,b in zip(self.dc_weights, self.dc_biases):
						dcResult1 = tf.nn.bias_add( tf.matmul( dcResult, w ), b)
						dcResult = tf.nn.relu( dcResult1 )
						
				self.netOutput = tf.reshape( dcResult1, [-1], 'decisionOutput')  # * switchflag + output2
			else:
				self.dc_weights = 0
				self.dc_biases = 0
				self.netOutput = l2_distanceCal(self.l_branch_out, self.r_branch_out) * (-1.0)
			with tf.name_scope('Testers'):
				self.netOutputMean = tf.reduce_mean(self.netOutput)
				self.net_p_mean  = tf.reduce_sum( tf.multiply(  tf.maximum(self.y_label,0), self.netOutput) ) / tf.reduce_sum(   tf.maximum(self.y_label,0) )
				self.net_n_mean  = tf.reduce_sum( tf.multiply(1-tf.maximum(self.y_label,0), self.netOutput) ) / tf.reduce_sum( 1-tf.maximum(self.y_label,0) )
				
			with tf.name_scope('Loss'):
				if( DCFLAG >= 2): # a mix with hinge_DC and hinge_FC
					# hinge loss, max(1- t*y, 0)
					self.net_loss = tf.maximum(1 - tf.multiply(self.y_label, self.netOutput), 0)
					# hinge on l2 of Fully connected layer, DCFLAG 3 stands for sqrt hinge loss on l2
					fcloss = hinge_loss_l2Dis( \
						l2_distanceCal(self.l_branch_out, self.r_branch_out, (DCFLAG==3)),\
						self.y_label, l_m, r_m)
					self.net_loss_mean = tf.reduce_mean( self.net_loss )+ tf.reduce_mean( fcloss )
					net_loss_sum = tf.reduce_sum( self.net_loss ) + tf.reduce_sum( fcloss )
				elif( DCFLAG == 1): # hinge_DC
					self.net_loss = tf.maximum(1 - tf.multiply(self.y_label, self.netOutput), 0)
					self.net_loss_mean = tf.reduce_mean( self.net_loss )
					net_loss_sum = tf.reduce_sum( self.net_loss )
				else: # hinge_FC
					self.net_loss = hinge_loss_l2Dis( \
						l2_distanceCal(self.l_branch_out, self.r_branch_out, (DCFLAG==3)),\
						self.y_label, l_m, r_m)
					self.net_loss_mean = tf.reduce_mean( self.net_loss )
					net_loss_sum = tf.reduce_sum( self.net_loss )
				
				weight_penalty = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])* (1.0 / self.variable_counter)
				self.loss = net_loss_sum + weight_penalty * wei_decay
					
			with tf.name_scope('Train'):
				# 1. learning rate 
				# 1). e^x decay, decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
				# global_step = tf.Variable(0, trainable=False)
				# learning_rate = tf.train.exponential_decay(st_lr, global_step, 500, 0.1, staircase=True)
				# 2). full batch decay
				self.learning_rate = tf.placeholder(tf.float32, shape=[])
				# 2. optmizer method
				self.optmizer = tf.train.AdamOptimizer(self.learning_rate)
				# 3. optmizer minimize target
				self.train_step = self.optmizer.minimize(self.loss)
				# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

			with tf.name_scope('histSummarys'):
				with tf.name_scope('testVals'):
					# group the result for understanding
					self.hist_sum_p = tf.summary.histogram('test/posDistance', self.netOutput)
					self.hist_sum_n = tf.summary.histogram('test/negDistance', self.netOutput)
				if(1):
					with tf.name_scope('weights'):
						self.hist_sum = []
						i = 0
						for lw,lb in zip(self.l_conv_weights, self.l_conv_biases):
							self.hist_sum.append(tf.summary.histogram('conv_layer_%d/lw'%i, lw) )
							self.hist_sum.append(tf.summary.histogram('conv_layer_%d/lb'%i, lb) )
							i = i+1
							
						i = 0
						for lw,lb in zip(self.l_fc_weights, self.l_fc_biases):
							self.hist_sum.append(tf.summary.histogram('fc_layer_%d/lw'%i, lw) )
							self.hist_sum.append(tf.summary.histogram('fc_layer_%d/lb'%i, lb) )
							i = i+1
						
						if(SAIMEFLAG == 0):
							i = 0
							for rw,rb in zip(self.r_conv_weights, self.r_conv_biases):
								self.hist_sum.append(tf.summary.histogram('conv_layer_%d/rw'%i, rw) )
								self.hist_sum.append(tf.summary.histogram('conv_layer_%d/rb'%i, rb) )
								i = i+1
							i = 0
							for rw,rb in zip(self.r_fc_weights, self.r_fc_biases):
								self.hist_sum.append(tf.summary.histogram('fc_layer_%d/rw'%i, rw) )
								self.hist_sum.append(tf.summary.histogram('fc_layer_%d/rb'%i, rb) )
								i = i+1
						
						if (DCFLAG):
							i = 0
							for dcw, dcb in zip( self.dc_weights, self.dc_biases ):
								self.hist_sum.append(tf.summary.histogram('dc_layer_%d/w'%i, dcw) )
								self.hist_sum.append(tf.summary.histogram('dc_layer_%d/b'%i, dcb) )
								i = i+1
				
			if(0):
				with tf.name_scope('kernelSummarys'):
					self.img_sums = []
					if(len(CNN_KNLS_N)):
						if(DIM == 2):
							graph_kernel_l = tf.transpose(self.l_conv_weights[0],  perm=[3, 0, 1, 2])
							self.img_sums.append( tf.summary.image(tag = 'l_kernels', tensor = graph_kernel_l, \
								max_images = 2*CNN_MAPS_N[0] ) )
						
							if(SAIMEFLAG == 0):
								graph_kernel_r = tf.transpose(self.r_conv_weights[0],  perm=[3, 0, 1, 2])
								self.img_sums.append( tf.summary.image(tag = 'r_kernels', tensor = graph_kernel_r, \
									max_images = 2*CNN_MAPS_N[0] ) )
		self.saver = tf.train.Saver()
		print ("Graph built. %d trainable variables in total.\n" % self.variable_counter)

	def initVals( self ):
		all_list = tf.global_variables()
		my_vars = []
		prename = 'NM%d/'%self.nms_id
		for vari in all_list:
			if (vari.name.startswith(prename) ):
				my_vars.append(vari)
		self.sess.run(tf.variables_initializer(my_vars))
		
	def loadModel( self, saved_model ):
		self.initVals()
		all_list = tf.trainable_variables()
		copy_dict = {}
		prename = 'NM%d/'%self.nms_id
		for vari in all_list:
			#print vari.name
			if (vari.name.startswith(prename) ):
				varname = vari.name[len(prename):-2]
				if(varname.count('/Adam') > 0): continue # skip adam created weights
				# Adam generate some variables...
				#varname = varname.replace( 'Train/%s'%prename, 'Train/')
				copy_dict[varname] = vari
				#print varname
			
		part_saver = tf.train.Saver(copy_dict)
		part_saver.restore(self.sess, saved_model)
		print ("load model from %s" % saved_model)

	def loadOneBranchModel( self, saved_model , branch_id = 0):
		self.initVals()
		all_list = tf.global_variables()
		copy_dict1 = {} # others
		copy_dict2 = {} # change
		prename = 'NM%d/'%self.nms_id
		for vari in all_list:
			if (vari.name.startswith(prename) ):
				varname = vari.name[len(prename):-2] # ignore :0
				if(varname.count('/Adam') > 0): continue
					
				if(branch_id == 0 and varname.count("rightBranch") > 0 ):
					varname = varname.replace("rightBranch", "leftBranch")
					copy_dict2[varname] = vari
				elif(branch_id == 1 and varname.count("leftBranch") > 0  ):
					varname = varname.replace("leftBranch", "rightBranch")
					copy_dict2[varname] = vari
				else:
					copy_dict1[varname] = vari
			
		part_saver1 = tf.train.Saver(copy_dict1)
		part_saver1.restore(self.sess, saved_model)
		part_saver2 = tf.train.Saver(copy_dict2)
		part_saver2.restore(self.sess, saved_model)
		
		print ("load one branch model from %s" % saved_model)

	def saveModel( self, save_dir_path, nowtstr ):
		all_list = tf.global_variables()
		copy_dict = {}
		prename = 'NM%d/'%self.nms_id
		for vari in all_list:
			if (vari.name.startswith(prename) ):
				varname = vari.name[len(prename):-2] # ignore :0
				if(varname.count('/Adam') > 0): continue
				copy_dict[varname] = vari
			
		part_saver = tf.train.Saver(copy_dict)
		model_file = part_saver.save(self.sess, "%smodel_%s.ckpt" % (save_dir_path,nowtstr) )
		#shutil.copyfile( os.path.realpath('MyCnnGraph.py'), '%sMyCnnGraph_%s.py' % (save_dir_path, nowtstr) )
		print("Trained model saved to %s"% model_file)
		
	def createSumWritter( self, dirPath ):
		self.train_writer = tf.summary.FileWriter('%sboard/'%dirPath, self.sess.graph)
		
	def histSummaryAdd(self, i):
		histlist = self.sess.run( self.hist_sum )
		for histsummary in histlist:
			self.train_writer.add_summary(histsummary, i)

	#def imgSummaryAdd(self, trainRange):
	#	if (len(self.img_sums)):
	#		imgsumlist = self.sess.run(self.img_sums, feed_dict={})
	#		for imgsum in imgsumlist:
	#			self.train_writer.add_summary(imgsum, trainRange)
	
	def sumWritterFlush( self ):
		self.train_writer.flush()

	def sessClose( self ):
		self.sess.close()
