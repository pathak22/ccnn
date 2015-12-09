# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Written by Deepak Pathak, Philipp Krahenbuhl
# --------------------------------------------------------

from config import *
import numpy as np

class WeakLoss(caffe.Layer):
	def DS(self, I, stride=32, factor=32,N=21 ):
		r = np.zeros(( (I.shape[0]-1)/stride+1, (I.shape[1]-1)/stride+1 ), dtype=I.dtype)
		for j in range(0,I.shape[0],stride):
			for i in range(0,I.shape[1],stride):
				p = I[j:j+factor,i:i+factor]
				b = np.bincount( p[p>=0],minlength=N )
				r[j/stride,i/stride] = np.argmax(b)
		return r

	def setup(self,bottom,top):

		self.bg_lower,self.bg_upper = 0.3,0.7
		self.bg_slack = 1e10 		# no slack : 1e10
		self.fg_lower_hard = 0.01
		self.fg_lower = 0.05
		self.fg_slack = 2 		# no slack : 1e10
		self.hardness = 1 				# no hardness : 1 and hardness : 1000

		self.semi_supervised = False
		self.apply_size_constraint = False

		if self.apply_size_constraint:
			self.bg_lower,self.bg_upper = 0.2,0.6		# This Ablation
			self.bg_slack = 3	# 1e10
			self.fg_lower_hard = 0.1
			self.fg_lower = 0.1 						# This Ablation
			self.fg_slack = 2	# 1e10
			self.hardness = 1000
			self.fg_upper_small = 0.01 		# upper bound on small object. Don't make it zero as strictly less than 0 is not satisfiable. Make it epsilon small.
		
		# self.counter = 1

	def reshape(self, bottom, top):
		top[0].reshape(1,1,1,1)
	
	def forward(self, bottom, top):
		# print "first : ",int(np.prod(bottom[0].data.shape[1:]))
		from time import time
		t0 = time()
		D = bottom[0].channels
		from ccnn import constraintloss
		self.diff = []
		loss,w = 0,0
		for i in range(bottom[0].num):
			# print '-------------------------------------'
			# print 'Image Number : ',self.counter 

			if self.semi_supervised:
				assert (len(bottom)>4),"Semi Supervised Flag ON, but full supervised images not supplied as additional bottom !"

			if (not self.semi_supervised) or (bottom[3].data[i]==0): 		# weakly-supervised downsampled training
				# Setup bottoms
				f = np.ascontiguousarray(bottom[0].data[i].reshape((D,-1)).T) 		# f : height*width x channels
				q = np.exp(f-np.max(f,axis=1)[:,None]) 								# expAndNormalize across channels
				q/= np.sum(q,axis=1)[:,None]

				# Setup the constraint softmax
				csm = constraintloss.ConstraintSoftmax(self.hardness)

				# Add Negative Label constraints
				# L = bottom[2].data[i].flatten() > 0.5 			
				L = bottom[1].data[i].flatten() > 0.5
				csm.addZeroConstraint( (~L).astype(np.float32) )
				
				# Add Small Object Size constraints
				# L_up = 0*L
				# if self.apply_size_constraint:
				# 	assert (len(bottom)>2),"Size constraint ON, but size information not supplied as additional bottom !"
				# 	L_up = 1*L
				# 	L = bottom[2].data[i].flatten() > 0.5

				# for l in np.flatnonzero(L_up):
				# 	if l>0 and not L[l]:
				# 		v = np.zeros(D).astype(np.float32); v[l] = 1
				# 		csm.addLinearConstraint( -v, -self.fg_upper_small, self.fg_slack )

				# Apply Positive Label Constraints
				for l in np.flatnonzero(L):
					if l>0:
						v = np.zeros(D).astype(np.float32); v[l] = 1
						if self.apply_size_constraint:
							csm.addLinearConstraint(  v, self.fg_lower_hard )
						csm.addLinearConstraint(  v, self.fg_lower, self.fg_slack )

				# Add Background Constraints
				v = np.zeros(D).astype(np.float32); v[0] = 1
				csm.addLinearConstraint(  v, self.bg_lower, self.bg_slack ) # lower bound
				if (np.sum(L[1:]) > 0): # i.e. image is not all background
					csm.addLinearConstraint( -v, -self.bg_upper ) # upper bound
				
				# Run constrained optimization
				p = csm.compute(f)

				self.diff.append( ((q-p).T.reshape(bottom[0].data[i].shape))/np.float32(f.shape[0]) )      # normalize by (f.shape[0])
				# self.diff.append( ((q-p).T.reshape(bottom[0].data[i].shape)) )      # unnormalize

				# Debugging Code ---------
				# temp = 1
				# for l in np.flatnonzero(L_up):
				# 	if l>0 and not L[l]:
				# 		if p[:,l].sum() > self.fg_upper_small:
				# 			print 'Small Object Class Index=',temp,'  sumP=',p[:,l].sum(),'  sumQ=',q[:,l].sum()
				# 			print '\tP=',repr(p[:,l])
				# 			print '\tQ=',repr(q[:,l])
				# 	temp += 1
				# print ''
				# np.savez('./debug/debug_im'+str(self.counter)+'.npz', hardness=self.hardness, bg_lower = self.bg_lower, bg_upper=self.bg_upper, L=L, L_up=L_up, fg_lower = self.fg_lower, fg_slack=self.fg_slack, fg_upper_small=self.fg_upper_small, f=f,p=p,q=q )
				# self.counter += 1
				# -----------------------

			else: 		# fully-supervised upsample training
				f = np.ascontiguousarray(bottom[5].data[i].reshape((D,-1)).T) 		# f : height*width x channels
				q = np.exp(f-np.max(f,axis=1)[:,None]) 								# expAndNormalize across channels
				q/= np.sum(q,axis=1)[:,None]

				gt = bottom[4].data[i]
				# print '\t q : ',q.shape
				# print '\t cnn_output_Shape : ',bottom[0].data[i].shape
				# print '\t gt_Shape : ',gt.shape
				# print '\t gt_resized_Shape : ', (np.float32(self.DS(np.uint8(gt[0,...])))).shape
				gt = np.uint8(gt[0,...]) 		# For downsampling the gt use this : self.DS(np.uint8(gt[0,...]))
				gt = np.ascontiguousarray(gt.reshape((1,-1)).T) 		# gt : height*width x 1
				gt = gt.squeeze()
				p = np.zeros(q.shape).astype(np.float32) 					# q,p,f : height*width x channels
				ind = np.where(gt!=255)
				p[ind,gt[ind]] = 1
				ind = np.where(gt==255)
				p[ind,:] = q[ind,:] 								# so that q-p=0 at this position because it is ignore label

				self.diff.append( ((q-p).T.reshape(bottom[5].data[i].shape))/np.float32(f.shape[0]) )      # normalize by (f.shape[0])

			
			loss += (np.sum(p*np.log(np.maximum(p,1e-10))) - np.sum(p*np.log(np.maximum(q,1e-10))))/np.float32(f.shape[0])    # normalize by (f.shape[0])
			# loss += (np.sum(p*np.log(np.maximum(p,1e-10))) - np.sum(p*np.log(np.maximum(q,1e-10))))    # unnormalize

# 			print( np.min(f), np.max(f) )
# 			np.set_printoptions(linewidth=150)
# 			print( L.astype(bool) )
# 			print( np.bincount(np.argmax(f,axis=1),minlength=21) )
# 			print( np.sum(p[:,~L]), 'P', np.sum(p,axis=0).astype(int)[L], 'H', np.bincount(np.argmax(p,axis=1),minlength=L.size)[L] )
		#print( "===== %f ====="%(time()-t0) )
		top[0].data[0,0,0,0] = loss
		self.diff = np.array(self.diff)

	def backward(self, top, propagate_down, bottom):
		for i in range(bottom[0].num):
			if (not self.semi_supervised) or (bottom[3].data[i]==0):
				bottom[0].diff[i] = top[0].diff[0,0,0,0]*self.diff[i]
			else:
				bottom[5].diff[i] = top[0].diff[0,0,0,0]*self.diff[i]
