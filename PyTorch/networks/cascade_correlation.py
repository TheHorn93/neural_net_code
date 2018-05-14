# -*- coding: utf-8 -*-
"""
Created on Sun May 13 14:44:35 2018

@author: JHorn
"""

import numpy as np
network = __import__( "2-layer_conv_net" )
    
class Network():
    
    def __init__( self, stage_list ):
        self.iteration = 0
        self.max_iterator = len( stage_list )
        self.stages = stage_list     
        
    def initIteration( self ):
        kernel_sizes = self.stages[self.iteration][0]
        num_kernels = self.stages[self.iteration][1]
        activation = self.stages[self.iteration][2]
        net = network.Network( kernel_sizes, num_kernels, activation )
        if( self.iteration > 0 ):
            diff_ch = num_kernels -self.stages[self.iteration -1][1]
            if( self.stages[self.iteration -1][0][0][0] < kernel_sizes[0][0] or diff_ch > 0 ):
                kernel_size = [num_kernels,1]
                kernel_size.append( self.stages[self.iteration][0][0][0] )
                kernel_size.append( self.stages[self.iteration][0][0][1] )
                kernel_size.append( self.stages[self.iteration][0][0][2] )
                diff = int((kernel_sizes[0][0] -self.stages[self.iteration-1][0][0][0] ) /2 )
                new_kernel = np.random.rand( num_kernels, 1, self.stages[self.iteration][0][0][0],self.stages[self.iteration][0][0][1],self.stages[self.iteration][0][0][2] )
                scale = max( np.amin( np.abs( self.weights[0][0] ) ), np.amax( np.abs( self.weights[0][0] ) ) )
                dq = self.stages[self.iteration][0][0][0]*3
                if( diff > 0 ):
                    if( diff_ch > 0 ):
                        new_kernel[0:-diff_ch,:,diff:-diff,diff:-diff,diff:-diff] = self.weights[0][0] /scale
                    else:
                        new_kernel[:,:,diff:-diff,diff:-diff,diff:-diff] = self.weights[0][0] /scale
                bias = self.weights[0][1]
                if( diff_ch > 0 ):
                    pr_bias = np.random.rand( num_kernels )
                    pr_bias[:-diff_ch] = bias
                    bias = pr_bias
                self.weights[0] = ( new_kernel /dq, bias /dq )
            if( self.stages[self.iteration -1][0][1][0] < kernel_sizes[1][0] or diff_ch > 0 ):               
                diff = int((kernel_sizes[1][0] -self.stages[self.iteration-1][0][1][0] ) /2 )
                new_kernel = np.random.rand( 1,num_kernels ,self.stages[self.iteration][0][0][0],self.stages[self.iteration][0][0][1],self.stages[self.iteration][0][0][2] )
                scale = max( np.amin( np.abs( self.weights[1][0] ) ), np.amax( np.abs( self.weights[1][0] ) ) )
                if( diff > 0 ):
                    new_kernel[:,0-diff,diff:-diff,diff:-diff,diff:-diff] = self.weights[1][0] /scale
                dq = self.stages[self.iteration][0][1][0]*3
                self.weights[1] = ( new_kernel /dq, self.weights[1][1] /dq )
            net.setWeights( self.weights )
        self.iteration = self.iteration +1
        return net
        
    def getWeights( self, net ):
        self.weights = net.getWeightsCuda()