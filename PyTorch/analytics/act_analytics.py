# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 22:27:54 2018

@author: JHorn
"""

import cv2
import numpy as np

class ActivationAnalytics:
    
    def __init__( self, net, log, loader ):
        self.net = net
        self.loader = loader
        self.log = log
    
    def fillActivationMaps( self, bt_size=4, path="" ):
        weights = self.log.getWeights( path )
        self.net.setWeights( weights )
        self.net.cuda()
        input_data, teacher = self.loader.getBatch( 0, bt_size )
        act_list = []
        for it in range( bt_size ):
            act_list.append( self.net.getActivationMap( input_data[:,it,:,:,:].unsqueeze(1) ) )
        self.act_list = act_list
    
    
    def getDiffFromMap( self, activation_map, teacher, layer ):
        """ input: 
            activation_map: 3d numpy array
            log: log object to write to """
        offset = 0
        for it in range( layer +1 ):
            offset += int( self.net.layer_offset[it] )
        activation_map *= 1.0 /( np.amax( activation_map ) -np.amin( activation_map ) )
        diff = activation_map -teacher[offset:-offset, offset:-offset, offset:-offset]
        diff_n = np.where( diff < 0.0, diff, 0.0 )
        diff_p = np.where( diff > 0.0, diff, 0.0 )
        output = []
        for it in range( activation_map.shape[2] ):
            diff_im = np.zeros( [activation_map.shape[0], activation_map.shape[1], 3] )
            diff_im[:,:,1] = activation_map[:,:,it]
            diff_im[:,:,0] = diff_n[:,:,it]
            diff_im[:,:,2] = diff_p[:,:,it]
            output.append( diff_im )
        return output
    
    def visualizeActivationMaps( self, path="" ):
        print( "Saving activation maps" )
        scan_it = 0
        for scan in self.act_list:
            layer_it = 0
            for layer in scan:
                shape = list( layer[0].shape )
                shape[1] = 1
                for it in range( layer[0].shape[1] ):
                    self.log.visualizeOutputStack( layer[0][:,it,:,:,:].reshape(shape), path, "scan_" +str(scan_it) +"/layer_" +str(layer_it) + "/filter_" + str(it) + "_pre_acc/" )
                    self.log.visualizeOutputStack( layer[1][:,it,:,:,:].reshape(shape), path, "scan_" +str(scan_it) +"/layer_" +str(layer_it) + "/filter_" + str(it) + "_post_acc/" )
                layer_it += 1
            scan_it += 1