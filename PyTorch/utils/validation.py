#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 19:08:29 2018

@author: root
"""

import numpy as np
import losses

class ValidationResult:

    def validate( self, net, loader ):
        res_shape = [ 3, 4, 3, 2, 2, 2, 3, 5, 3 ]
        self.results = np.zeros( res_shape )
        loss_func = losses.NegativeLogLikelihood( -1 )
        roots = ["lupine_small", "lupine_22", "gtk"]
        for root in range( res_shape[0] ):
            for rad in range( res_shape[1] ):
                for rot in range( res_shape[2] ):
                    for x_flip in range( res_shape[3] ):
                        for y_flip in range( res_shape[4] ):
                            for swap in range( res_shape[5] ):
                                for noise in range( res_shape[6] ):
                                    for nbr in range( res_shape[7] ):
                                        batch, teacher = loader.getPair( roots[root], rad, rot, x_flip, y_flip, swap, noise, nbr )
                                        output = net( batch )
                                        loss, loss_rt, loss_sl = loss_func( output, teacher, 1 ) 
                                        loss, loss_rt, loss_sl = loss.cpu().data.numpy(), loss_rt.cpu().data.numpy(), loss_sl.cpu().data.numpy()
                                        self.results[ root, rad, rot, x_flip, y_flip, swap, noise, nbr, : ] = np.array( [loss, loss_rt, loss_sl] )
    
    def getMean( self, dim=-1 ):
        if dim == -1:
            return np.mean( self.results )
        else:
            return np.mean( self.results, dim )
        
    def save( self, path ):
        np.save( path, self.results )