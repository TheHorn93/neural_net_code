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
        res_shape = [ 3, 4, 3, 2, 2, 2, 3, 5 ]
        self.results = [ np.zeros( res_shape ), np.zeros( res_shape ), np.zeros( res_shape ) ]
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
                                        self.results[0][ root, rad, rot, x_flip, y_flip, swap, noise, nbr ] = loss
                                        self.results[1][ root, rad, rot, x_flip, y_flip, swap, noise, nbr ] = loss_rt
                                        self.results[2][ root, rad, rot, x_flip, y_flip, swap, noise, nbr ] = loss_sl
    
    def getMean( self, dim=-1 ):
        output = np.ones( 3 )
        if dim == -1:
            for it in range( 3 ):
                output[it] = np.mean( self.results[it] )
        else:
            for it in range( 3 ):
                output[it] = np.mean( self.results[it], dim )
        return output
    
        
    def fillDictionary( self ):
        self.type = ["lupine_small", "lupine_22", "gtk"]
        self.r_fac_dic = ["r_factor_0.34","r_factor_0.71","r_factor_1.00","r_factor_1.41"]
        self.rot_dic = ["rot_0","rot_60","rot_120"]
        self.x_flip_dic = ["x_flip_0","x_flip_1"]
        self.y_flip_dic = ["y_flip_0","y_flip_1"]
        self.swap_dic = ["x_y_swap_0","x_y_swap_1"]
        self.c_dic = ["g","h","l"]
        self.val_dic = {}
        self.val_dic["overall"] = self.getParamDic( self.type, -1 )
        self.val_dic["type"] = self.getParamDic( self.type, 0 )
        self.val_dic["r_fac"] = self.getParamDic( self.r_fac_dic, 1 ) 
        self.val_dic["rot"] = self.getParamDic( self.rot_dic, 2 ) 
        self.val_dic["x_flip"] = self.getParamDic( self.x_flip_dic, 3 ) 
        self.val_dic["y_flip"] = self.getParamDic( self.y_flip_dic, 4 ) 
        self.val_dic["x_y_swap"] = self.getParamDic( self.swap, 5 ) 
        self.val_dic["noise"] = self.getParamDic( self.c_dic, 6 ) 
        
    
    def getParamDic( self, dic, dim ):
        it = 0
        out_dic = {}
        for t in self.type:
            out_dic[t] = self.getMean( dim )[it]
            it += 1
        return out_dic
 