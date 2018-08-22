#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 19:08:29 2018

@author: root
"""

import numpy as np
import losses
import evaluator
import split_data

class ValidationResult:

    def load( self, results ):
        self.results = results
    
    def validate( self, net, loader ):
        res_shape = [ 3, 4, 3, 2, 2, 2, 3, 5 ]
        #res_shape=[2,2,2,2,2,2,2,2]
        num_elems = 3*4*3*2*2*2*3*5
        self.results = [ np.zeros( res_shape ), np.zeros( res_shape ), np.zeros( res_shape ) ]
        loss_func = losses.CrossEntropy( 0 )
        roots = ["gtk","lupine_small", "lupine_22", "gtk"]
        it = 0
        for root in range( res_shape[0] ):
            for rad in range( res_shape[1] ):
                for rot in range( res_shape[2] ):
                    for x_flip in range( res_shape[3] ):
                        for y_flip in range( res_shape[4] ):
                            for swap in range( res_shape[5] ):
                                for noise in range( res_shape[6] ):
                                    for nbr in range( res_shape[7] ):
                                        batch, teacher = loader.getPair( roots[root], rad, rot, x_flip, y_flip, swap, noise, nbr )
                                        output = net( batch, loss_func.apply_sigmoid  )
                                        loss, loss_rt, loss_sl = loss_func( output, teacher, 1 ) 
                                        loss, loss_rt, loss_sl = loss.cpu().data.numpy(), loss_rt.cpu().data.numpy(), loss_sl.cpu().data.numpy()
                                        del output
                                        self.results[0][ root, rad, rot, x_flip, y_flip, swap, noise, nbr ] = loss
                                        self.results[1][ root, rad, rot, x_flip, y_flip, swap, noise, nbr ] = loss_rt
                                        self.results[2][ root, rad, rot, x_flip, y_flip, swap, noise, nbr ] = loss_sl
                                        it += 1
                                        del loss
                                        del loss_rt 
                                        del loss_sl
                                        print( str(it) +'/' +str(num_elems), end='\r' )
    
    def getMean( self, t_it, dim=-1 ):
        output = np.ones( 3 )
        if dim == -1:
            for it in range( 3 ):
                output[it] = np.mean( self.results[it] )
        else:
            for it in range( 3 ):
                slide = np.take( self.results[it], t_it, dim )
                output[it] = np.mean( slide )
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
        self.val_dic["overall"] = self.getParamDic( ["overall"], -1 )
        self.val_dic["type"] = self.getParamDic( self.type, 0 )
        self.val_dic["r_fac"] = self.getParamDic( self.r_fac_dic, 1 ) 
        self.val_dic["rot"] = self.getParamDic( self.rot_dic, 2 ) 
        self.val_dic["x_flip"] = self.getParamDic( self.x_flip_dic, 3 ) 
        self.val_dic["y_flip"] = self.getParamDic( self.y_flip_dic, 4 ) 
        self.val_dic["x_y_swap"] = self.getParamDic( self.swap_dic, 5 ) 
        self.val_dic["noise"] = self.getParamDic( self.c_dic, 6 ) 
        
    
    def getParamDic( self, dic, dim ):
        it = 0
        out_dic = {}
        for t in dic:
            out_dic[t] = self.getMean( it, dim )
            it += 1
        return out_dic
 

def FullSetEvaluation( network, loader, log, splits ):
    loss_func = losses.CrossEntropy( 0 )
    f1 = evaluator.F1Score()
    f1_score = np.array([0.0,0.0,0.0])
    loader.createPool()
    eval_loss, eval_loss_rt, eval_loss_sl = 0.0, 0.0, 0.0
    num_splits = splits[0] +splits[1] +splits[2]
    while loader.it < loader.set_size:
        inp, gt, key = loader.getNextInput() 
        inp_l, gt_l, w_l = split_data.validationSplit( inp, gt, splits, network.ups )
        comp_loss, comp_rt_loss, comp_sl_loss = 0,0,0
        for it in range( len(inp_l) ): 
            output = network( inp_l[it], loss_func.apply_sigmoid )
            loss, loss_rt, loss_sl = loss_func( output, gt_l[it], 1 )
            comp_loss += loss /num_splits
            comp_rt_loss += loss_rt *w_l[it][0] /num_splits
            comp_sl_loss += loss_sl *w_l[it][1] /num_splits
            f1_temp = f1( output[0,0,:,:,:], gt_l[it][0,0,:,:,:] )
            f1_temp[0] /= num_splits
            w_f1 = w_l[it][0] /num_splits
            f1_temp[1] /= w_f1
            f1_temp[2] /= w_f1
            f1_score += f1_temp
        del inp_l
        del gt_l
        eval_loss += comp_loss.cpu().data.numpy()
        eval_loss_rt += comp_sl_loss.cpu().data.numpy()
        eval_loss_sl += comp_rt_loss.cpu().data.numpy()
        del inp
        del gt
        del loss
        del loss_rt
        del loss_sl
        print( str(loader.it) +"/" +str(loader.set_size) +"   " +str(key), end="\r"  )
    eval_loss = eval_loss /loader.set_size
    eval_loss_rt = eval_loss_rt /loader.set_size
    eval_loss_sl = eval_loss_sl /loader.set_size
    f1_score /= loader.set_size
    log.saveEvalResults( network.getStructure(), eval_loss, eval_loss_rt, eval_loss_sl, f1_score )
    
