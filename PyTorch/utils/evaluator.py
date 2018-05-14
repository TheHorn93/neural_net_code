# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 04:52:59 2018

@author: JHorn
"""

import numpy as np

class F1Score:
    def __init__( self, threshold = 0.5 ):
        self.threshold = threshold
        
    def __call__( self, output, gt, inverse=False ):
        if inverse:
            output = np.add( -output, 1.0 )
            gt = np.add( -gt, 1.0 )
            
        pre = self.precision(output, gt)
        re = self.recall(output, gt)
        return np.array( [2 /( 1/re + 1/pre ), re, pre] )
        
    def truePositives( self, output, gt ):
        positives = np.where( output > self.threshold, 1.0, 0.0 )
        tp = np.where( gt > 0, positives, 0.0 )
        return tp.sum()
        
    def precision( self, output, gt ):
        prec_sum = np.where( output > self.threshold, 1.0, 0.0 ).sum()
        if prec_sum > 0:
            return self.truePositives( output, gt ) /prec_sum
        else:
            return 1e-15
        
    def recall( self, output, gt ):
        rec_sum = gt.sum()
        if rec_sum > 0:
            return self.truePositives( output, gt ) /rec_sum
        else:
            return 1e-15
    
    
#a = np.array([[0,0,0,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,1],[0,0,0,1,1,1],[0,0,1,1,1,1],[0,1,1,1,1,1],[1,1,1,1,1,1]])
#b = np.array([[0,0,0,0,0,0.6],[0,0,0,0,0,0.6],[0,0,0,0,0,1],[0,0,0,0,0,1],[0,0,0,0,0,1],[0,0,0,0,0,1],[0,0,0,0,0,1]])
#a = np.random.rand( 256,256,128 )
#b = a -0.1
#b = np.where( b > 0.5, 1.0,0.0 )
#f = F1Score( 0.5 )
#print( f( a, b, False ) )