# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 19:17:56 2018

@author: JHorn
"""

import torch.nn.functional as funcs

class Sigmoid:
    
    def __str__( self ):
        return "Sigmoid 1/(1+exp(-x))"
    
    def __call__( self, inputs ):
        return funcs.sigmoid( inputs )
    
class ReLU:
    
    def __str__( self ):
        return "Rectified Linear max(0,x)"
    
    def __call__( self, inputs ):
        return funcs.relu( inputs )
    
class TanH:
    
    def __str__( self ):
        return "TanH (exp(x)-exp(-x))/(exp(x)+exp(-x))"
    
    def __call__( self, inputs ):
        return funcs.tanh( inputs )