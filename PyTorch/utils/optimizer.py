# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 02:34:32 2018

@author: JHorn
"""

import torch.optim as optim

class GradientDescent:
    
    def __str__( self ):
        return "Stochastic Gradient Descent"
    
    def __call__( self, net, lr, momentum=0, weight_decay=0 ):
        return optim.SGD( net.parameters(), lr, momentum, weight_decay )


class AdamOptimizer:
    
    def __str__( self ):
        return "Adam Optimizer"
    
    def __call__( self, net, lr, betas=(0.9,0.999), eps=1e-08, weight_decay = 0 ):
        return optim.Adam( net.parameters(), lr, betas, eps, weight_decay )