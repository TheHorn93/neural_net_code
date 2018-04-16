# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:45:14 2018

@author: JHorn
"""

import torch

class CrossEntropyDynamic:
        
    def __str__( self ):
        return "Dynamic weighted CrossEntropy"
    
    def __call__( self, inp, teacher ):
        #Cross Entropy with logits
        loss_out = torch.max( inp, torch.zeros_like( inp ) )
        loss_out = loss_out - torch.mul( inp, teacher )
        loss_out = loss_out + torch.log1p( torch.exp( -torch.abs( inp ) ) )
        
        #Weighting
        weight = torch.mean( teacher )
        weight_n = torch.mul( torch.add( torch.ones_like(teacher), -teacher ), weight )
        weights = torch.add( weight_n, teacher )
        loss_out = torch.mul( loss_out, weights )
    
        loss_out = torch.mean( loss_out )
        
        return loss_out

