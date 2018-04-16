# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:14:43 2018

@author: JHorn
"""

import torch
import torch.nn as nn

class Network( nn.Module ):
    
    def __init__( self, kernel_size ):
        super( Network, self ).__init__()
        self.teacher_offset = (kernel_size[0] -1) /2
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d( 1, 1, kernel_size )
        
        
    def forward( self, input_data ):
        output = self.conv( input_data )
        return output
    
    def getStructure( self ):
        structure = str( self.conv ) + ", act = sigmoid"
        return structure
    
    def getWeights( self ):
        weight_list = [(self.conv.weight.data.numpy(), self.conv.bias.data.numpy())]
        return weight_list
    
    def getWeightsCuda( self ):
        weight_list = [(self.conv.weight.cpu().data.numpy(), self.conv.bias.cpu().data.numpy())]
        return weight_list
    
    def setWeights( self, weight_list ):
        self.conv.weight = nn.Parameter( torch.Tensor( weight_list[0][0] ) )
        self.conv.bias = nn.Parameter( torch.Tensor( weight_list[0][1] ) )
        