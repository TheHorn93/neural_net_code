# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:02:58 2018

@author: JHorn
"""

import torch
import torch.nn as nn
import torch.nn.functional as funcs

def getParameterConf():
    parameter_list = []
    parameter_list.append( [ "kernel", "dims", "act" ] )
    parameter_list.append( [ "kernel" ] )
    

class Network( nn.Module ):
    
    def __init__( self, kernel_sizes, num_kernels, activation ):
        super( Network, self ).__init__()
        self.teacher_offset = (kernel_sizes[0][0] -1) /2 +(kernel_sizes[1][0] -1) /2
        self.layer_offset = [(kernel_sizes[0][0] -1) /2, (kernel_sizes[1][0] -1) /2]
        self.kernel_sizes = []
        self.kernel_sizes.append( kernel_sizes[0] )
        self.activation = activation
        self.conv1 = nn.Conv3d( 1, num_kernels, kernel_sizes[0] )
        self.conv2 = nn.Conv3d( num_kernels, 1, kernel_sizes[1] )
        
        
    def forward( self, input_data, ff=False ):
        output = self.activation( self.conv1( input_data ) )
        output = self.conv2( output )
        if( ff ):
            output = funcs.sigmoid( output )
        return output
    
    def getActivationMap( self, input_data ):
        output = []
        layer_1 = self.conv1( input_data )
        output.append( ( layer_1.cpu().data.numpy(), self.activation( layer_1 ).cpu().data.numpy() ) )
        layer_2 = self.conv2( self.activation( layer_1 ) )
        output.append( ( layer_2.cpu().data.numpy(), funcs.sigmoid( layer_2 ).cpu().data.numpy() ) )
        return output
        
    
    def getStructure( self ):
        structure = str( self.conv1 ) + ", act = " + str(self.activation) + "\n" +str( self.conv2 )+ ", act = sigmoid"
        return structure
    
    def getWeights( self ):
        weight_list = [(self.conv1.weight.data.numpy(), self.conv1.bias.data.numpy())]
        weight_list.append[(self.conv2.weight.data.numpy(), self.conv2.bias.data.numpy())]
        return weight_list
    
    def getWeightsCuda( self ):
        weight_list = [(self.conv1.weight.cpu().data.numpy(), self.conv1.bias.cpu().data.numpy())]
        weight_list.append((self.conv2.weight.cpu().data.numpy(), self.conv2.bias.cpu().data.numpy()))
        return weight_list
    
    def setWeights( self, weight_list ):
        self.conv1.weight = nn.Parameter( torch.Tensor( weight_list[0][0] ) )
        self.conv1.bias = nn.Parameter( torch.Tensor( weight_list[0][1] ) )
        self.conv2.weight = nn.Parameter( torch.Tensor( weight_list[1][0] ) )
        self.conv2.bias = nn.Parameter( torch.Tensor( weight_list[1][1] ) )