#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 00:32:39 2018

@author: root
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
        self.offset = int( (kernel_sizes[0][0] -1) /2 +(kernel_sizes[1][0] -1) /2 +(kernel_sizes[2][0] -1) /2) *2
        self.layer_offset = [(kernel_sizes[0][0] -1) /2, (kernel_sizes[1][0] -1) /2, +(kernel_sizes[2][0] -1) /2]
        self.sl_offset = self.offset *2 + int( ( kernel_sizes[3][0] -1 ) /2 )
        self.teacher_offset = self.offset +int( (kernel_sizes[3][0] -1)/2 )
        self.kernel_sizes = []
        self.kernel_sizes.append( kernel_sizes[0] )
        self.activation1 = activation[0]
        self.activation2 = activation[1]
        self.activation3 = activation[2]
        self.conv1 = nn.Conv3d( 1, num_kernels[0], kernel_sizes[0] )
        self.bt_norm1 = nn.BatchNorm3d( num_kernels[0] )
        self.conv2 = nn.Conv3d( num_kernels[0], num_kernels[1], kernel_sizes[1] )
        self.bt_norm2 = nn.BatchNorm3d( num_kernels[1] )
        self.conv3 = nn.Conv3d( num_kernels[1], 1, kernel_sizes[2] )
        self.bt_norm3 = nn.BatchNorm3d( 1 )
        self.ups = nn.Upsample( scale_factor=2, mode='trilinear' )
        self.conv4 = nn.Conv3d( 1,1,kernel_sizes[3] )
        self.bt_norm4 = nn.BatchNorm3d( 1 )
        
    def forward( self, input_data, ff=False ):
        output = self.bt_norm1( self.activation1( self.conv1( input_data ) ) )
        output = self.bt_norm2( self.activation2( self.conv2( output ) ) )
        offset = int(self.layer_offset[0] +self.layer_offset[1])
        output += input_data[:,:,offset:-offset,offset:-offset,offset:-offset]
        output = self.bt_norm3( self.activation3( self.conv3( output ) ) )
        output = self.ups( output )
        output = self.bt_norm4( self.conv4( output ) )
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
        structure = ( str( self.conv1 ) + ", act = " + str(self.activation1) + "\n"
                    +str( self.conv2 ) + ", act = " + str(self.activation2) + ", residual input \n"            
                    +str( self.conv3 )+ ", act = sigmoid " )
        return structure
    
    def getWeights( self ):
        weight_list = [(self.conv1.weight.data.numpy(), self.conv1.bias.data.numpy())]
        weight_list.append[(self.conv2.weight.data.numpy(), self.conv2.bias.data.numpy())]
        weight_list.append[(self.conv3.weight.data.numpy(), self.conv3.bias.data.numpy())]
        weight_list.append[(self.conv4.weight.data.numpy(), self.conv4.bias.data.numpy())]
        return weight_list
    
    def getWeightsCuda( self ):
        weight_list = [(self.conv1.weight.cpu().data.numpy(), self.conv1.bias.cpu().data.numpy())]
        weight_list.append((self.conv2.weight.cpu().data.numpy(), self.conv2.bias.cpu().data.numpy()))
        weight_list.append((self.conv3.weight.cpu().data.numpy(), self.conv3.bias.cpu().data.numpy()))
        weight_list.append((self.conv4.weight.cpu().data.numpy(), self.conv4.bias.cpu().data.numpy()))
        weight_list.append((self.bt_norm1.weight.cpu().data.numpy(), self.bt_norm1.bias.cpu().data.numpy()))
        weight_list.append((self.bt_norm2.weight.cpu().data.numpy(), self.bt_norm2.bias.cpu().data.numpy()))
        weight_list.append((self.bt_norm3.weight.cpu().data.numpy(), self.bt_norm3.bias.cpu().data.numpy()))
        weight_list.append((self.bt_norm4.weight.cpu().data.numpy(), self.bt_norm4.bias.cpu().data.numpy()))
        return weight_list
    
    def setWeights( self, weight_list ):
        self.conv1.weight = nn.Parameter( torch.Tensor( weight_list[0][0] ) )
        self.conv1.bias = nn.Parameter( torch.Tensor( weight_list[0][1] ) )
        self.conv2.weight = nn.Parameter( torch.Tensor( weight_list[1][0] ) )
        self.conv2.bias = nn.Parameter( torch.Tensor( weight_list[1][1] ) )
        self.conv3.weight = nn.Parameter( torch.Tensor( weight_list[2][0] ) )
        self.conv3.bias = nn.Parameter( torch.Tensor( weight_list[2][1] ) )
        self.conv4.weight = nn.Parameter( torch.Tensor( weight_list[3][0] ) )
        self.conv4.bias = nn.Parameter( torch.Tensor( weight_list[3][1] ) )
        self.conv4.weight = nn.Parameter( torch.Tensor( weight_list[3][0] ) )
        self.bt_norm1.bias = nn.Parameter( torch.Tensor( weight_list[4][1] ) )
        self.bt_norm1.weight = nn.Parameter( torch.Tensor( weight_list[4][0] ) )
        self.bt_norm2.weight = nn.Parameter( torch.Tensor( weight_list[5][0] ) )
        self.bt_norm2.bias = nn.Parameter( torch.Tensor( weight_list[5][1] ) )
        self.bt_norm3.weight = nn.Parameter( torch.Tensor( weight_list[6][0] ) )
        self.bt_norm3.bias = nn.Parameter( torch.Tensor( weight_list[6][1] ) )
        self.bt_norm4.weight = nn.Parameter( torch.Tensor( weight_list[7][0] ) )
        self.bt_norm4.bias = nn.Parameter( torch.Tensor( weight_list[7][1] ) )
        
    def setWeight( self, weight, it, has_bias ):
        if it == 1:
            self.conv1.weight = nn.Parameter( torch.Tensor( weight[0] ) )
            if has_bias:
                self.conv1.bias = nn.Parameter( torch.Tensor( weight[1] ) )
        elif it == 2:
            self.conv2.weight = nn.Parameter( torch.Tensor( weight[0] ) )
            if has_bias:
                self.conv2.bias = nn.Parameter( torch.Tensor( weight[1] ) )
        else:
            self.conv3.weight = nn.Parameter( torch.Tensor( weight[0] ) )
            if has_bias:
                self.conv3.bias = nn.Parameter( torch.Tensor( weight[1] ) )

net = Network( [(3,3,3),(3,3,3),(3,3,3),(3,3,3)],[16,8], [funcs.relu, funcs.relu, funcs.relu] )
import torch.optim as opter
#print( net.state_dict() ) 
for idx, m in enumerate( net.children() ):
    print( str(idx) + '->' +str(m) )
inp = torch.ones( (1,1,20,20,20) )
net( torch.autograd.Variable( inp ) )
opt = opter.Adam( net, 0.5 ) 