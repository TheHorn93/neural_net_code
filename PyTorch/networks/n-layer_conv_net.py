# -*- coding: utf-8 -*-
"""
Created on Fry May 25 06:41:21 2018

@author: JHorn
"""

#TODO allow for input padding on demand! Networks should be evaluated on same size output.

import torch
import torch.nn as nn
import torch.nn.functional as funcs
import numpy as np

class Network( nn.Module ):
    """Create conv net from cmd-line args"""
    
    class ConvLayerBase( nn.Module ):
        def __init__( self ):
            super( Network.ConvLayerBase, self ).__init__()
            
        def forward( self, inp, res_list ):
            output = inp
            for idx, ops in enumerate( self.children() ):
                output = ops( output, self.res_dict )
            return output
            
        def getParams( self ):
            if self.apply_bt_norm:
                params = [ self.conv.weight.cpu().data.numpy(), self.conv.bias.cpu().data.numpy() ]
                params += [ self.bt_norm.weight.cpu().data.numpy(), self.bt_norm.bias.cpu().data.numpy() ]
            else:
                params = (self.conv.weight.cpu().data.numpy(), self.conv.bias.cpu().data.numpy() )
            return params
                
        def setParams( self, params ):
            if self.apply_bt_norm:
                self.conv.weight = nn.Parameter( torch.Tensor( params[0] ) )
                self.conv.bias = nn.Parameter( torch.Tensor( params[1] ) )
                self.bt_norm.weight = nn.Parameter( torch.Tensor( params[2] ) )
                self.bt_norm.bias = nn.Parameter( torch.Tensor( params[3] ) )
            else:
                self.conv.weight = nn.Parameter( torch.Tensor( params[0] ) )
                self.conv.bias = nn.Parameter( torch.Tensor( params[1] ) )
                
    
    class ConvLayer( ConvLayerBase ):
        def __init__( self, kernel_size, num_kernels, activation, bt_norm=False ):
            super( Network.ConvLayer, self ).__init__()
            self.apply_bt_norm = bt_norm
            if bt_norm:
            #    self.bt_norm = nn.BatchNorm3d( num_kernels[0] )
                self.add_module( "bt_norm", nn.BatchNorm3d( num_kernels[0]) )
            #self.conv = nn.Conv3d( num_kernels[0], num_kernels[1], kernel_size=( kernel_size, kernel_size, kernel_size ) )
            self.add_module( "conv", nn.Conv3d( num_kernels[0], num_kernels[1], kernel_size=( kernel_size, kernel_size, kernel_size ) ) )
            if activation is not None:
                self.add_module( "act", activation )
            
        def forward( self, inp, res_list=0 ):
            output = inp
            return super( Network.ConvLayer, self ).forward( output, res_list )
        
    
    class ResConvLayer( ConvLayerBase ):
        def __init__( self, kernel_size, num_kernels, res_layer, res_offset, activation, bt_norm=False ):
            super( Network.ResConvLayer, self ).__init__()
            #self.ops = nn.ModuleList()
            self.apply_bt_norm = bt_norm
            if bt_norm:
            #    self.bt_norm = nn.BatchNorm3d( num_kernels[0] )
                self.add_module( "bt_norm", nn.BatchNorm3d( num_kernels[0]) )
            #self.conv = nn.Conv3d( num_kernels[0], num_kernels[1], ( kernel_size, kernel_size, kernel_size ) )
            self.add_module( "conv", nn.Conv3d( num_kernels[0], num_kernels[1], kernel_size=( kernel_size, kernel_size, kernel_size ) ) )
            if activation is not None:
                self.add_module( "act", activation )
            self.res_layer = res_layer
            self.offset = res_offset
            
        def forward( self, inp, res_list ):
            output = inp +res_list[self.res_layer][:,:,self.offset:-self.offset,self.offset:-self.offset,self.offset:-self.offset]
            return super( Network.ResConvLayer, self ).forward( output, res_list )
                
                
    class Upsample( ConvLayerBase ):
        def __init__( self, num_kernels, activation, bt_norm ):
            super( Network.Upsample, self ).__init__()
            self.apply_bt_norm = bt_norm
            if bt_norm:
                self.bt_norm = nn.BatchNorm3d( num_kernels[0] )
            self.conv = nn.Upsample( scale_factor = 2, mode='nearest' )
            self.activation = activation
            
        def forward( self, inp, res_inp=0 ):
            output = inp
            return super( Network.Upsample, self ).forward( output, res_inp )
        
        def getParams( self ):
            return ( np.ones( (1,1,1,1,1) ), np.ones( (1,1,1,1,1) ) )
        
        def setParams( self, params ):
            pass
        
    
    def __init__( self, arg_list, arg_line='' ):
        super( Network, self ).__init__()
        self.model = ' '.join( arg_line )
        self.ups = False
        self.teacher_offset = 0
        self.res_dict = {}
        print( "Creating network: " +self.model )
        self.parseArgs( arg_list )
        self.teacher_offset = int( self.teacher_offset )
        print( "Created network" )
        
    def parseArgs( self, arg_list ):
        self.num_layer = len( arg_list )
        self.offset_list = []
        num_kernels = [1]
        for layer_str in arg_list:
            if len( layer_str ) == 3:
                self.offset_list.append( 0 )
                num_kernels.append( int( layer_str[0] ) )
            else:
                self.offset_list.append( int( (int( layer_str[0] ) -1) /2 ) )
                num_kernels.append( int( layer_str[1] ) )
        for l_it in range( len( arg_list ) ):
            args = arg_list[l_it]
            if len( args ) == 4:
                if args[3] == 'True':
                    bt_norm = True
                else:
                    bt_norm = False
                self.add_module( "conv_" +str(l_it), self.ConvLayer( int( args[0] ), ( num_kernels[l_it], num_kernels[l_it +1] ), self.parseAct(args[2]), bt_norm ) )
            elif len( args ) == 5:
                if args[4] == 'True':
                    bt_norm = True
                else:
                    bt_norm = False
                res_layer = int( args[3] )
                res_offset = 0
                self.res_dict[res_layer] = None
                for off_it in range( res_layer, l_it ):
                    res_offset += self.offset_list[off_it]
                self.add_module( "conv_" +str(l_it), self.ResConvLayer( int( args[0] ), ( num_kernels[l_it], num_kernels[l_it +1] ), res_layer, res_offset, self.parseAct(args[3]), bt_norm ) )
            elif len( args ) == 3:
                if args[2] == 'True':
                    bt_norm = True
                else:
                    bt_norm = False
                if not self.ups:
                    self.teacher_offset *= 2
                    self.add_module( "transp_conv" +str(l_it), self.Upsample( ( num_kernels[l_it], num_kernels[l_it +1] ), self.parseAct(args[2]), bt_norm ) )
                    self.ups = True
            self.teacher_offset += self.offset_list[l_it]

    def parseAct( self,  act_string ):
        if act_string == "relu":
            return nn.ReLU()
        elif act_string == "sigmoid":
            return nn.Sigmoid()
        elif act_string == "tanh":
            return nn.Tanh()
        elif act_string == "sigmoid_out":
            return None
    
    
    def getStructure( self ):
        return self.model

    def getWeights( self ):
        weights = []
        for idx, layer in enumerate( self.children() ):
            weights.append( layer.getParams() )
        return weights
    
    def setLayer( self, it, weights ):
        self.layers[it].setParams( weights )
    
    def setWeights( self, weight_list ):
        for idx, layer in enumerate( self.children() ):
            layer.setParams( weight_list[idx] )


    def forward( self, inp, ff=False ):
        output = inp
        if 0 in self.res_dict.keys():
            self.res_dict[0] = output
        for idx, layer in enumerate( self.children() ):
            output = layer( output, self.res_dict )
            if idx+1 in self.res_dict.keys():
                self.res_dict[idx+1] = output
        if( ff ):
            output = funcs.sigmoid( output )
        return output

net = Network( ["5 3 relu False".split(),"5 3 relu True".split(),"1 sigmoid_out False".split(), "3 1 relu True".split()] , '')
#net.cuda()
#inp = torch.autograd.Variable( torch.ones( (1,1,200,200,200) ) ).cuda()
print( net.children )
#print( net.teacher_offset )
