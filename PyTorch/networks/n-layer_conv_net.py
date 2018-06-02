# -*- coding: utf-8 -*-
"""
Created on Fry May 25 06:41:21 2018

@author: JHorn
"""

import torch
import torch.nn as nn
import torch.nn.functional as funcs        

class Network( nn.Module ):
    """Create conv net from cmd-line args"""
    
    class ConvLayer( nn.Module ):
        def __init__( self, kernel_size, num_kernels, activation, bt_norm=False ):
            super( Network.ConvLayer, self ).__init__()
            self.apply_bt_norm = bt_norm
            if bt_norm:
                self.bt_norm = nn.BatchNorm3d( num_kernels[0] )
            self.conv = nn.Conv3d( num_kernels[0], num_kernels[1], ( kernel_size, kernel_size, kernel_size ) )
            self.activation = activation
            
        def forward( self, inp, res_inp=0 ):
            output = inp
            if self.apply_bt_norm:
                output = self.bt_norm( output )
            output = self.conv( output )
            if self.activation is not None:
                output = self.activation( output )
            return output
        
        def getParams( self ):
            if self.bt_norm:
                params = [( self.conv.weight.cpu().data.numpy(), self.conv.bias.cpu().data.numpy() )]
                params.append(( self.bt_norm.weight.cpu().data.numpy(), self.bt_norm.bias.cpu().data.numpy() ))
            else:
                params = [(self.conv.weight.cpu().data.numpy(), self.conv.bias.cpu().data.numpy() )]
                
        def setParams( self, params ):
            if self.bt_norm:
                self.conv.weight = nn.Parameter( torch.Tensor( params[0] ) )
                self.conv.bias = nn.Parameter( torch.Tensor( params[1] ) )
                self.bt_norm.weight = nn.Parameter( torch.Tensor( params[2] ) )
                self.bt_norm.bias = nn.Parameter( torch.Tensor( params[3] ) )
            else:
                self.conv.weight = nn.Parameter( torch.Tensor( params[0] ) )
                self.conv.bias = nn.Parameter( torch.Tensor( params[1] ) )
  
    
    class ResConvLayer( nn.Module  ):
        def __init__( self, kernel_size, num_kernels, res_offset, activation, bt_norm=False ):
            super( Network.ResConvLayer, self ).__init__()
            self.offset = res_offset
            self.ops = nn.ModuleList()
            self.bt_norm = bt_norm
            if bt_norm:
                self.ops.append( nn.BatchNorm3d( num_kernels[0] ) )
            self.ops.append( nn.Conv3d( num_kernels[0], num_kernels[1], ( kernel_size, kernel_size, kernel_size ) ) )
            self.activation = activation
            
        def forward( self, inp, res_inp ):
            output = inp +res_inp[:,:,self.offset:-self.offset,self.offset:-self.offset,self.offset:-self.offset]
            if self.apply_bt_norm:
                output = self.bt_norm( output )
            output = self.conv( output )
            if self.activation is not None:
                output = self.activation( output )
            return output
        
        def getParams( self ):
            if self.bt_norm:
                params = [( self.conv.weight.cpu().data.numpy(), self.conv.bias.cpu().data.numpy() )]
                params.append(( self.bt_norm.weight.cpu().data.numpy(), self.bt_norm.bias.cpu().data.numpy() ))
            else:
                params = [(self.conv.weight.cpu().data.numpy(), self.conv.bias.cpu().data.numpy() )]
                
        def setParams( self, params ):
            if self.bt_norm:
                self.conv.weight = nn.Parameter( torch.Tensor( params[0] ) )
                self.conv.bias = nn.Parameter( torch.Tensor( params[1] ) )
                self.bt_norm.weight = nn.Parameter( torch.Tensor( params[2] ) )
                self.bt_norm.bias = nn.Parameter( torch.Tensor( params[3] ) )
            else:
                self.conv.weight = nn.Parameter( torch.Tensor( params[0] ) )
                self.conv.bias = nn.Parameter( torch.Tensor( params[1] ) )
      
    
    def __init__( self, arg_list, arg_line ):
        super( Network, self ).__init__()
        self.model = ' '.join( arg_line )
        print( "Creating network: " +self.model )
        self.parseArgs( arg_list )
        self.teacher_offset = 0
        self.ups = False
        for offs in self.offset_list:
            self.teacher_offset += offs
        print( "Created network" )
        
    def parseArgs( self, arg_list ):
        self.num_layer = len( arg_list )
        self.offset_list = []
        num_kernels = [1]
        for layer_str in arg_list:
            self.offset_list.append( ( int( layer_str[0] ) -1 ) /2 )
            num_kernels.append( int( layer_str[1] ) )
        for l_it in range( len( arg_list ) ):
            args = arg_list[l_it]
            if len( args ) < 5:
                if args[3] == 'True':
                    bt_norm = True
                else:
                    bt_norm = False
                self.add_module( "conv_" +str(l_it), self.ConvLayer( int( args[0] ), ( num_kernels[l_it], num_kernels[l_it +1] ), self.parseAct(args[2]), bt_norm ) )
            else:
                res_offset = 0
                for it in range( args[4], l_it ):
                    res_offset += self.offset_list[it]
                if args[3] == 'True':
                    bt_norm = True
                else:
                    bt_norm = False
                self.add_module( "conv_" +str(l_it), self.ResConvLayer( int( args[0] ), ( num_kernels[l_it], num_kernels[l_it +1] ), res_offset, self.parseAct(args[2]), bt_norm ) )

    def parseAct( self,  act_string ):
        if act_string == "relu":
            return funcs.relu
        elif act_string == "sigmoid":
            return funcs.sigmoid
        elif act_string == "tanh":
            return funcs.tanh
        elif act_string == "sigmoid_out":
            return None
    
    
    def getStructure( self ):
        return self.model

    def getWeights( self ):
        weights = []
        for layer in self.layers:
            weights.append( layer.getParams() )
        return weights
    
    def setLayer( self, it, weights ):
        self.layers[it].setParams( weights )
    
    def setWeights( self, weight_list ):
        for l_it in range( len( self.layers ) ):
            self.setLayer( l_it, weight_list[l_it] )


    def forward( self, inp, ff=False ):
        output = inp
        for idx, layer in enumerate( self.children() ):
            output = layer( output )
        if( ff ):
            output = funcs.sigmoid( output )
        return output
    
net = Network( ["3 8 relu False".split(), "3 1 sigmoid_out True".split()], "" )
import torch.optim as opter
#print( net.state_dict() ) 
for idx, m in enumerate( net.children() ):
    print( str(idx) + '->' +str(m) )
inp = torch.ones( (1,1,20,20,20) )
net( torch.autograd.Variable( inp ) )
opt = opter.Adam( net, 0.5 ) 