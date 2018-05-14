# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:45:14 2018

@author: JHorn
"""

import torch
import numpy as np

class LossBase:
    
    def __init__( self, gate, apply_sigmoid ):
        self.gate = gate
        self.apply_sigmoid = apply_sigmoid
        
    def __str__( self ):
        return "Loss Base Class"
    
    #linear weigthing
    def weighting( self, loss_out, teacher, soil, epoch ):
        if epoch < self.gate:
            weight = torch.mean( teacher )
            weight_n = torch.mul( soil, weight )
            weights = torch.add( weight_n, teacher )
            loss_out = torch.mul( loss_out, weights )
        return loss_out
        

class CrossEntropy( LossBase ):
    
    def __init__( self, gate = 500 ):
        super().__init__( gate, False )
        
    def __str__( self ):
        return "Dynamic weighted CrossEntropy"
    
    def __call__( self, inp, teacher, epoch ):
        #Cross Entropy with logits
        loss_out = torch.max( inp, torch.zeros_like( inp ) )
        loss_out = loss_out - torch.mul( inp, teacher )
        loss_out = loss_out + torch.log1p( torch.exp( -torch.abs( inp ) ) )
        
        soil = torch.add( torch.ones_like(teacher), -teacher )
        soil_loss = torch.div( torch.mul( soil, loss_out ), soil.sum() ).sum()
        root_loss = torch.div( torch.mul( teacher, loss_out ), teacher.sum() ).sum()
        
        #Weighting
        loss_out = self.weighting( loss_out, teacher, soil, epoch )
    
        loss_out = torch.mean( loss_out )
        
        return ( loss_out, root_loss, soil_loss )


class NegativeLogLikelihood( LossBase ):
    
    def __init__( self, gate = 500 ):
        super().__init__( gate, True )
        
    def __str__( self ):
        return "Negative Log Likelihood"
    
    def __call__( self, inp, teacher, epoch ):
        comparator = torch.zeros_like( inp ) +1e-15
        
        root_loss = -torch.log( torch.max( inp, comparator ) )
        root_loss = torch.mul( root_loss, teacher )
        
        soil = torch.add( torch.ones_like(teacher), -teacher )
        soil_val = torch.add( torch.ones_like( inp ), -inp )
        soil_loss = -torch.log( torch.max( soil_val, comparator ) )
        soil_loss = torch.mul( soil_loss, soil )
        
        loss_out = torch.add( root_loss, soil_loss )
        root_loss = torch.div( root_loss, teacher.sum() ).sum()
        soil_loss = torch.div( soil_loss, soil.sum() ).sum()
        
        loss_out = self.weighting( loss_out, teacher, soil, epoch )
            
        loss_out = torch.mean( loss_out )
        
        return ( loss_out, root_loss, soil_loss )


class LossDynaBase:
    
    def __init__( self, gate_init, gate_end, apply_sigmoid ):
        self.gate_init = gate_init
        self.gate_end = gate_end
        self.weight_interval = gate_end -gate_init
        self.apply_sigmoid = apply_sigmoid
        
    def __str__( self ):
        return "Dynamicaly weighted Loss Base"
    
    def weighting( self, loss_out, teacher, soil, epoch ):
        if epoch < self.gate_end:
            weight = torch.mean( teacher )
            if epoch > self.gate_init:
                weight_val = -( self.gate_init -epoch ) /self.weight_interval
                #weight_val = np.power( weight_val, 1/3 )
                weight += (1 -weight) *weight_val
            weight_n = torch.mul( soil, weight )
            weights = torch.add( weight_n, teacher )
            loss_out = torch.mul( loss_out, weights )
        return loss_out


class CrossEntropyDynamic( LossDynaBase ):
    
    def __init__( self, gate_init = 400, gate_end = 600 ):
        super().__init__( gate_init, gate_end, False )
        
    def __str__( self ):
        return "Dynamic, changing weighted CrossEntropy " +str(self.gate_init) +" -> " +str(self.gate_end)
    
    def __call__( self, inp, teacher, epoch ):
        #Cross Entropy with logits
        loss_out = torch.max( inp, torch.zeros_like( inp ) )
        loss_out = loss_out - torch.mul( inp, teacher )
        loss_out = loss_out + torch.log1p( torch.exp( -torch.abs( inp ) ) )
        
        soil = torch.add( torch.ones_like(teacher), -teacher )
        soil_loss = torch.div( torch.mul( soil, loss_out ), soil.sum() ).sum()
        root_loss = torch.div( torch.mul( teacher, loss_out ), teacher.sum() ).sum()
        
        #Weighting
        loss_out = self.weighting( loss_out, teacher, soil, epoch )
    
        loss_out = torch.mean( loss_out )
        
        return ( loss_out, root_loss, soil_loss )    
    
#inp = torch.Tensor([[10,5,-10,-10],[-10,10,-10,-10],[-10,-10,10,-10]])
#inp = torch.Tensor([[0,0,0,0],[0,1,0,0],[0,0,1,0]])
#teacher = torch.Tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
#loss = NegativeLogLikelihood( 0 )    
#import visdom
#import numpy as np
#vis = visdom.Visdom()
#x = np.arange(0.0,1.01,0.01)
#y = np.array([])
#for x_it in x:
#    loss_val, root, soil = loss( torch.Tensor([x_it]), torch.Tensor( [1.0] ), 1 )
#    y = np.append( y, loss_val )
#    print( str(root) + ", " + str(soil) )
#vis.line( y, x, win="Test" )
#print( loss( inp, teacher, 1 ) )
#loss = CrossEntropyDynamic(10,20)
#for epoch in range(10,21):
#    loss.weighting( torch.Tensor( [1,0,0,0,0,0] ), torch.Tensor([1,0,0,0,0,0]), torch.Tensor([0,1,1,1,1,1]), epoch )