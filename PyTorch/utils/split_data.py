#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 21:21:45 2018

@author: root
"""

import math
import torch

def reassemble( split_list, num_splits ):
    size = [ 1, 1, 0, 0, 0 ]
    list_it = 0
    x_split = num_splits[2] *num_splits[1]
    y_split = num_splits[2]
    for dim0 in range( num_splits[0] ):
        elem = split_list[list_it]
        size[2] += elem.size()[2]
        list_it += x_split
    list_it = 0
    for dim1 in range( num_splits[1] ):
        elem = split_list[list_it]
        size[3] += elem.size()[3]
        list_it += y_split
    list_it = 0
    for dim2 in range( num_splits[2] ):
        elem = split_list[list_it]
        size[4] += elem.size()[4]
        list_it += 1
    print("Size to reassemble: " +str(size))
    out = torch.zeros( size )
    pos_x = 0
    list_it = 0
    elem = split_list[list_it]
    for dim0 in range( num_splits[0] ):
        elem = split_list[list_it]
        n_pos_x = pos_x +elem.size()[2]
        pos_y = 0
        for dim1 in range( num_splits[1] ):
            elem = split_list[list_it]
            n_pos_y = pos_y +elem.size()[3]
            pos_z = 0
            for dim2 in range( num_splits[2] ):
                elem = split_list[list_it]
                n_pos_z = pos_z +elem.size()[4]
                #print( "IT: " +str(elem.size()) )
                print( str(pos_x) +"-" +str( n_pos_x ) + "  " +str(pos_y) +"-" +str( n_pos_y ) + "  " +str(pos_z) +"-" +str( n_pos_z )  )
                out[ 0, 0 ,pos_x : n_pos_x, pos_y : n_pos_y, pos_z : n_pos_z ] = elem 
                list_it += 1
                pos_z = n_pos_z
            pos_y = n_pos_y
        pos_x = n_pos_x
    return out

def validationSplit( tensor, teacher, num_splits, ups=False ):
    num_roots = teacher.sum()
    soil = torch.ones_like( teacher ) - teacher
    num_soil = soil.sum()
    inp, tch = splitInputAndTeacher( tensor, teacher, num_splits, ups )
    weights = []
    w1,w2=0,0
    for split in tch:
        w_r = split.sum() /num_roots
        w_s = ( torch.ones_like(split) -split ).sum() /num_soil
        weights.append( (w_r,w_s) )
        w1 += w_r
        w2 += w_s
    #print( str(w1) +" <> " +str(w2) )
    return inp, tch, weights
               
def splitInputAndTeacher( tensor, teacher, num_splits, ups=False ): 
    if not ups:
        offset = tensor.size()[2] -teacher.size()[2]
        div = 1
    else:
        offset = tensor.size()[2] -int( teacher.size()[2] /2 )
        div = 2
    teacher_its = [ getSplitIts( teacher, 0, num_splits[0], ups ) ] 
    teacher_its.append( getSplitIts( teacher, 1, num_splits[1], ups ) )
    teacher_its.append( getSplitIts( teacher, 2, num_splits[2], ups ) )
    #print(teacher_its)
    #print(offset)
    teacher_splits = splitArray( teacher, teacher_its )
    tensor_splits = splitArray( tensor, teacher_its, offset, div )
    return tensor_splits, teacher_splits


def splitArray( tensor, num_splits, offset = 0, div = 1 ):
    dim0_its = num_splits[0]
    dim1_its = num_splits[1]
    dim2_its = num_splits[2]
    split_tensor = []
    for dim0 in range( len( dim0_its ) -1 ):
        for dim1 in range( len( dim1_its ) -1 ):
            for dim2 in range( len( dim2_its ) -1 ):
                d0 = splitIt( dim0_its, dim0, offset, div )
                d1 = splitIt( dim1_its, dim1, offset, div )
                d2 = splitIt( dim2_its, dim2, offset, div )
                split_tensor.append( split( tensor, d0, d1, d2 ) )
    return split_tensor

def splitIt( its, i, offset, div = 1 ):
    return ( int( its[i] /div ), int( (its[i+1] /div) ) +offset )


def getSplitIts( tensor, dim, num_splits, ups ):
    split_its = [0]
    split_it = math.ceil( tensor.size()[dim+2] /num_splits )
    for it in range( num_splits ):
        end = (it+1)*split_it
        if ups:
            if end %2 != 0:
                end += 1
        split_its.append( min( tensor.size()[dim+2], end ) )
    return split_its


def split( tensor, x_split, y_split, z_split ):
    return tensor[:,:,x_split[0]:x_split[1], y_split[0]:y_split[1], z_split[0]:z_split[1] ]

#import torch
#offset = 2
#div = 2
#a = torch.ones( 1,1,19,53,16 )
#teacher = torch.ones( 1,1,(a.size()[2] -offset *2) *div, (a.size()[3] -offset *2) *div, (a.size()[4] -offset *2) *div )
#print( str(a.size()) + "  "+ str(teacher.size()) +"\n" )
#tensor, teacher = splitInputAndTeacher( a, teacher, (2,2,1), True )
#for t in tensor:
#    print(t.size())
#print( "" )
#for t in teacher:
#    print(t.size())
import numpy as np
#import torch
a = torch.rand( 1,1,100,100,100 )
#print(a)
#a = np.random.randint( 10, size=(1,1,10,10,10) )
print(a)
splits = [2,7,2]
spl = splitInputAndTeacher( a, a, splits )
spl_np = []
it = 0
for sp in spl[0]:
    print( str(it) +": " +str(sp.size()) )
    it += 1
    spl_np.append( sp.data.numpy() )
out = reassemble( spl[0], splits )
print( torch.sum(out -a) )
