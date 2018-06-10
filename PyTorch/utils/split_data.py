#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 21:21:45 2018

@author: root
"""

import math

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
    teacher_splits = splitArray( teacher, teacher_its )
    tensor_splits = splitArray( tensor, teacher_its, offset, div )
    return ( tensor_splits, teacher_splits )


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