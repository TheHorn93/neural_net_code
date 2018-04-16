# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 03:25:33 2018

@author: JHorn
"""

import numpy as np

class kernel3:
    
    def edgeFilterFB( scale = 0.2 ):
        a = np.array( [[[-1,-1,-1],[0,0,0],[1,1,1]],
                       [[-1,-1,-1],[0,0,0],[1,1,1]],
                       [[-1,-1,-1],[0,0,0],[1,1,1]]] )
        return a*scale
    
    def edgeFilterBF( scale = 0.2 ):
        a = np.array( [[[1,1,1],[0,0,0],[-1,-1,-1]],
                       [[1,1,1],[0,0,0],[-1,-1,-1]],
                       [[1,1,1],[0,0,0],[-1,-1,-1]]] )
        return a*scale
    
    def edgeFilterLR( scale = 0.2 ):
        a = np.array( [[[-1,0,1],[-1,0,1],[-1,0,1]],
                       [[-1,0,1],[-1,0,1],[-1,0,1]],
                       [[-1,0,1],[-1,0,1],[-1,0,1]]] )
        return a*scale
    
    def edgeFilterRL( scale = 0.2 ):
        a = np.array( [[[1,0,-1],[1,0,-1],[1,0,-1]],
                       [[1,0,-1],[1,0,-1],[1,0,-1]],
                       [[1,0,-1],[1,0,-1],[1,0,-1]]] )
        return a*scale
    
    def edgeFilterTB( scale = 0.2 ):
        a = np.array( [[[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
                       [[0,0,0],[0,0,0],[0,0,0]],
                       [[1,1,1],[1,1,1],[1,1,1]]] )
        return a*scale
    
    def edgeFilterBT( scale = 0.2 ):
        a = np.array( [[[1,1,1],[1,1,1],[1,1,1]],
                       [[0,0,0],[0,0,0],[0,0,0]],
                       [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]] )
        return a*scale
    
    def edgeFilterLBRT( scale = 0.2 ):
        a = np.array( [[[1,1,0],[1,1,0],[1,1,0]],
                       [[1,0,-1],[1,0,-1],[1,0,-1]],
                       [[0,-1,-1],[0,-1,-1],[0,-1,-1]]] )
        return a*scale
    
    def edgeFilterRTLB( scale = 0.2 ):
        a = np.array( [[[-1,-1,0],[-1,-1,0],[-1,-1,0]],
                       [[-1,0,1],[-1,0,1],[-1,0,1]],
                       [[0,1,1],[0,1,1],[0,1,1]]] )
        return a*scale
    
    def init3x3( kernels =8 ):
        a = np.zeros( [kernels,1,3,3,3] )
        a[0,0,:,:,:] = kernel3.edgeFilterFB()
        a[1,0,:,:,:] = kernel3.edgeFilterBF()
        a[2,0,:,:,:] = kernel3.edgeFilterLR()
        a[3,0,:,:,:] = kernel3.edgeFilterRL()
        a[4,0,:,:,:] = kernel3.edgeFilterTB()
        a[5,0,:,:,:] = kernel3.edgeFilterBT()
        a[6,0,:,:,:] = kernel3.edgeFilterLBRT()
        a[7,0,:,:,:] = kernel3.edgeFilterRTLB()
        return a
    
    def fillList( layers=2, scale=0.2 ):
        w_list = []
        w_list.append( ( kernel3.init3x3() *scale, np.random.rand(8) *scale) )
        w_list.append( ( np.random.rand( 1,8,3,3,3 ) *scale, np.random.rand(1) *scale) )
        return w_list
        

class kernel5: 
    
    def edgeFilterFB( scale = 0.2 ):
        a = np.array( [[[-1,-1,-1,-1,-1],[-0.5,-0.5,-0.5,-0.5,-0.5],[0,0,0,0,0],[0.5,0.5,0.5,0.5,0.5],[1,1,1,1,1]],
                       [[-1,-1,-1,-1,-1],[-0.5,-0.5,-0.5,-0.5,-0.5],[0,0,0,0,0],[0.5,0.5,0.5,0.5,0.5],[1,1,1,1,1]],
                       [[-1,-1,-1,-1,-1],[-0.5,-0.5,-0.5,-0.5,-0.5],[0,0,0,0,0],[0.5,0.5,0.5,0.5,0.5],[1,1,1,1,1]],
                       [[-1,-1,-1,-1,-1],[-0.5,-0.5,-0.5,-0.5,-0.5],[0,0,0,0,0],[0.5,0.5,0.5,0.5,0.5],[1,1,1,1,1]],
                       [[-1,-1,-1,-1,-1],[-0.5,-0.5,-0.5,-0.5,-0.5],[0,0,0,0,0],[0.5,0.5,0.5,0.5,0.5],[1,1,1,1,1]]] )
        return a*scale
    
    #TODO Rest
