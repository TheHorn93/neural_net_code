# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 03:25:33 2018

@author: JHorn
"""

import numpy as np

class kernel:
    
    def __init__( self, size ):
        self.s = size
    
    def get1DGrad( self ):
        a = np.zeros( self.s )
        wing = (self.s -1) /2
        for it in range( self.s ):
            a[it] = ( it -wing ) /wing
        return a
    
    def get2DGrad( self, invert ):
        a = np.zeros( (self.s,self.s) )
        grad = self.get1DGrad()
        if invert:
            grad = np.flipud( grad )
        for it in range( self.s ):
            a[it,:] = grad
        return a
    
    def get2DGradDiag( self, orientation ):
        a = np.zeros( (self.s,self.s) )
        grad = self.get1DGrad()
        for it in range( self.s *self.s ):
            x, y = int( it /self.s ), it %self.s
            if orientation == 0:
                a[x,y] = (grad[x] +grad[y]) /2
            elif orientation == 1:
                a[x,-1-y] = (grad[x] +grad[y]) /2
            elif orientation == 2:
                a[-1-x,-1-y] = (grad[x] +grad[y]) /2
            else:
                a[-1-x,y] = (grad[x] +grad[y]) /2
        return a
 
    def center2DSorround( self ):
        a = np.zeros( (self.s,self.s) )
        center = int( (self.s -1) /2 )
        grad = np.zeros(self.s)
        for it in range( center+1 ):
            grad[it] = it+1
            grad[-it-1] = it+1
        for it in range( self.s ):
            a[:,it] = grad*grad[it]
        a /= a[center, center]
        return a
    
    def get3DGrad( self, orientation, invert ):
        a = np.zeros( (self.s,self.s,self.s) )
        grad = self.get2DGrad( invert )
        for it in range( self.s ):
            a[it,:,:] = grad
        if orientation == 1:
            a = a.transpose()
        elif orientation == 2:
            for it in range( self.s ):
                a[:,:,it] = grad 
        return a
    
    def get3DGradDiag( self, orientation, invert ):
        a = np.zeros( (self.s,self.s,self.s) )
        grad = self.get2DGradDiag( orientation )
        for it in range( self.s ):
            a[it,:,:] = grad
        if invert:
            a = a.transpose()
        return a
    
    def get3DCenterSorround( self, cylinder ):
        a = np.zeros( (self.s,self.s,self.s) )
        grad = self.center2DSorround()
        for it in range( self.s ):
            a[it,:,:] = grad *grad[it, int( (self.s-1) /2)]
        return a

    def fill8( self, scale = 0.2 ):
        a = np.zeros( (8,1,self.s,self.s,self.s) )
        l = [[0,False],[1,False],[2,False],[0,True],[1,True],[2,True]]
        for it in range( 6 ):
            a[it,0,:,:,:] = self.get3DGrad( l[it][0], l[it][1] )
        a[6,0,:,:,:] = self.get3DCenterSorround(0)
        a[7,0,:,:,:] = np.random.rand( self.s,self.s,self.s )
        return (a*scale,None)
    
    def fill8Diag( self, scale = 0.2 ):
        a = np.zeros( (8,1,self.s,self.s,self.s) )
        l1, l2 = range(4), [False,True]
        for ori in l1:
            for inv in l2:
                it = ori + inv*4
                a[it,0,:,:,:] = self.get3DGradDiag( ori, inv )
        return (a*scale, None)
      
    def fill16( self, scale = 0.2 ):
        a = np.zeros( (16,1,self.s,self.s,self.s) )
        k1, _ = self.fill8( scale )
        k2, _ = self.fill8Diag( scale )
        a[0:8,:,:,:,:] = k1
        a[8:16,:,:,:,:] = k2
        return (a, None)
        
#init = kernel( 5 )
#print( init.get3DCenterSorround(0) )
#print( init.fill8() )
#for axis in range(3):
#    for inv in [False, True]:
#        print( str( init.get3DGradDiag(axis, inv) ) +"\n----------\n" )
#for ori in range(3):
#    a = init.get3DGrad(ori, False)
#    print( str( a ) +"\n----------\n" )    