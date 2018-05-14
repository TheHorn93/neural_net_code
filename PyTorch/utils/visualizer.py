# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 23:20:13 2018

@author: JHorn
"""

import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly
import plotly.graph_objs as go
import cv2

def flattenWeights( weights ):
    flt_weights = np.zeros( [(weights.shape[1] +1) *weights.shape[0]-1, weights.shape[2]*(weights.shape[3]+1)-1] )
    for it in range( weights.shape[0] ):
        start_x = weights.shape[1]*it +it
        for jt in range( weights.shape[3] ):
            start_y = weights.shape[2]*jt +jt
            flt_weights[start_x : start_x +weights.shape[1], start_y : start_y +weights.shape[2] ] = weights[it,:,:,jt]
    return flt_weights
        

def visualizeWeights( weight_list ):
    fig_list = []
    for layer in weight_list:
        fig = plt.figure()
        dpi = fig.get_dpi()
        fig.set_size_inches( max(layer[0].shape[2]*100,200)/dpi, ( layer[0].shape[1]* layer[0].shape[0]*85 +20)/dpi )
        n = int( (layer[0].shape[2] +1) /2 )
        grid = gridspec.GridSpec( layer[0].shape[0] *layer[0].shape[1] *layer[0].shape[2] +n, 1, figure=fig )
        
        maxe = np.amax( layer[0] )
        mine = np.amin( layer[0] )
        step = layer[0].shape[2]*layer[0].shape[1]
        for it in range( layer[0].shape[0] ):
            grid_st = step*it
            ax = fig.add_subplot( grid[grid_st:grid_st+step] )
            mat = flattenWeights( layer[0][it,:,:,:,:] )
            im = ax.imshow( mat, cmap='Greys', vmax = maxe, vmin = mine)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title( "Bias: " + str( layer[1][it] ) )
        ax_cbr = fig.add_subplot( grid[-n:] )
        fig.colorbar( im, cax = ax_cbr, orientation='horizontal' )                
        fig.tight_layout()
        fig_list.append( fig )
        #plt.close(fig)
    return fig_list
        

def visualizeWeights2( weight_list ):
    fig_list = []
    for layer in weight_list:
        mlp.rcParams.update({'font.size':7})
        fig = plt.figure()
        n = int( (layer[0].shape[2] +1) /2 )
        main_grid = gridspec.GridSpec( (layer[0].shape[1]*layer[0].shape[4] +n) *layer[0].shape[0], 1, figure=fig )
        dpi = fig.get_dpi()
        fig.set_size_inches( layer[0].shape[2]*120/dpi, ( layer[0].shape[1]*150 +25)/dpi )
        maxe = np.amax( layer[0] )
        mine = np.amin( layer[0] )
        for it in range( layer[0].shape[0] ):
            for jt in range( layer[0].shape[1] ):
                start = layer[0].shape[2]*it
                grid = gridspec.GridSpecFromSubplotSpec( 1, layer[0].shape[4], subplot_spec=main_grid[start:start +layer[0].shape[2]] )
                for kt in range( layer[0].shape[4] ):
                    ax = fig.add_subplot( grid[kt] )
                    im = ax.imshow( layer[0][it,jt,:,:,kt], cmap='Greys', vmax = maxe, vmin = mine)
                    ax.set_xticks([])
                    ax.set_yticks([])
            ax.set_title( "Bias: " + str( layer[1][0] ) )
        ax_cbr = fig.add_subplot( main_grid[-n:] )
        fig.colorbar( im, cax = ax_cbr, orientation='horizontal' )
        fig.tight_layout()
        fig_list.append( fig )
    return fig_list


def getRawImage( fig, for_vis=False ):
    canvas = FigureCanvas(fig)
    canvas.draw()
    w,h = canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape( [h,w,3] )
    if for_vis:
        image = np.swapaxes( image, 0, 2 )
        image = np.swapaxes( image, 1, 2 )
    return image


def visualizeOutput( input_data, output_path, axis=2, invert=False ):
    max_val = np.amax( input_data )
    min_val = np.amin( input_data )
    print( "[" + str(min_val) + ":" + str(max_val) + "]" )
    input_data -= min_val
    fac = 255 / ( max_val-min_val )
    print( str(max_val-min_val) + "->" + str(fac) )
    for it in range( input_data.shape[axis] ):
        if axis == 0:
            im = input_data[it,:,:] *fac
        elif axis == 1:
            im = input_data[:,it,:] *fac
        else:
            im = input_data[:,:,it] *fac
        if invert:
            im = np.add( 255, -im )
        cv2.imwrite( output_path + str(it) + ".png", im )
        
 
def getCombinedSlice( input_data, start, thickness ): 
    dim1 = input_data.shape[0]
    dim2 = input_data.shape[1]
    output_slice = np.zeros( [dim1,dim2] )
    #it_scale = 1 / thickness
    for it in range( start, start + thickness ):
        #scale = it_scale * ( start + thickness - it )
        #output_slice = output_slice + ( input_data[it,:,:] * scale )
        output_slice = np.maximum( output_slice, input_data[:,:,it] )
    return output_slice

       
def multiOutput4( input_batch ):
    fig, ( l1, l2, l3, l4 ) = plt.subplots( 1,4 )
    dpi = fig.get_dpi()
    fig.set_size_inches( 1000/dpi, 500/dpi )
    thickness = input_batch[0].shape[4] -1 
    #print(thickness)
    l1.imshow( getCombinedSlice( input_batch[0][0,0,:,:,:], 0, thickness ), cmap="Greys", vmin=0.0, vmax=1.0 )
    l2.imshow( getCombinedSlice( input_batch[1][0,0,:,:,:], 0, thickness ), cmap="Greys", vmin=0.0, vmax=1.0 )
    l3.imshow( getCombinedSlice( input_batch[2][0,0,:,:,:], 0, thickness ), cmap="Greys", vmin=0.0, vmax=1.0 )
    l4.imshow( getCombinedSlice( input_batch[3][0,0,:,:,:], 0, thickness ), cmap="Greys", vmin=0.0, vmax=1.0 )
    fig.tight_layout()
    return fig
    
    
def cvMultiSlice( input_data, output_path, axis=2, invert=False ):
    max_val = np.amax( input_data )
    min_val = np.amin( input_data )
    print( "[" + str(min_val) + ":" + str(max_val) + "]" )
    log = open( output_path +"scale.txt", "w" )
    log.write( str( min_val ) +" - " +str( max_val ) )
    log.close
    input_data -= min_val
    fac = 255 / ( max_val-min_val )
    print( str(max_val-min_val) + "->" + str(fac) )
    for it in range( input_data.shape[axis] ):
        if axis == 0:
            im = input_data[it,:,:] *fac
        elif axis == 1:
            im = input_data[:,it,:] *fac
        else:
            im = input_data[:,:,it] *fac
        if invert:
            im = np.add( 255, -im )
        cv2.imwrite( output_path + str(it) + ".png", im )
        
def cvSaveStack( stack, output_path, invert=False ):
    max_val = 1.0
    min_val = 0.0
    fac = 255 /( max_val - min_val )
    for it in range( len(stack) ):
        im = stack[it] *fac
        cv2.imwrite( output_path +str(it) +".png", im )
    
def scatterRoot( stack, path ):
    pr_stack = np.where( stack > 0.5, stack, 0 )
    c = pr_stack.nonzero()
    fig = go.Figure(data=[go.Scatter3d(x=c[0],y=c[1],z=c[2], mode='markers', marker=dict(symbol='circle-dot', size=1))],
                    layout=go.Layout(scene=dict(aspectmode='data')))
    plotly.offline.plot(fig, filename=path)
#a = np.random.rand( 1,8,1,1,1,1)
#print(a)
#b = flattenWeights(a)
#print(b)
#fig_list = visualizeWeights( [(a,[1,2,1,0,1,1,1,1])] )
#fig_list[0].show()