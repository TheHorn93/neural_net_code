#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:15:05 2018

@author: jhorn
"""
import subprocess
import numpy as np
import time
import os

def createPerlin( size, iteration, frequency, num_threads, allow_output=False ):
    start_time = time.time()
    size_string = str( size[0] ) +" " +str( size[1] ) +" " +str( size[2] )
    file_name = "temp.raw"
    args = "bin/Release/PerlinC++ -p " +file_name +" -s " +size_string +" -i " +str(iteration) +" -f " +str(frequency) +" -t " +str(num_threads)
    args = args.split( )
    subprc = subprocess.Popen(args, stdout=subprocess.PIPE)
    subprc.wait()
    if allow_output:
        output = subprc.stdout.read()
        print( output )
    perlin_noise = np.fromfile( file_name, dtype=np.float32 ).reshape( size )
    end_time = time.time()
    exec_time = end_time -start_time
    print( "Needed " +str( exec_time ) )
    os.remove( file_name )
    perlin_noise -= np.amin( perlin_noise )
    perlin_noise /= np.amax( perlin_noise )
    np.save( "perlin_output", perlin_noise )
    return perlin_noise

if __name__ == "__main__":
    sizes = [ (20,20,20), (40,40,40), (80,80,80), (160,160,160), (200,200,200) ]
    times = []
    #for size in sizes:
    for it in range( 100 ):
        _ = createPerlin( [100,100,100], 4, 4, 4, False )
    print( times )
