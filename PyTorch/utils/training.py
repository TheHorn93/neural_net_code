#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 05:19:33 2018

@author: root
"""

def training( log, net, loader_list, loss, opt, lr, epochs, batch_size, slices ):
    print( "Starting Training for " +str( epochs ) )
    optimizer = opt( net, lr )
    for epoch in range( epochs ):
        print( epoch )