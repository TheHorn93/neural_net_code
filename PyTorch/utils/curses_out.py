#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 04:35:15 2018

@author: root
"""

import curses
import timeit

class Display:
    
    def __init__( self, stdscr, net_string, instance_string ):
        self.screen = stdscr
        self.screen.addstr( 0, 0, "Training Network", curses.A_UNDERLINE  )
        self.screen.addstr( 2, 0, net_string, curses.A_BOLD )
        self.screen.addstr( 3, 0, instance_string )
        self.screen.refresh()
        self.avg_time = 0
        self.offset = 7
        self.bt_l_it = 7
        curses.init_pair( 1, curses.COLOR_YELLOW, curses.COLOR_BLACK )
        curses.init_pair( 2, curses.COLOR_GREEN, curses.COLOR_BLACK )
        curses.init_pair( 3, curses.COLOR_RED, curses.COLOR_BLACK )

    def __call__( self ):
        inp = self.screen.getch()
        return inp != ord( 'q' )
    
    def newEpoch( self, epoch, max_epochs ):
        self.st_pt = timeit.default_timer()
        self.cur_epoch = epoch
        self.screen.addstr( 6, 0, "Epoch: ", curses.A_BOLD | curses.color_pair(2) )
        self.screen.addstr( 6, 7, str(epoch) + " of "+ str(max_epochs) )
        self.screen.refresh()
    
    def addBatches( self, bt_nbr, of_slices ):
        for it in range( bt_nbr ):
            self.screen.addstr( self.bt_l_it, 0, "Batch " +str( it ) +":" )
            self.screen.addstr( self.bt_l_it, 9, " 0/" +str( of_slices ), curses.color_pair(3) )
            self.bt_l_it += 1
    
    def addComputed( self, batch_nbr, slce, of_slices ):
        self.screen.addstr( batch_nbr +self.offset, 8, ": " +str( slce +1 ) +"/" +str( of_slices ) )
        self.screen.refresh()

    def endEpoch( self, train_loss ):
        self.ed_pt = timeit.default_timer()
        self.time = self.ed_pt -self.st_pt
        self.avg_time += self.time
        self.screen.addstr( self.bt_l_it +1, 0, "Train Loss: " )
        self.screen.addstr( self.bt_l_it +1, 12, str( train_loss ), curses.color_pair(1) | curses.A_BOLD )
        self.screen.addstr( self.bt_l_it +2, 0, str( self.time ) +" average: " +str( self.avg_time /self.cur_epoch ) )
        self.bt_l_it = self.offset
        self.screen.refresh()
        
