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
        self.screen.clear()
        self.screen.addstr( 0, 0, "Training Network", curses.A_UNDERLINE  )
        self.screen.addstr( 2, 0, net_string, curses.A_BOLD )
        self.screen.addstr( 3, 0, instance_string )
        self.screen.refresh()
        self.avg_time = 0
        self.offset = 8
        self.bt_l_it = self.offset
        self.save_offset = 23
        curses.init_pair( 1, curses.COLOR_YELLOW, curses.COLOR_BLACK )
        curses.init_pair( 2, curses.COLOR_GREEN, curses.COLOR_BLACK )
        curses.init_pair( 3, curses.COLOR_RED, curses.COLOR_BLACK )
        self.log_file = open( "./session.log", 'w' )

    def __call__( self ):
        self.screen.refresh()
    
    def clrLine( self, line, offset = 0 ):
        self.screen.move( line, offset )
        self.screen.clrtoeol()
        
    def clrBlock( self, lines, offset ):
        for line in range( lines[0], lines[1]+1 ):
            self.clrLine( line, offset )
    
    def newEpoch( self, epoch, max_epochs, num_runs ):
        self.st_pt = timeit.default_timer()
        self.cur_epoch = epoch
        self.num_runs = num_runs
        self.num_cpl = 1
        self.clrLine( 6, 0 )
        self.screen.addstr( 6, 0, "Epoch: ", curses.A_BOLD | curses.color_pair(2) )
        self.screen.addstr( 6, 7, str(epoch) + " of "+ str(max_epochs) )
        self.screen.addstr( 7, 0, "Batch: 0/" +str(self.num_runs) )
        self.screen.refresh()
    
    def addBatches( self, bt_nbr, of_slices, noise_lvl ):
        self.bt_l_it = self.offset
        self.bt_st_pt = timeit.default_timer()
        self.clrLine( 7 )
        self.screen.addstr( 7, 0, "Batch: " +str( self.num_cpl ) +"/" +str(self.num_runs) + " Noise LvL: " +str(noise_lvl) )
        for it in range( bt_nbr ):
            self.clrLine( self.bt_l_it, 0 )
            self.screen.addstr( self.bt_l_it, 0, "Scan " +str( it ) +":" )
            self.screen.addstr( self.bt_l_it, 9, " 0/" +str( of_slices ), curses.color_pair(3) )
            self.bt_l_it += 1
    
    def addComputed( self, batch_nbr, slce, of_slices ):
        self.clrLine( batch_nbr +self.offset, 10 )
        slc = slce +1
        if slc == of_slices:
            clr = 2
        else:
            clr = 0
        self.screen.addstr( batch_nbr +self.offset, 10, str( slc ) +"/" +str( of_slices ), curses.color_pair(clr) )
        self.screen.refresh()

    def endBatch( self, train_loss ):
        self.bt_ed_pt = timeit.default_timer()
        self.time = self.bt_ed_pt -self.bt_st_pt
        self.clrLine( self.bt_l_it +1, 0 )
        self.screen.addstr( self.bt_l_it +1, 0, "Batch Loss: " +str(train_loss) +" Run time: " +str( self.time ) )
        self.num_cpl += 1
        #self.screen.addstr( 7, 0, "Batch: " +self.num_cpl+ "/" +str(self.num_runs) )
        self.screen.refresh()

    def endEpoch( self, train_loss ):
        self.ed_pt = timeit.default_timer()
        self.time = self.ed_pt -self.st_pt
        self.avg_time += self.time
        self.clrBlock( (self.bt_l_it +1, self.bt_l_it +2), 0 )
        self.screen.addstr( self.bt_l_it +2, 0, "Train Loss: " )
        self.screen.addstr( self.bt_l_it +2, 12, str( train_loss ), curses.color_pair(1) | curses.A_BOLD )
        self.screen.addstr( self.bt_l_it +3, 0, str( self.time ) +" average: " +str( self.avg_time /self.cur_epoch ) )
        self.bt_l_it = self.offset
        self.screen.refresh()

    def addLine( self, line, opt='a' ):
        self.clrLine( self.offset, 0 )
        self.screen.addstr( self.offset, 0, line )
        self.log_file.write( line +"\n" )
        self.screen.refresh()


#import time
#def main( stdscr ):
#    scr = Display( stdscr, "net", "inst" )
#    for e in range( 1, 21 ):
#        scr.newEpoch( e, 20 )
#        scr.addBatches( 12, 10 )
#        for it in range( 12 ):
#            for jt in range( 10 ):
#                time.sleep( 0.1 )
#                scr.addComputed( it, jt, 10 )
#        scr.endEpoch( 1 )
    
#curses.wrapper( main )
