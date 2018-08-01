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



class DisplayFullSet:
    
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
        curses.init_pair( 4, curses.COLOR_BLUE, curses.COLOR_BLACK )
        self.log_file = open( "./session.log", 'w' )
      
    def initParams( self, max_epochs, sc_per_bt, num_scans ):
        self.max_eps = max_epochs
        self.sc_per_bt = sc_per_bt
        self.num_scans = num_scans
        self.scan_lines = ( 8, 8 +sc_per_bt )
        #self.screen.addstr( 8 +num_scans +3, 0, " " )
        self.sc_comp = 0
        self.it = 0
        
    def clrLine( self, line, offset = 0 ):
        self.screen.move( line, offset )
        self.screen.clrtoeol()
        
    def clrBlock( self, lines, offset = 0 ):
        for line in range( lines[0], lines[1]+1 ):
            self.clrLine( line, offset )
            
    def writeLine( self, line, string, clr_pair=0 ):
        self.clrLine( line )
        self.screen.addstr( line, 0, curses.color_pair( clr_pair ) )
        
        
    def newEpoch( self, epoch ):
        self.cur_epoch = epoch
        self.screen.addstr( 6, 0, "Epoch: ", curses.A_BOLD | curses.color_pair(2) )
        self.screen.addstr( 6, 7, str(epoch) + " of "+ str(self.max_eps) )
        self.sc_comp = 0
        self.createProgressBar( 7, 0 )
        self.screen.refresh()

    def createProgressBar( self, line, scan_nbr, length_bar=30 ):
        self.clrLine( line )
        progress = float(scan_nbr) /float(self.num_scans)
        if progress < 0.25:
            clr = 3
        elif progress < 0.5:
            clr = 1
        elif progress < 0.75:
            clr = 2
        else:
            clr = 4 
        bar_fill = int( round( length_bar *progress ) )
        prog_str = "{:4}".format( str(scan_nbr) ) +"/" +str(self.num_scans)
        bar_str = "["
        for it in range( bar_fill+1 ):
            bar_str += '='
        if bar_fill < length_bar:
            bar_str += '>'
        for it in range( bar_fill+1, length_bar ):
            bar_str += ' '
        bar_str += ']'
        self.screen.addstr( line, 0, "Progress: " )
        self.screen.addstr( line, 10, prog_str, curses.A_BOLD | curses.color_pair(clr) )
        self.screen.addstr( line, 20, bar_str, curses.A_BOLD )
        self.screen.addstr( line, 20 +length_bar +5, str(int(progress *100)) +"%" )
            
    def addBatch( self ):
        self.st_pt = timeit.default_timer()
        for it in range( self.sc_per_bt ):
            cur_line = self.scan_lines[0] +it
            self.clrLine( cur_line )
            self.screen.addstr( cur_line, 0, "  Scan " +"{:>2}".format( str(it+1) ) +": ", curses.A_BOLD )
            self.screen.addstr( cur_line, 11, "{:>4}".format( "0%" ), curses.color_pair(3) )
        self.screen.refresh()
            
    def computed( self, it, prog ):
        cur_line = self.scan_lines[0] +it
        self.clrLine( cur_line )
        if prog < 1.0:
            clr = 3
        else:
            clr = 2
        self.screen.addstr( cur_line, 0, "  Scan " +"{:>2}".format( str(it+1) ) +": ", curses.A_BOLD )
        self.screen.addstr( cur_line, 11, "{:>4}".format( str(int(prog*100)) +"%" ), curses.color_pair(clr) )
        self.screen.refresh()
        
    def endBatch( self, train_loss ):
        cur_line = self.scan_lines[1] +2
        self.bt_ed_pt = timeit.default_timer()
        self.time = self.bt_ed_pt -self.st_pt
        self.avg_time += self.time
        self.it += 1
        self.clrBlock( ( cur_line, cur_line +2 ) )
        self.screen.addstr( cur_line, 0, "Train Loss: " )
        self.screen.addstr( cur_line, 12, str(train_loss), curses.A_BOLD | curses.color_pair(1) ) 
        self.screen.addstr( cur_line +1, 0, "  Run Time: ", curses.color_pair(4) )
        self.screen.addstr( cur_line +1, 12, str( self.time ) ) 
        self.screen.addstr( cur_line +2, 0, "  Avg Time: ", curses.color_pair(4) )
        self.screen.addstr( cur_line +2, 12, str( self.avg_time /self.it ) ) 
        self.sc_comp += self.sc_per_bt
        self.createProgressBar( 7, self.sc_comp )
        self.screen.refresh()
        
        
        

#import time
#def main( stdscr ):
#    scr = DisplayFullSet( stdscr, "net", "inst" )
#    scr.initParams( 100, 10, 100 )
#    for e in range( 1, 21 ):
#        scr.newEpoch( e )
#        for bt in range( 10 ):
#            scr.addBatch()
#            for it in range( 10 ):
#            #scr.createProgressBar( 7, it, 30 )
#                scr.computed( it %10, 1 )
#                time.sleep(0.1)
#            scr.endBatch( 1.0 )
        
    
#curses.wrapper( main )
