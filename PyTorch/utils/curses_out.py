#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 04:35:15 2018

@author: root
"""

import curses

class Display:
    
    def __init__( self, stdscr ):
        self.screen = stdscr
        self.screen.addstr( 0, 0, "Training Network", curses.A_UNDERLINE  )
        self.screen.refresh()

    def __call__( self ):
        inp = self.screen.getch()
        return inp != ord( 'q' )

def main( stdscr ):
    disp = Display( stdscr )
    while True:
        if not disp():
            break

curses.wrapper( main )