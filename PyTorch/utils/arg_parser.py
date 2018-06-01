#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 01:02:31 2018

@author: root
"""

import sys
import argparse

class SessionArgumentParser( ):
    
    def __init__( self ):
        self.parser = argparse.ArgumentParser()
        self.mode = self.parser.add_subparsers( help="Command modes" )
        self.parser.add_argument( "-d", "--device", help="Device to run", nargs=1, default=["cuda"] )
        self.feed = self.mode.add_parser( "feed", help="Feed pretrained network" )
        self.feed.set_defaults( mode="feed" )        
        self.train = self.mode.add_parser( "train", help="Train network" )
        self.train.set_defaults( mode="train" )
        
    def __call__( self, inp=sys.argv ):
        return self.parser.parse_args( inp )
        
    
class FeedParser():
    
    def __init__( self ):
        self.parser = argparse.ArgumentParser( prog="Feed network(s)" )
        self.mode = self.parser.add_subparsers( help="Add log or run" )
        self.add = self.mode.add_parser( "add", help="Add log to list" )
        self.add.set_defaults( mode="add" )
        self.add.add_argument("-l", "--log", help="Path to log to load from (required)" , nargs=1, required=True )
        self.add.add_argument("-e", "--epoch", help="Specific epoch to load from", nargs=1, default=None )
        
        self.run = self.mode.add_parser( "run", help="Run all added networks" )
        self.run.set_defaults( mode="run" )
        self.run.add_argument("-d", "--data", help="Which type of data to use", nargs='+')
        self.run.add_argument("-r", "--real", help="Use real data as input", action="store_true", default=False )
        self.run.add_argument("-s", "--synth", help="Use synthetic data as input", action="store_true", default=False )

    def __call__( self, inp ):
        return self.parser.parse_args( inp )


class TrainParser():
    
    def __init__( self ):
        self.parser = argparse.ArgumentParser( prog="Train network(s)" )
        self.parser.add_argument( "--mode", default='None' )
        self.mode = self.parser.add_subparsers( help="Add network or run all" )
        self.add = self.mode.add_parser( "add", help="Add log to list" )
        self.add.set_defaults( mode="add" )
        self.type_gr = self.add.add_mutually_exclusive_group( required=True )
        self.type_gr.add_argument("-n", "--net", help="Create new network with parameters" , nargs=argparse.REMAINDER )
        self.type_gr.add_argument("-l", "--log", help="Load network from logfile", nargs=1 )
        self.add.add_argument( "-d", "--data", help="Root type used for training", nargs='+', required=True )
        self.add.add_argument( "-lr", "--learning_rate", help="Network learning rate", nargs=1, type=float, required=True )
        self.add.add_argument( "-e", "--epochs", help="Epochs to train", nargs=1, type=int, required=True )
        self.add.add_argument( "-ls", "--loss", help="Loss function to use and parameters", nargs=1, default=["cross_entropy"] )
        self.add.add_argument( "-eg", "--epoch_gates", help="Epoch gates for loss method", nargs='+', type=int )
        self.add.add_argument( "-op", "--optimizer", help="Optimizer to use", nargs=1, default=["adam"] )
        self.add.add_argument( "-sl", "--slices", help="Numbers of slices per run", nargs=1, type=int, default=[1] )
        self.add.add_argument( "-bs", "--batch_size", help="Number of data processes before gradient applience", nargs=1, type=int, default=[12] )
        
        self.run = self.mode.add_parser( "run", help="Run all added networks" )
        self.run.set_defaults( mode="run" )
        self.run.add_argument("-i", "--interactive", help="Allow interactive keyboard input during execution")
        
        self.show = self.mode.add_parser( "show", help="Show current networks in queue" )
        self.show.set_defaults( mode="show" )

      
    def __call__( self, inp ):
        return self.parser.parse_args( inp )
 

class NetworkParser():

    def __init__( self ):
        self.net_parser = argparse.ArgumentParser( prog="n-Layer conv net")
        #self.net_parser.add_argument( "-c", "--conv", help="3D Convolutional Layer", dest='layers', nargs=3, action='append', metavar=('KERNEL_SIZE', 'NUM_KERNELS', 'ACT_FUNC') )
        #self.net_parser.add_argument( "-bn", "--batch_norm", help="Batch Norm Layer", dest='layers', action='append_const', const='bn' )
        #self.net_parser.add_argument( "-rc", "--res_conn", help="Residual connection from -> pos", dest='layers', nargs=1, action='append', metavar=('FROM') )
        self.net_parser.add_argument( "-l", "--layer", help="Add Convolutional Layer", nargs='*', action='append', dest='layers')
        
    def __call__( self, inp ):
        return self.net_parser.parse_args( inp )
        
#par = NetworkParser()
#print( par( "-l 5 8 relu -l 5 1 sigmoid bt".split() ) )