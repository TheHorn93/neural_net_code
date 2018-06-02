#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:12:43 2018

@author: root
"""

import sys
import torch
import curses

sys.path.insert(0,"/home/work/horn/code/neural_net_code/PyTorch/networks/")
#sys.path.insert(0,"./networks/")
network = __import__( 'n-layer_conv_net' )

sys.path.insert(1, '/home/work/horn/code/neural_net_code/PyTorch/utils/' )
#sys.path.insert(1,"./utils/")
import arg_parser
import losses
import optimizer
import logger
import bc4_data_loader as data_loader
import training
import curses_out


logging_path = "/home/work/horn/data/"
input_path = "/home/work/uzman/Real_MRI/manual-reconstruction/"
real_scan_path = "/home/work/uzman/Real_MRI/Lupine_small/01_tiff_stack/"

class Instance:
    
    class ProtoInstance:
        
        def __init__( self, network_set, loss, opt, lr, epochs, data, batch_size, slices, epoch_gates ):
            self.network_set = network_set
            self.loss = loss[0]
            self.opt = opt[0]
            self.lr = lr[0]
            self.epochs = epochs[0]
            self.data = data
            self.batch_size = batch_size[0]
            self.slices = slices[0]
            self.epoch_gates = epoch_gates
            
    def __init__( self, network_set, loss, opt, lr, epochs, data, batch_size, slices, epoch_gates=(None,None) ):
        self.proto_set = self.ProtoInstance( network_set, loss, opt, lr, epochs, data, batch_size, slices, epoch_gates )
        
    def __call__( self, stdscr, device ):
        print( "Creating training instance" )
        self.parseArgs( self.proto_set.loss, self.proto_set.opt, self.proto_set.epochs, self.proto_set.data, self.proto_set.epoch_gates )
        self.lr = self.proto_set.lr
        self.batch_size = self.proto_set.batch_size
        self.slices = self.proto_set.slices
        self.net = network.Network( self.proto_set.network_set[0], self.proto_set.network_set[1] )
        self.log = logger.DummyLogger( logging_path )
        instance_string = "Training: loss="+ str( self.loss ) +", lr=" +str( self.lr ) + ", opt="+ str( self.opt ) +"\n" +"Data: data="+ str( self.data ) +", batch_size=" +str( self.batch_size ) +", slices=" +str( self.slices ) 
        self.log.masterLog( self.net.getStructure(), instance_string )
        self.loaders = []
        for data_str in self.data:
            self.loaders.append( data_loader.BatchLoader( input_path +data_str, self.net.teacher_offset, device ) )
        if device is not None:
            self.net.cuda( device )
        training.training( curses_out.Display( stdscr, self.net.getStructure(), instance_string ), self.log, self.net, self.loaders, self.loss, self.opt, self.lr, self.epochs, self.batch_size, self.slices )
                     
    def parseArgs( self, loss, opt, epochs, data, epoch_gates=(None,None) ):
        if loss == "cross_entropy":
            if epoch_gates == None:
                self.loss = losses.CrossEntropy( epochs )
            elif epoch_gates[1] == None:
                self.loss = losses.CrossEntropy( epoch_gates[0] )
            else:
                self.loss = losses.CrossEntropyDynamic( epoch_gates[0], epoch_gates[1] )
        if opt == "adam":
            self.opt = optimizer.AdamOptimizer()
        self.data = []
        for data_set in data:   
            if data_set == 'lupine_small':
                self.data.append( 'lupine_small_xml/' )
            elif data_set == 'lupine_22':
                self.data.append( 'Lupine_22august/' )
            elif data_set == 'gtk':
                self.data.append( 'gtk/' )
        self.epochs = epochs


class Session:
    
    def __init__( self, inp, stdscr ):
        self.screen = stdscr
        parser = arg_parser.SessionArgumentParser() 
        args = parser( inp )
        self.device = args.device[0]
        if args.mode == "feed":
            self.feed = True
            self.parser = arg_parser.FeedParser()
        elif args.mode == "train":
            self.feed = False
            self.parser = arg_parser.TrainParser()
        self.mode = 'add'
        self.net_parser = arg_parser.NetworkParser()
        self.instances = []
            
    def __call__( self ):
        while( self.mode != 'run' ):
            new_cmd = input( "Next action: " )
            try:
                cmd = self.parser( new_cmd.split() )
                self.mode = cmd.mode
                if self.mode == 'add':
                    if cmd.net is not None:
                        self.addInstance( cmd )
                    elif cmd.log is not None:
                        self.addInstanceFromLog( cmd )
                if self.mode == 'show':
                    self.show()
                elif self.mode == 'run':
                    self.runSession()
            except SystemExit:
                pass
           
   
    def addInstance( self, args ):
        """ Get line and test it in n-layer_conv_net parser """
        try:
            net = self.net_parser( args.net )
            new_inst = Instance( ( net.layers, args.net ), args.loss, args.optimizer, args.learning_rate, args.epochs, args.data, args.batch_size, args.slices, args.epoch_gates )
            self.instances.append( new_inst )
            print( "Adding training instance: " +str( args ) )    
        except SystemExit:
            pass

    def addInstanceFromLog( self, log_file ):
        """ Load line and send to add Network """
        print( "Loading from: " +log_file[0] )
        self.addNetwork( "fromlogfile" )
        
    def show( self ):
        for net_string in self.instances:
            print( net_string[0] )
        
    def runSession( self ):
        print( "Starting Session: " )
        for instance in self.instances:
            instance( self.screen, torch.device( self.device ) )
  

def main( stdscr ):
    ses = Session( sys.argv[1:], stdscr )
    ses()    
    
if __name__ == '__main__':
    curses.wrapper( main )
