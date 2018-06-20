#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:12:43 2018

@author: root
"""

import sys
import torch
import curses
import time
import os

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
import validation


logging_path = "/home/work/horn/data/"
input_path = "/home/work/uzman/Real_MRI/manual-reconstruction/"
real_scan_path = "/home/work/uzman/Real_MRI/manual-reconstruction/"

debug_mode = False

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
            self.slices = slices
            self.epoch_gates = epoch_gates
            
    def __init__( self, network_set, loss, opt, lr, epochs, data, batch_size, slices, epoch_gates=(None,None) ):
        self.proto_set = self.ProtoInstance( network_set, loss, opt, lr, epochs, data, batch_size, slices, epoch_gates )
        
    def __call__( self, stdscr ):
        print( "Creating training instance" )
        device = "cuda"
        self.parseArgs( self.proto_set.loss, self.proto_set.opt, self.proto_set.epochs, self.proto_set.data, self.proto_set.epoch_gates )
        self.lr = self.proto_set.lr
        self.batch_size = self.proto_set.batch_size
        self.slices = self.proto_set.slices
        self.net = network.Network( self.proto_set.network_set[0], self.proto_set.network_set[1] )
        if debug_mode:
            self.log = logger.DummyLogger( logging_path )
        else:
            self.log = logger.Logger( logging_path )
        instance_string = "Training: loss="+ str( self.loss ) +", lr=" +str( self.lr ) + ", opt="+ str( self.opt ) +"\n" +"Data: data="+ str( self.data ) +", batch_size=" +str( self.batch_size ) +", slices=" +str( self.slices ) 
        self.log.masterLog( self.net.getStructure(), instance_string )
        self.loaders = []
        for data in self.data:
            self.loaders.append( data_loader.BatchLoader( input_path, data, self.net.teacher_offset, device ) )
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
        self.data = data
        self.epochs = epochs


class FeedInstance:
    
    def __init__( self, log, epoch ):
        self.log = logger.Log( logging_path +log +"/" )
        self.epoch = epoch
        self.net_parser = arg_parser.NetworkParser()
        
    def __call__( self, data, data_usage ):
        print( "Loading Log: " +self.log.log_path ) 
        epoch_str = "epoch_" +str( self.epoch ) +"/"
        model = self.log.getNetwork()
        print( "Generating network: " +model )
        net = network.Network( self.net_parser( model.split() ).layers )
        weights = self.log.getWeights( epoch_str )
        net.setWeights( weights )
        device = "cuda" # TODO change to command line arg
        if device is not None:
            net.cuda( device )
        
        if data_usage[0]:
            for it in range( len(data)):
                rd_loader = data_loader.RealDataLoader( real_scan_path, data[it] , device )
                output = training.feedForward( net, rd_loader, 0, 1 )
                self.log.visualizeOutputStack( output[0], epoch_str +"output/", str(it) +"/real/" )
                self.log.saveOutputAsNPY( output[0], epoch_str +"output/" +str(it) +"/real/", resize=(128,256,256) )

        if data_usage[1]:
            for lt in range( len(data)):
                loader = data_loader.BatchLoader( input_path, data[lt], net.teacher_offset, device )
                output = training.feedForward( net, loader )
                #teacher = loader.getTeacherNp( 0, 4, loaders[lt.offset )
                for it in range( 4 ):
                    self.log.visualizeOutputStack( output[it], epoch_str +"output/", str(lt) +"/scan_" +str(it) +"/" )
                    self.log.saveOutputAsNPY( output[0], epoch_str +"output/" +str(lt) +"/scan_" +str(it) +"/", resize=(128,256,256) )
        
        if data_usage[2]:
            val_loader = data_loader.ValidationLoader( input_path, net.teacher_offset, net.ups, device )
            val = validation.ValidationResult()
            val.validate( net, val_loader )
            val.fillDictionary()
            self.log.saveValResults( val.results, val.val_dic, epoch_str )
            

class Session:
    
    def __init__( self, inp ):
        parser = arg_parser.SessionArgumentParser() 
        args = parser( inp )
        global debug_mode 
        debug_mode = args.debug[0]
        self.device = args.device[0]
        if args.mode == "feed":
            self.is_feed = True
            self.parser = arg_parser.FeedParser()
        elif args.mode == "train":
            self.is_feed = False
            session_log = "session_" +time.strftime( "%Y-%m-%d_%H%M%S/" )
            global logging_path 
            logging_path += session_log
            if not os.path.exists( logging_path ):
                os.makedirs( logging_path )
            self.parser = arg_parser.TrainParser()
        self.mode = 'add'
        self.net_parser = arg_parser.NetworkParser()
        self.instances = []
        self.ses_str = ""
            
    def __call__( self ):
        if not self.is_feed:
            self.train()
        elif self.is_feed:
            self.feed()
     
    def feed( self ):
        while( self.mode != 'run' ):
            new_cmd = input( "Next action: " )
            try:
                cmd = self.parser( new_cmd.split() )
                self.mode = cmd.mode
                if self.mode == 'add':
                    self.instances.append( FeedInstance( cmd.log[0], cmd.epoch[0] ) )
                if self.mode == 'show':
                    self.show()
                elif self.mode == 'run':
                    self.feedSession( cmd.data, ( cmd.real, cmd.synth, cmd.vali ) )
                elif self.mode == 'read':
                    log_list = arg_parser.readFromFile( cmd.input[0] )
                    for arg in log_list:
                        print(arg)
                        new_inst = self.parser( arg.split() )
                        self.instances.append( FeedInstance( new_inst.log[0], new_inst.epoch[0] ) )
            except SystemExit:
                pass
    
    def train( self ):
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
                elif self.mode == 'read':
                    net_list = arg_parser.readFromFile( cmd.input[0] )
                    for arg in net_list:
                        print(arg)
                        new_inst = self.parser( arg.split() )
                        self.addInstance( new_inst )
            except SystemExit:
                pass
          
   
    def addInstance( self, args ):
        """ Get line and test it in n-layer_conv_net parser """
        try:
            net = self.net_parser( args.net )
            new_inst = Instance( ( net.layers, args.net ), args.loss, args.optimizer, args.learning_rate, args.epochs, args.data, args.batch_size, args.slices, args.epoch_gates )
            self.instances.append( new_inst )
            inst_str = ' '.join( ['-l', args.loss[0], '-o', args.optimizer[0], '-lr', str( args.learning_rate[0] ), '-e', str( args.epochs[0] ), '-eg', str( args.epoch_gates ), '-d', str( args.data ), '-bs', str( args.batch_size[0] ), '-n', ' '.join( args.net ) ] )
            self.ses_str += inst_str +"\n"
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
        global logging_path
        ses_file = open( logging_path +"session.txt", "a" )
        ses_file.write( self.ses_str )
        ses_file.close()
        for instance in self.instances:
            curses.wrapper( instance.__call__ )

    def feedSession( self, data, data_usage ):
        print( "Starting Feast: " )
        for instance in self.instances:
            instance( data, data_usage )
  

def main( ):
    ses = Session( sys.argv[1:] )
    ses()    
    
if __name__ == '__main__':
    main()
