# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:52:03 2018

@author: JHorn
"""

import os
import time
import visdom
import numpy as np
import visualizer

class Logger:
    
    def __init__( self, path ):
        self.folder = time.strftime( "%Y-%m-%d_%H%M%S" )
        self.folder_path = path + self.folder +"/"
        if not os.path.exists( self.folder_path ):
            os.makedirs( self.folder_path )
        print( "Logging to: " + self.folder_path )
            
        self.vis = visdom.Visdom( env=self.folder )
        self.train_loss = self.vis.line(np.array([0]), win="train_loss", opts=dict(title="Train Loss"))
        self.eval_loss = self.vis.line(np.array([0]), win="eval_loss", opts=dict(title="Eval Loss"))
        self.eval_loss = self.vis.line(np.array([0]), win="root_loss", opts=dict(title="Root Loss"))
        self.eval_loss = self.vis.line(np.array([0]), win="soil_loss", opts=dict(title="Soil Loss"))
        self.f1_score_r = self.vis.line(np.array([0]), win="f1_score_r", opts=dict(title="F1-Score Root"))
        self.re_score_r = self.vis.line(np.array([0]), win="re_score_r", opts=dict(title="Recall Root"))
        self.pre_score_r = self.vis.line(np.array([0]), win="pre_score_r", opts=dict(title="Precision Root"))
        self.f1_score_s = self.vis.line(np.array([0]), win="f1_score_s", opts=dict(title="F1-Score Soil"))
        self.re_score_s = self.vis.line(np.array([0]), win="re_score_s", opts=dict(title="Recall Soil"))
        self.pre_score_s = self.vis.line(np.array([0]), win="pre_score_s", opts=dict(title="Precision Soil"))
        print( "Visdom instantiated" )
        
        
    def masterLog( self, structure, loss, optimizer, lr ):
        output = structure +"\n Loss: " + str( loss ) +"\n Optimizer: " +str( optimizer ) +"\n Learning rate: " +str(lr)
        self.vis.text( output, win="structure" )
        log_file = open( self.folder_path + "model.txt", "a" )
        log_file.write( output )
        log_file.close()
        
        
    def logMilestone( self, epoch, weight_list, output ):
        folder_path = "epoch_" +str(epoch) +"/"
        os.mkdir( self.folder_path +folder_path )
        self.logWeights( weight_list, folder_path )
        fig = visualizer.multiOutput4( output )
        fig.savefig( self.folder_path +folder_path + "output.png" )
        image = visualizer.getRawImage( fig, True )
        self.vis.image( image, win="output", opts=dict(title="Output") )
    
    def logEpoch( self, epoch, train_err, eval_err, root_loss, soil_loss ):
        ep = np.array([epoch])
        self.vis.line( train_err, ep, win="train_loss", update="append" )
        self.vis.line( eval_err, ep, win="eval_loss", update="append" )
        self.vis.line( root_loss, ep, win="root_loss", update="append" )
        self.vis.line( soil_loss, ep, win="soil_loss", update="append" )
        file = open( self.folder_path + "loss.txt", "a" )
        file.write( str( epoch ) + ": " + str(train_err) +"\n" )
        file.close()
        
    def logF1Root( self, epoch, f1_t ):
        ep = np.array([epoch])
        self.f1_score_r = self.vis.line(np.array([f1_t[0]]), ep, win="f1_score_r", update="append" )
        self.re_score_r = self.vis.line(np.array([f1_t[1]]), ep, win="re_score_r", update="append" )
        self.pre_score_r = self.vis.line(np.array([f1_t[2]]), ep, win="pre_score_r", update="append" )
        
    def logF1Soil( self, epoch, f1_s ):
        ep = np.array([epoch])
        self.f1_score_r = self.vis.line(np.array([f1_s[0]]), ep, win="f1_score_s", update="append" )
        self.re_score_r = self.vis.line(np.array([f1_s[1]]), ep, win="re_score_s", update="append" )
        self.pre_score_r = self.vis.line(np.array([f1_s[2]]), ep, win="pre_score_s", update="append" )
        
        
    def logWeights( self, weight_list, path="" ):
        self.saveWeights( weight_list, path )
        self.visualizeWeights( weight_list, path )
        
    def saveWeights( self, weight_list, path="" ):
        log_path = self.folder_path +path
        if log_path[len( log_path )-1] != '/':
            log_path += '/'
            
        for it in range( len( weight_list ) ):
            np.save( log_path +"weights_" +str(it), weight_list[it][0] )
            np.save( log_path +"bias_" +str(it), weight_list[it][1] )
            
    def visualizeWeights( self, weight_list, path="" ):
        log_path = self.folder_path +path
        figs = visualizer.visualizeWeights( weight_list )
        for it in range( len(figs) ):
            figs[it].savefig( log_path +"vis_weights_" +str(it) +".png" )
            image = visualizer.getRawImage( figs[it], True )
            self.vis.image( image, win="weights_"+str(it), opts=dict(title="Layer " +str(it)) )
            figs[it].clear()
                
           

class Log:
    def __init__( self, log_path ):
        self.log_path = log_path
        
    def getWeights( self, path="" ):
        print( "Loading weights from: " +self.log_path + path )
        w_path = self.log_path + path
        it = 0
        weight_list = []
        while True:
            w_file = w_path + "weights_" +str(it) +".npy"
            b_file = w_path + "bias_" +str(it) +".npy"
            if not os.path.isfile( w_file ) or not os.path.isfile( b_file ):
                break
            weight = np.load( w_file )
            bias = np.load( b_file )
            weight_list.append( [weight, bias] )
            it += 1
        print( "Got " + str(len(weight_list)) + " set of weights" )
        return weight_list
              
    def visualizeOutputStack( self, input_data, path="", output_folder = "output/" ):
        out_path = self.log_path + path
        folder = out_path +output_folder
        print( "Writing output stack to: " +folder )
        if not os.path.exists( folder ):
            os.makedirs( folder )
        visualizer.cvMultiSlice( input_data[0,0,:,:,:], folder, 2, True )
      
    def saveImageStack( self, stack, path="" ):
        folder = self.log_path +path 
        if not os.path.exists( folder ):
            os.makedirs( folder )
        visualizer.cvSaveStack( stack, folder )
        
    def saveOutputAsNPY( self, stack, path="", resize=None ):
        folder = self.log_path + path
        print( "Saving output to: " +folder )
        if not os.path.exists( folder ):
            os.makedirs( folder )
        if resize is not None:
            x_s = int(( resize[0] -stack.shape[2] ) /2)
            y_s = int(( resize[1] -stack.shape[3] ) /2)
            z_s = int(( resize[2] -stack.shape[4] ) /2)
            out = np.zeros( (resize) )
            out[x_s:-x_s,y_s:-y_s,z_s:-z_s] = stack[0,0,:,:,:]
        else:
            out = stack
        np.save( folder +"output.npy", out )
        
    def saveScatterPlot( self, stack, path="" ):
        folder = self.log_path + path
        print( "Saving scatter to: " +folder )
        if not os.path.exists( folder ):
            os.makedirs( folder )
        visualizer.scatterRoot( stack, folder +"scatter.html" )
     
#a = np.random.rand( 1,1,3,3,3 )
#a = a.astype(np.float32)
#a = a-0.5
#print(a)
#b = flattenWeights(a)
#print(b)
#log = Logger("")
#log.logMilestone( 100, [(a,[0.05856])] )