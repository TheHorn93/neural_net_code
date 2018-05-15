# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:28:05 2018

@author: JHorn
"""

import numpy as np
import sys
sys.path.insert( 0, '/home/work/horn/code/neural_net_code/PyTorch/networks/' )

sys.path.insert( 1, '/home/work/horn/code/neural_net_code/PyTorch/utils/' )
import losses
import optimizer as opt
import logger
import bc4_data_loader as data_loader
import acc_funcs as act
import evaluator as evl
import init_kernels as init
import time
import torch

def feedForward( net, loader, bt_nbr = 0, bt_size = 4 ):
    batch, _ = loader.getBatch( bt_nbr, bt_size )
    out_list = []
    for it in range( bt_size ):
        input_data = batch[:,it,:,:,:].unsqueeze(1)
        output = net( input_data, True )
        output = output.cpu().data.numpy()
        out_list.append( output )
    return out_list
        

def trainNetwork( logging_path, loader, bt_size, eval_size, is_cuda, evle, 
                  net, loss_func, optimizer, num_epochs, str_epochs, lr, arg_list=[] ):
    if is_cuda is not None:
        net.cuda( is_cuda )
    
    print( "Training for " +str(num_epochs) )
    
    log = logger.Logger( logging_path )
    log.masterLog( net.getStructure(), loss_func, optimizer, lr )
    
    opt = optimizer( net, lr )
    
    for epoch in range( str_epochs, num_epochs+1 ):
        print( "Training epoch: " + str(epoch) )
        
        tr_loss = 0.0
        tr_root_loss = 0.0
        tr_soil_loss = 0.0
        ev_loss = 0.0
        opt.zero_grad()
        
        bt_per_it = 1
        for bt_it in range( bt_per_it ):
            #Load Data
            #bt_nbr = np.random.randint( num_bts )
            batch, teacher = loader.getBatchAndShuffle( bt_size )
            
            for it in range( bt_size -eval_size ):
                num_slices = 1
                cut_it = int( round( batch.size()[4] /num_slices ) )
                cut_id = 0
                for jt in range( num_slices ):
                    print( "   " +str(it) +" Slice: "  +str(jt) )
                    start, end = cut_id, min( batch.size()[4], cut_it *(jt+1) )
                    start_t, end_t = start, min( teacher.size()[4], end -net.teacher_offset*2 )
                    cut_id = end
                    input_data = batch[:,it,:,:,start:end].unsqueeze(1)
                    teacher_data = teacher[:,it,:,:,start_t:end_t].unsqueeze(1)
                    
                    #Train
                    output = net( input_data, loss_func.apply_sigmoid )
                    loss, root_loss, soil_loss = loss_func( output, teacher_data, epoch )
                    loss /=( bt_size -eval_size ) *bt_per_it *num_slices
                    loss.backward()
                    
                    tr_loss += loss
                    tr_root_loss += root_loss
                    tr_soil_loss += soil_loss
                
        #Eval
        output = net( batch[:,bt_size -1,:,:,:].unsqueeze(1) )
        loss, _, _ = loss_func( output, teacher[:,bt_size -1,:,:,:].unsqueeze(1), epoch )
            
        ev_loss += loss
           
        tr_root_loss /= ( bt_size -eval_size ) *bt_per_it *num_slices
        tr_soil_loss /= ( bt_size -eval_size ) *bt_per_it *num_slices
        opt.step()

        print( "Train Loss: " +str( tr_loss.cpu().data.numpy() ) )
        print( "Test Loss: " +str( ev_loss.cpu().data.numpy() ) )
        
        #Log
        log.logEpoch( epoch, tr_loss.cpu().data.numpy(), ev_loss.cpu().data.numpy(), tr_root_loss.cpu().data.numpy(), tr_soil_loss.cpu().data.numpy() )
        if( epoch %20 == 0):
            weights = net.getWeightsCuda()
            output = feedForward( net, loader, 0 )
            log.logWeights( weights )
            teacher = loader.getTeacherNp( 0, 4, loader.offset )
            f1_r = np.array([0.0,0.0,0.0])
            f1_s = np.array([0.0,0.0,0.0])
            for it in range( 4 ):
                f1_r += evle( output[it][0,0,:,:,:], teacher[0,it,:,:,:] )
                f1_s += evle( output[it][0,0,:,:,:], teacher[0,it,:,:,:], True )
            log.logF1Root( epoch, f1_r /4 )
            log.logF1Soil( epoch, f1_s /4 )
            if( epoch %100 == 0 ): 
                log.logMilestone( epoch, weights, output )


def trainCascade( logging_path, loader, bt_size, eval_size, is_cuda, evle, 
                  cascade, loss_func, optimizer, num_epochs, str_epochs, lr, arg_list=[] ):
    for it in range( cascade.max_iterator ):        
        net = cscd.initIteration()
        loader.offset = net.teacher_offset
        trainNetwork( logging_path, loader, bt_size, eval_size, is_cuda, evle, 
                      net, loss_func, optimizer, num_epochs, str_epochs, lr )
        cscd.getWeights( net )
                
def getArg( key, default = None ):
    for it in range( 1, len(sys.argv) ):
        parts = sys.argv[it].partition("=")
        key_str = parts[0]
        if key_str == key:
            return parts[2]
    return default

def parseSysArgs():
    args = {}
    ff = getArg("ff")
    if ff is not None and ff == "True": #feed forward
        args["ff"] = True
        rd = getArg("rd")
        if rd is not None and rd =="True":
            args["rd"] = True
        else:
            args["rd"] = False
        path=getArg( "path" ) #logging path
        args["path"]=path
        args["epoch"]=getArg( "epoch", "" )
        if args["epoch"] != "":
            args["epoch"] = "epoch_" +args["epoch"]
            
    else:
        args["ff"] = False
        init = getArg("init")
        if init is not None and init=="True": #init vals
            args["init"] = True
            csc = getArg( "csc" )
            if csc is not None and csc=="True":
                args["csc"] == "True"
            else:
                args["csc"] == "False"
                path=getArg( "path" ) #logging path
                args["path"]=path
                args["init_epoch"]=getArg("epoch") #epoch
                args["init_epoch_s"] = getArg( "epoch_s" ) #starting epoch
                args["init_epoch_e"] = getArg( "epoch_e" ) #ending epoch
        else:
            args["init"] = False
        args["lr"] = float( getArg( "lr", 0.0004 ) )
        args["epochs"] = int( getArg( "epochs", 1200 ) )
        args["w_gate_s"] = int( getArg( "w_gate_s", 500 ) )
        args["w_gate_e"] = int( getArg( "w_gate_e", 900 ) )
        args["per_batch"] = int( getArg( "per_batch", 10 ) )
        args["slices"] = int(getArg( "slices", 1 ) )
        args["device"] = getArg( "device", "cuda" )
    return args


if __name__ == '__main__':
    args = parseSysArgs()
    logging_path = "/home/work/horn/data/"
    input_path = "/home/work/uzman/Real_MRI/manual-reconstruction/"
    teacher_path = "../../Data/real_scans/Artificial/Teacher/"
    num_bts = 60
    epochs = [ args["epochs"],args["w_gate_s"],args["w_gate_e"] ]
    lr = args["lr"]
    evle = evl.F1Score()
    is_cuda = torch.device(args["device"])
    
    if args["ff"]:
        network = __import__( "3-layer_conv_net" )
        #log_path = logging_path + "2018-04-30_061116" +"/"
        log_path = logging_path + args["path"] +"/"
        log = logger.Log( log_path )
        epoch_str = args["epoch"] +"/"
        weights = log.getWeights( epoch_str )
        net = network.Network( [(3,3,3),(3,3,3),(3,3,3)], (16,8), (act.ReLU(),act.ReLU()) )
        net.setWeights( weights )
        net.cuda(is_cuda)
        if args["rd"]:
            loader = data_loader.RealDataLoader( "../../Data/real_scans/Real MRI/Lupine_small/01_tiff_stack/", is_cuda )
            output = feedForward( net, loader, 0, 1 )
            log.visualizeOutputStack( output[0], args["epoch"] )
            log.saveOutputAsNPY( output[0], epoch_str, resize=(256,256,128) )
            log.saveScatterPlot( output[0], epoch_str )
        else:
            loader = data_loader.BatchLoader( input_path, teacher_path, net.teacher_offset, num_bts, is_cuda )
            output = feedForward( net, loader )
            teacher = loader.getTeacherNp( 0, 4, loader.offset )
            for it in range( 4 ):
                f1, re, pre = evle( output[it][0,0,:,:,:], teacher[0,it,:,:,:] )
                print( "F1 root: " + str(f1) + " Recall: " +str(re) + " Precision: " +str(pre) )
                f1, re, pre = evle( output[it][0,0,:,:,:], teacher[0,it,:,:,:], True )
                print( "F1 soil: " + str(f1) )
                log.visualizeOutputStack( output[it], epoch_str +"output/", "scan_" +str(it) +"/" )
        
    else:
        #network = __import__( "1-layer_conv_net" )    
        #kernel_size_list = [(3,3,3),(5,5,5),(7,7,7)]
        #loss_list = [losses.CrossEntropyDynamic()]
        #optimizer_list = [opt.GradientDescent(), opt.AdamOptimizer()]
        #for it in range( len(optimizer_list) ):
        #    for jt in range( len(loss_list) ):
        #        for kt in range( len(kernel_size_list) ):
        #            net = network.Network( kernel_size_list[kt] )
        #            trainNetwork( logging_path, input_path, teacher_path, num_bts, True, 
        #                          net, loss_list[jt], optimizer_list[it], num_epochs, lr )
                    
        #network = __import__( "2-layer_conv_net" )
        #cascade = __import__( "cascade_correlation" )
        #kernel_size_list = [[(3,3,3),(1,1,1)],[(5,5,5),(1,1,1)],[(3,3,3),(3,3,3)],[(5,5,5),(3,3,3)]]
        kernel_size_list = [#[(3,3,3),(3,3,3)],
                            #[(5,5,5),(3,3,3)],
                            #[(7,7,7),(3,3,3)],
                            #[(9,9,9),(3,3,3)],
                            #[(3,3,3),(5,5,5)],
                            #[(5,5,5),(5,5,5)],
                            #[(7,7,7),(5,5,5)],
                            #[(9,9,9),(5,5,5)],
                            [(11,11,11),(5,5,5)]
                           ]
        network = __import__( "3-layer_conv_net" )
        kernel_size_list = [[(3,3,3),(3,3,3),(3,3,3)], 
                            [(5,5,5),(3,3,3),(3,3,3)],
                            [(5,5,5),(5,5,5),(3,3,3)],
                            [(5,5,5),(5,5,5),(5,5,5)],
                            [(7,7,7),(3,3,3),(3,3,3)],
                            [(7,7,7),(5,5,5),(3,3,3)],
                            [(7,7,7),(5,5,5),(5,5,5)]]
        num_kernels = (16,8)
        act_list = [#act.Sigmoid() 
                    (act.ReLU(),act.ReLU())
                   ]
        loss_list = [losses.CrossEntropyDynamic(epochs[0],epochs[1])
                     #,losses.NegativeLogLikelihood(1500)
                    ]
        optimizer_list = [opt.AdamOptimizer()]
        for opti in optimizer_list:
            for lss in loss_list:
                for ks_size in kernel_size_list:
                    for acti in act_list: 
                        net = network.Network( ks_size, num_kernels, acti )
                        #w_init = init.kernel( ks_size[0][0] )
                        str_epoch = 1
                        loader = data_loader.BatchLoader( input_path, teacher_path, net.teacher_offset, num_bts, is_cuda )
                        if args["init"]:
                            if not ["csc"]:
                                log_path = logging_path +args["path"] +"/"
                                log = logger.Log( log_path )
                                w_init = log.getWeights( args["init_epoch"] )
                                w_init[0][0] = w_init[0][0] /4
                                w_init[1][0] = w_init[1][0] /4
                                w_init[0][1] = w_init[0][1] /4
                                w_init[1][1] = w_init[1][1] /4
                                #net.setWeight( w_init.fill16(), 1, False )
                                net.setWeights( w_init )
                                str_epoch = args["init_epoch_s"]+1
                                epochs[2] = args["init_epoch_w"]
                                trainNetwork( logging_path, loader, 4, 1, is_cuda, evle,
                                              net, lss, opti, epochs[2], str_epoch, lr )
                            else:
                                stages = [([(3,3,3),(3,3,3)],8,act.ReLU()),
                                          ([(3,3,3),(3,3,3)],16,act.ReLU()),
                                          ([(5,5,5),(3,3,3)],16,act.ReLU()),
                                          ([(7,7,7),(3,3,3)],16,act.ReLU()),
                                          ([(9,9,9),(3,3,3)],16,act.ReLU()),
                                          ([(9,9,9),(5,5,5)],16,act.ReLU()),
                                          ([(11,11,11),(5,5,5)],16,act.ReLU()),
                                          ([(11,11,11),(5,5,5)],16,act.Sigmoid())]
                                cscd = cascade.Network( stages )
                                trainCascade( logging_path, loader, args["per_batch"], 1, is_cuda, evle,
                                              cscd, lss, opti, epochs[2], str_epoch, lr )
                        else:
                            trainNetwork( logging_path, loader, args["per_batch"], 1, is_cuda, evle,
                                              net, lss, opti, epochs[2], str_epoch, lr )
