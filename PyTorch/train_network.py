# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:28:05 2018

@author: JHorn
"""

import numpy as np
import sys
sys.path.insert( 0, 'networks/' )

sys.path.insert( 1, 'utils/' )
import losses
import optimizer as opt
import logger
import data_loader
import acc_funcs as act
import evaluator as evl
import init_kernels as init

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
                  net, loss_func, optimizer, num_epochs, lr, arg_list=[] ):
    if is_cuda:
        net.cuda()
    
    log = logger.Logger( logging_path )
    log.masterLog( net.getStructure(), loss_func, optimizer, lr )
    
    opt = optimizer( net, lr )
    
    for epoch in range( 1, num_epochs+1 ):
        print( "Training epoch: " + str(epoch) )
        
        #Load Data
        bt_nbr = np.random.randint( num_bts )
        batch, teacher = loader.getBatch( bt_nbr, bt_size )
        
        tr_loss = 0.0
        tr_root_loss = 0.0
        tr_soil_loss = 0.0
        opt.zero_grad()
        for it in range( bt_size -eval_size ):
            input_data = batch[:,it,:,:,:].unsqueeze(1)
            teacher_data = teacher[:,it,:,:,:].unsqueeze(1)
            
            #Train
            output = net( input_data, loss_func.apply_sigmoid )
            loss, root_loss, soil_loss = loss_func( output, teacher_data, epoch )
            loss /=( bt_size -eval_size )
            loss.backward()
            
            tr_loss += loss
            tr_root_loss += root_loss
            tr_soil_loss += soil_loss
          
        tr_root_loss /= ( bt_size -eval_size )
        tr_soil_loss /= ( bt_size -eval_size )
        opt.step()
        #Eval
        output = net( batch[:,3,:,:,:].unsqueeze(1) )
        ev_loss, _, _ = loss_func( output, teacher[:,3,:,:,:].unsqueeze(1), epoch )
        
        #Log
        log.logEpoch( epoch, tr_loss.cpu().data.numpy(), ev_loss.cpu().data.numpy(), tr_root_loss.cpu().data.numpy(), tr_soil_loss.cpu().data.numpy() )
        if( epoch %20 == 0):
            weights = net.getWeightsCuda()
            output = feedForward( net, loader, 0 )
            if( epoch %100 == 0 ): 
                log.logMilestone( epoch, weights, output )
                #log epoch: weights, output, f_score in own folder
            else:
                log.logWeights( weights )
                teacher = loader.getTeacherNp( 0, 4, loader.offset )
                f1_r = np.array([0.0,0.0,0.0])
                f1_s = np.array([0.0,0.0,0.0])
                for it in range( 4 ):
                    f1_r += evle( output[it][0,0,:,:,:], teacher[0,it,:,:,:] )
                    f1_s += evle( output[it][0,0,:,:,:], teacher[0,it,:,:,:], True )
                log.logF1Root( epoch, f1_r /4 )
                log.logF1Soil( epoch, f1_s /4 )
                
                


if __name__ == '__main__':
    logging_path = "../../Data/logs/"
    input_path = "../../Data/real_scans/Artificial/Gauss+Perlin+Uniform/"
    teacher_path = "../../Data/real_scans/Artificial/Teacher/"
    num_bts = 30
    num_epochs = 2000
    lr = 0.003
    evle = evl.F1Score()

    feed_forward = sys.argv[1]
    if feed_forward == "True":
        real = sys.argv[2]
        network = __import__( "2-layer_conv_net" )
        log_path = logging_path + "2018-04-23_124841" +"/"
        log = logger.Log( log_path )
        epoch_str = "epoch_" +sys.argv[3] +"/"
        weights = log.getWeights( epoch_str )
        net = network.Network( [(5,5,5),(5,5,5)], 8, act.ReLU() )
        net.setWeights( weights )
        net.cuda()
        if real == "True":
            loader = data_loader.RealDataLoader( "../../Data/real_scans/Real MRI/Lupine_small/01_tiff_stack/", True )
            output = feedForward( net, loader, 0, 1 )
            log.visualizeOutputStack( output[0], epoch_str +"real_scan/" )
        else:
            loader = data_loader.BatchLoader( input_path, teacher_path, net.teacher_offset, num_bts, True )
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
                    
        network = __import__( "2-layer_conv_net" )
        #kernel_size_list = [[(3,3,3),(1,1,1)],[(5,5,5),(1,1,1)],[(3,3,3),(3,3,3)],[(5,5,5),(3,3,3)]]
        kernel_size_list = [[(3,3,3),(3,3,3)],
                            [(5,5,5),(3,3,3)],
                            [(7,7,7),(3,3,3)],
                            [(3,3,3),(5,5,5)],
                            [(5,5,5),(5,5,5)],
                            [(7,7,7),(5,5,5)]]
        act_list = [#act.Sigmoid(), 
                    act.ReLU()
                   ]
        loss_list = [#losses.CrossEntropyDynamic(1500),
                     losses.NegativeLogLikelihood(1500)
                    ]
        optimizer_list = [opt.AdamOptimizer()]
        for opti in optimizer_list:
            for lss in loss_list:
                for ks_size in kernel_size_list:
                    for acti in act_list: 
                        net = network.Network( ks_size, 8, acti )
                        w_init = init.kernel3.fillList()
                        log_path = logging_path + "2018-04-16_050603/"
                        log = logger.Log( log_path )
                        w_init = log.getWeights( "epoch_2500/" )
                        w_init[0][0] = w_init[0][0] /4
                        w_init[1][0] = w_init[1][0] /4
                        w_init[0][1] = w_init[0][1] /4
                        w_init[1][1] = w_init[1][1] /4
                        #net.setWeights( w_init )
                        loader = data_loader.BatchLoader( input_path, teacher_path, net.teacher_offset, num_bts, True )
                        trainNetwork( logging_path, loader, 4, 1, True, evle,
                                      net, lss, opti, num_epochs, lr )
