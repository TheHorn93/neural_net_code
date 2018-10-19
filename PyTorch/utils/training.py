#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 05:19:33 2018

@author: root
"""
import evaluator
import split_data as sd
import numpy as np
import torch
import timeit
from random import shuffle

time = 0
def feedForward( net, loader, bt_nbr = 0, bt_size = 4, num_slices=[1,1,1] ):
    batch, teacher = loader.getDefaultBatch( bt_nbr, bt_size )
    batch.requires_grad=False
    out_list = []
    for it in range( bt_size ):
        #input_data = batch[:,it,:,:,:].unsqueeze(1)
        #split_list, tch_list, _ = sd.validationSplit( batch[:,it,:,:,:].unsqueeze(1), teacher[:,it,:,:,:].unsqueeze(1), num_slices, net.ups ) 
        split_list, tch_list, _ = sd.validationSplit( batch[:,it,:,:,:].unsqueeze(1), teacher[:,it,:,:,:].unsqueeze(1), num_slices, net.ups ) 
        del tch_list
        out_cpu_list = []
        num_sl = num_slices[0] *num_slices[1] *num_slices[2]
        for s_it in range(num_sl):
            c_input = split_list[s_it].cuda()
            output = net( c_input, True )
            del c_input
            cpu_output = output.cpu().data.numpy()
            out_cpu_list.append(cpu_output)
            #del split_list[s_it]
            #print(str(s_it) +"/" +str(len(split_list)))
            del output
        output = sd.reassemble( out_cpu_list, num_slices )
        out_list.append( output )
        #print( "DELETING" )
        del output
        #out_cpu_list.append( output.cpu() )
        #del output
        #del input_data
        #del split_list
        #out_list.append( sd.reassemble( out_cpu_list, num_slices ) )
    return out_list


def training( display, log, net, loader_list, loss_func, optimizer, lr, epochs, bt_size, num_slices ):
    evle = evaluator.F1Score()
    opt = optimizer( net, lr )
    opt.zero_grad()

    for epoch in range( 1, epochs+1 ):
        tr_loss = 0.0
        tr_root_loss = 0.0
        tr_soil_loss = 0.0
        #opt.zero_grad()
        
        display.newEpoch( epoch, epochs, 5 )
        for loader in loader_list:
            
            bt_per_it = 1
            for bt_it in range( bt_per_it ):
                noise_list = [x for x in range(5)]
                shuffle( noise_list )
                for noise_it in noise_list:
                #Load Data
                #bt_nbr = np.random.randint( num_bts )
                    if not net.ups:
                        batch, teacher = loader.getBatch( bt_size, noise_it )
                        offset_dif = int( ( batch.size()[4] -teacher.size()[4] ) /2 )
                    else:
                        batch, teacher = loader.getBatchAndUpsampledGT( bt_size, noise_it )
                        offset_dif = int( ( batch.size()[4] -teacher.size()[4] /2 ) /2 )
                        #display.addLine( str(offset_dif) )
                        #batch, teacher = loader.getBatch( bt_size )
        
                    num_sl = num_slices[0] *num_slices[1] *num_slices[2]
                    display.addBatches( bt_size, num_sl, noise_it )
                    bt_loss = 0
                    for it in range( bt_size ):
                        inp, tch = sd.splitInputAndTeacher( batch[:,it,:,:,:].unsqueeze(1), teacher[:,it,:,:,:].unsqueeze(1), num_slices, net.ups )
                        for jt in range( num_sl ):
                            input_data = inp[jt]
                            teacher_data = tch[jt]
                            
                            #Train
                            output = net( input_data, loss_func.apply_sigmoid )
                            loss, root_loss, soil_loss = loss_func( output, teacher_data, epoch )
                            loss /=( bt_size ) *bt_per_it *num_sl
                            display.addComputed( it, jt, num_sl )
                            
                            bt_loss += loss.cpu().data.numpy()
                            tr_loss += loss.cpu().data.numpy()
                            loss.backward()
        
                        tr_root_loss += root_loss.cpu().data.numpy()
                        tr_soil_loss += soil_loss.cpu().data.numpy()
    
                        del inp
                        del tch
                        del root_loss
                        del soil_loss
                        
                    opt.step()
                    opt.zero_grad()
                    del loss

                    display.endBatch( bt_loss )
                    del batch
                    del teacher
                    #torch.cuda.empty_cache()
        #Eval
        #output = net( batch[:,bt_size -1,:,:,:].unsqueeze(1) )
        #loss, _, _ = loss_func( output, teacher[:,bt_size -1,:,:,:].unsqueeze(1), epoch )
           
        tr_root_loss /= bt_size *bt_per_it *num_sl *5
        tr_soil_loss /= bt_size *bt_per_it *num_sl *5

        display.endEpoch( tr_loss /( 5 *len(loader_list)) )
        
        #Log
        cpu_loss = tr_loss /( 5 *len(loader_list) ) 
        log.logEpoch( epoch, cpu_loss, 0, tr_root_loss, tr_soil_loss )
        del tr_loss
        del tr_root_loss
        del tr_soil_loss
        if( epoch %5 == 0):
            net.train = False
            weights = net.getWeights()
            torch.cuda.empty_cache()
            output = feedForward( net, loader, 0, num_slices=num_slices )
            net.train = True
            log.logWeights( weights )
            teacher = loader.getTeacherNp( 0, 4, net.ups, loader.offset )
            f1_r = np.array([0.0,0.0,0.0])
            f1_s = np.array([0.0,0.0,0.0])
            for it in range( 4 ):
                f1_r += evle( output[it][0,0,:,:,:], teacher[0,it,:,:,:] )
                f1_s += evle( output[it][0,0,:,:,:], teacher[0,it,:,:,:], True )
            log.logF1Root( epoch, f1_r /4 )
            log.logF1Soil( epoch, f1_s /4 )
            if( epoch %25 == 0 ): 
                log.logMilestone( epoch, weights, output, cpu_loss, f1_r, f1_s )



def trainOnFullSet( display, log, net, loader, loss_func, optimizer, lr, epochs, bt_size, num_slices ):
    evle = evaluator.F1Score()
    opt = optimizer( net, lr )
    opt.zero_grad()
    #loader.createPool()
    display.initParams( epochs, bt_size, loader.set_size )
    its_per_ep = int( loader.set_size /bt_size )
    ep_it = 0

    for epoch in range( 1, epochs+1 ):
        loader.newEpoch()
        tr_loss = 0.0
        tr_root_loss = 0.0
        tr_soil_loss = 0.0
        #opt.zero_grad()
        
        display.newEpoch( epoch )

        for bt_it in range( its_per_ep ):
            display.startBatch()
            #num_sl = num_slices[0] *num_slices[1] *num_slices[2]
            bt_loss = 0.0
            bt_root_loss = 0.0
            bt_soil_loss = 0.0
            
            batch, teacher, key_list = loader.getNextInput( bt_it, bt_size )
            display.addBatch( key_list )
            for it in range( bt_size ):
                if batch[it].size()[2] > 200:
                    nm_slc = [3*num_slices[0],num_slices[1],num_slices[2]]
                else:
                    nm_slc = num_slices
                num_sl = nm_slc[0] *nm_slc[1] *nm_slc[2]
                inp, tch = sd.splitInputAndTeacher( batch[it], teacher[it], nm_slc, net.ups )
                for jt in range( num_sl ):
                    #display.addLine("THIS IS IT: " +str(jt) +" off " +str(len(inp))+"\n")
                    input_data = inp[jt]
                    teacher_data = tch[jt]
                    
                    #Train
                    output = net( input_data, loss_func.apply_sigmoid )
                    loss, root_loss, soil_loss = loss_func( output, teacher_data, epoch )
                    loss /=( bt_size ) *num_sl
                    display.computed( it, (jt+1) /num_sl )
                    
                    bt_loss += loss.cpu().data.numpy()
                    tr_loss += loss.cpu().data.numpy()
                    loss.backward()

                    bt_root_loss += root_loss.cpu().data.numpy() /num_sl 
                    bt_soil_loss += soil_loss.cpu().data.numpy() /num_sl

                print("Memory:: " +str(torch.cuda.memory_allocated()/1024/1024))
                opt.step()
                opt.zero_grad()

                del inp
                del tch
                del root_loss
                del soil_loss
                
            #opt.step()
            #opt.zero_grad()
            del loss

            del batch
            del teacher
           
            bt_root_loss /= bt_size
            bt_soil_loss /= bt_size
        
            #Log
            tr_root_loss += bt_root_loss
            tr_soil_loss += bt_soil_loss
            cpu_loss = bt_loss 
            log.logEpoch( ep_it, cpu_loss, 0, bt_root_loss, bt_soil_loss )
            ep_it += 1
            display.endBatch( cpu_loss )
        
        ep_loss = tr_loss /its_per_ep
        log.endEpoch( epoch, ep_loss, tr_root_loss /its_per_ep, tr_soil_loss /its_per_ep )
        del tr_loss
        del tr_root_loss
        del tr_soil_loss
        if( epoch %1 == 0):
            weights = net.getWeights()
            torch.cuda.empty_cache()
            output = feedForward( net, loader, 0, num_slices=num_slices )
            log.logWeights( weights )
            teacher = loader.getTeacherNp( 0, 4, net.ups, loader.offset )
            f1_r = np.array([0.0,0.0,0.0])
            f1_s = np.array([0.0,0.0,0.0])
            for it in range( 4 ):
                f1_r += evle( output[it][0,0,:,:,:], teacher[0,it,:,:,:] )
                f1_s += evle( output[it][0,0,:,:,:], teacher[0,it,:,:,:], True )
            log.logF1Root( epoch, f1_r /4 )
            log.logF1Soil( epoch, f1_s /4 )
            if( epoch %2 == 0 ): 
                log.logMilestone( epoch, weights, output, cpu_loss, f1_r, f1_s )
        
