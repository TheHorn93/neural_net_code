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
from random import shuffle

def feedForward( net, loader, bt_nbr = 0, bt_size = 4, num_slices=[1,1,1] ):
    batch, teacher = loader.getDefaultBatch( bt_nbr, bt_size )
    batch.requires_grad=False
    out_list = []
    for it in range( bt_size ):
        split_list, _ = sd.splitInputAndTeacher( batch[:,it,:,:,:].unsqueeze(1), teacher[:,it,:,:,:].unsqueeze(1), num_slices, net.ups ) 
        out_cpu_list = []
        for input_data in split_list:
        #print(it)
            output = net( input_data, True )
            out_cpu_list( output.cpu() )
            del output
        del split_list
        out_list.append( sd.reassmble( out_cpu_list, num_slices ).data.numpy() )
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
        
        display.newEpoch( epoch, epochs )
        for loader in loader_list:
            
            bt_per_it = 1
            for bt_it in range( bt_per_it ):
                noise_list = [x for x in range(5)]
                noise_list.shuffle()
                for noise_it in noise_list:
                #Load Data
                #bt_nbr = np.random.randint( num_bts )
                    if not net.ups:
                        batch, teacher = loader.getBatch( bt_size, noise_it )
                        offset_dif = int( ( batch.size()[4] -teacher.size()[4] ) /2 )
                    else:
                        batch, teacher = loader.getBatchAndUpsampledGT( bt_size, noise_it )
                        offset_dif = int( ( batch.size()[4] -teacher.size()[4] /2 ) /2 )
                        display.addLine( str(offset_dif) )
                        #batch, teacher = loader.getBatch( bt_size )
        
                    num_sl = num_slices[0] *num_slices[1] *num_slices[2]
                    display.addBatches( bt_size, num_sl, noise_it )
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
                            
                            tr_loss += loss
                            loss.backward()
        
                        tr_root_loss += root_loss
                        tr_soil_loss += soil_loss
    
                        del inp
                        del tch
                        
                    opt.step()
                    opt.zero_grad()
                    display.endBatch()
                    del batch
                    del teacher
        #Eval
        #output = net( batch[:,bt_size -1,:,:,:].unsqueeze(1) )
        #loss, _, _ = loss_func( output, teacher[:,bt_size -1,:,:,:].unsqueeze(1), epoch )
           
        tr_root_loss /= bt_size *bt_per_it *num_sl *5
        tr_soil_loss /= bt_size *bt_per_it *num_sl *5

        display.endEpoch( tr_loss.cpu().data.numpy() )
        
        #Log
        cpu_loss = tr_loss.cpu().data.numpy() /5
        log.logEpoch( epoch, cpu_loss, 0, tr_root_loss.cpu().data.numpy(), tr_soil_loss.cpu().data.numpy() )
        del tr_loss
        del tr_root_loss
        del tr_soil_loss
        if( epoch %1 == 0):
            weights = net.getWeights()
            torch.cuda.empty_cache()
            output = feedForward( net, loader, 0 )
            log.logWeights( weights )
            teacher = loader.getTeacherNp( 0, 4, net.ups, loader.offset )
            f1_r = np.array([0.0,0.0,0.0])
            f1_s = np.array([0.0,0.0,0.0])
            for it in range( 4 ):
                f1_r += evle( output[it][0,0,:,:,:], teacher[0,it,:,:,:] )
                f1_s += evle( output[it][0,0,:,:,:], teacher[0,it,:,:,:], True )
            log.logF1Root( epoch, f1_r /4 )
            log.logF1Soil( epoch, f1_s /4 )
            if( epoch %1 == 0 ): 
                log.logMilestone( epoch, weights, output, cpu_loss, f1_r, f1_s )
