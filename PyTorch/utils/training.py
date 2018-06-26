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

def feedForward( net, loader, bt_nbr = 0, bt_size = 4 ):
    batch, _ = loader.getDefaultBatch( bt_nbr, bt_size )
    batch.requires_grad=False
    out_list = []
    for it in range( bt_size ):
        #print(it)
        input_data = batch[:,it,:,:,:].unsqueeze(1)
        output = net( input_data, True )
        output_cp = output.cpu().data.numpy()
        del output
        out_list.append( output_cp )
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
                #Load Data
                #bt_nbr = np.random.randint( num_bts )
                if not net.ups:
                    batch, teacher = loader.getBatch( bt_size )
                    offset_dif = int( ( batch.size()[4] -teacher.size()[4] ) /2 )
                else:
                    batch, teacher = loader.getBatchAndUpsampledGT( bt_size )
                    offset_dif = int( ( batch.size()[4] -teacher.size()[4] /2 ) /2 )
                    display.addLine( str(offset_dif) )
                    #batch, teacher = loader.getBatch( bt_size )
    
                num_sl = num_slices[0] *num_slices[1] *num_slices[2]
                display.addBatches( bt_size, num_sl )
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
                
                del batch
                del teacher
        #Eval
        #output = net( batch[:,bt_size -1,:,:,:].unsqueeze(1) )
        #loss, _, _ = loss_func( output, teacher[:,bt_size -1,:,:,:].unsqueeze(1), epoch )
           
        tr_root_loss /= bt_size *bt_per_it *num_sl
        tr_soil_loss /= bt_size *bt_per_it *num_sl
        opt.step()
        opt.zero_grad()

        display.endEpoch( tr_loss.cpu().data.numpy() )
        
        #Log
        cpu_loss = tr_loss.cpu().data.numpy()
        log.logEpoch( epoch, cpu_loss, 0, tr_root_loss.cpu().data.numpy(), tr_soil_loss.cpu().data.numpy() )
        del tr_loss
        del tr_root_loss
        del tr_soil_loss
        if( epoch %20 == 0):
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
            if( epoch %100 == 0 ): 
                log.logMilestone( epoch, weights, output, cpu_loss, f1_r, f1_s )
