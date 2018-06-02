#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 05:19:33 2018

@author: root
"""
import evaluator
import numpy as np

def feedForward( net, loader, bt_nbr = 0, bt_size = 4 ):
    batch, _ = loader.getDefaultBatch( bt_nbr, bt_size )
    out_list = []
    for it in range( bt_size ):
        input_data = batch[:,it,:,:,:].unsqueeze(1)
        output = net( input_data, True )
        output = output.cpu().data.numpy()
        out_list.append( output )
    return out_list


def training( display, log, net, loader_list, loss_func, optimizer, lr, epochs, bt_size, num_slices ):
    evle = evaluator.F1Score()
    opt = optimizer( net, lr )
    
    for epoch in range( 1, epochs+1 ):
        tr_loss = 0.0
        tr_root_loss = 0.0
        tr_soil_loss = 0.0
        opt.zero_grad()
        
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
                    #batch, teacher = loader.getBatch( bt_size )
    
                display.addBatches( bt_size, num_slices )
                for it in range( bt_size ):
                    cut_it = int( round( ( batch.size()[4] -offset_dif*2) /num_slices ) )
                    cut_it_t = int( round( teacher.size()[4] /num_slices ) )
                    cut_id_t = 0
                    cut_id = offset_dif
                    for jt in range( num_slices ):
                        start_t, end_t = cut_id_t, min( teacher.size()[4], cut_it_t *(jt+1) )
                        start = cut_id
                        end =  min( batch.size()[4], cut_it +start)
                        cut_id_t = end_t
                        cut_id = end
                        input_data = batch[:,it,:,:,start-offset_dif:end+offset_dif].unsqueeze(1)
                        teacher_data = teacher[:,it,:,:,start_t:end_t].unsqueeze(1)
                        display.addComputed( it, jt, num_slices )
                        
                        #Train
                        output = net( input_data, loss_func.apply_sigmoid )
                        loss, root_loss, soil_loss = loss_func( output, teacher_data, epoch )
                        loss /=( bt_size ) *bt_per_it *num_slices
                        
                        tr_loss += loss
                        loss.backward()
    
                    tr_root_loss += root_loss
                    tr_soil_loss += soil_loss
                
        #Eval
        output = net( batch[:,bt_size -1,:,:,:].unsqueeze(1) )
        loss, _, _ = loss_func( output, teacher[:,bt_size -1,:,:,:].unsqueeze(1), epoch )
           
        tr_root_loss /= bt_size *bt_per_it *num_slices
        tr_soil_loss /= bt_size *bt_per_it *num_slices
        opt.step()

        display.endEpoch( tr_loss.cpu().data.numpy() )
        
        #Log
        log.logEpoch( epoch, tr_loss.cpu().data.numpy(), 0, tr_root_loss.cpu().data.numpy(), tr_soil_loss.cpu().data.numpy() )
        if( epoch %20 == 0):
            weights = net.getWeights()
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
                log.logMilestone( epoch, weights, output, tr_loss, f1_r, f1_s )
