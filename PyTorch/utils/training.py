#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 05:19:33 2018

@author: root
"""

def training( stdscr, log, net, loader_list, loss, opt, lr, epochs, bt_size, num_slices ):
    print( "Starting Training for " +str( epochs ) )
    optimizer = opt( net, lr )
    for epoch in range( epochs ):
        for loader in loader_list:
            tr_loss = 0.0
            tr_root_loss = 0.0
            tr_soil_loss = 0.0
            ev_loss = 0.0
            opt.zero_grad()
            
            bt_per_it = 1
            for bt_it in range( bt_per_it ):
                #Load Data
                #bt_nbr = np.random.randint( num_bts )
                if not ups:
                    batch, teacher = loader.getBatch( bt_size )
                else:
                    batch, teacher = loader.getBatchAndUpsampledGT( bt_size )
                
                for it in range( bt_size ):
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
                    log.logMilestone( epoch, weights, output, tr_loss, f1_r, f1_s )