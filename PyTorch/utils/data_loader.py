# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 04:08:51 2018

@author: JHorn
"""

import torch
from torch.autograd import Variable
import numpy as np

def getFilename( n ):
    return "FlippedData" + str(n) + ".npy"


class BatchLoader:
    
    def __init__( self, input_path, teacher_path, offset, num_bts, is_cuda = False ):
        self.input_path = input_path
        self.teacher_path = teacher_path
        self.offset = int(offset)
        self.num_bts = num_bts
        self.is_cuda = is_cuda
        
        
    def getBatch( self, bt_nbr, bt_size ):
        print( "Loading batch " +str(bt_nbr) )
        bt_path = self.input_path +'batch_' +str( bt_nbr ) +'/'
        
        data_list = []
        teacher_list = []
        for it in range( bt_size ):
            data_list.append( np.load( bt_path +getFilename( it ) ) )
            teacher_list.append( np.load( self.teacher_path +getFilename( it ) ) )
            
        shape = data_list[0].shape
        bt_data_size = ( 1, bt_size, shape[0], shape[1], shape[2] )
        teacher_data_size = ( 1, bt_size, shape[0] -int(self.offset*2), shape[1] -int(self.offset*2), shape[2] -int(self.offset*2) )

        batch = np.zeros( bt_data_size )
        teacher = np.zeros( teacher_data_size )

        for it in range( bt_size ):
            batch[0,it,:,:,:] = data_list[it]
            teacher[0,it,:,:,:] = teacher_list[it][self.offset:-self.offset,self.offset:-self.offset,self.offset:-self.offset]
        torch_batch = Variable( torch.Tensor(batch) )
        torch_teacher = Variable( torch.Tensor(teacher) )
        if self.is_cuda:
            torch_batch = torch_batch.cuda()
            torch_teacher = torch_teacher.cuda()
            
        return torch_batch, torch_teacher
    
    
    def getTeacherNp( self, bt_nbr, bt_size, offset = 0 ):
        print( "Loading teacher " +str(bt_nbr) )
        
        teacher_list = []
        for it in range( bt_size ):
            teacher_list.append( np.load( self.teacher_path +getFilename( it ) ) )
            
        shape = teacher_list[0].shape
        teacher_data_size = ( 1, bt_size, shape[0]-offset*2, shape[1]-offset*2, shape[2]-offset*2 )
        teacher = np.zeros( teacher_data_size )

        for it in range( bt_size ):
            if offset > 0:
                teacher[0,it,:,:,:] = teacher_list[it][offset:-offset,offset:-offset,offset:-offset]
            else:
                teacher[0,it,:,:,:] = teacher_list[it]
           
        return teacher