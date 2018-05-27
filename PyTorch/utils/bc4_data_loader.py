# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:33:53 2018

@author: JHorn
"""

import torch
from torch.autograd import Variable
import numpy as np

def getFilename( n, c ):
    return "noisy_" +str(n) +"_" +str(c) +"_.npy"


class BatchLoader:
    
    def __init__( self, input_path, teacher_path, offset, num_bts, is_cuda = -1 ):
        self.input_path = input_path
        #self.teacher_path = teacher_path
        self.offset = int(offset)
        self.num_bts = num_bts
        self.is_cuda = is_cuda
        self.r_fac_dic = ["r_factor_0.34/","r_factor_0.71/","r_factor_1.00/","r_factor_1.41/"]
        self.rot_dic = ["rot_0/","rot_60/","rot_120/"]
        self.x_flip_dic = ["x_flip_0/","x_flip_1/"]
        self.y_flip_dic = ["y_flip_0/","y_flip_1/"]
        self.swap_dic = ["x_y_swap_0/","x_y_swap_1/"]
        self.c_dic = ["g","h","l"]
        
        
    def getDefaultBatch( self, bt_nbr, bt_size ):
        r_fac_rnd = np.array( [0,1,2,3] )
        rot_rnd = np.array( [0,1,0,2] )
        x_flip_rnd = np.array( [0,0,0,1] )
        y_flip_rnd = np.array( [0,1,0,0] )
        swap_rnd = np.array( [1,1,0,0] )
        sc_id = np.array( [0,0,0,0] )
        c_id = np.array( [0,1,0,2] )
        print( "Loading from batches: " )
        
        data_list = []
        teacher_list = []
        for it in range( bt_size ):
            file_str = ( self.input_path +self.r_fac_dic[r_fac_rnd[it]]
                         +self.rot_dic[rot_rnd[it]]
                         +self.x_flip_dic[x_flip_rnd[it]]
                         +self.y_flip_dic[y_flip_rnd[it]]
                         +self.swap_dic[swap_rnd[it]]
                         +"256x256x128/"
                        )
            print( str(it) +": " + file_str )
            data = np.load( file_str +getFilename( sc_id[it], self.c_dic[c_id[it]] ) )[:,0,:,:]
            teacher = np.load( file_str +"ground_truth.npy" )[:,0,:,:]
            data_list.append( data.astype(np.float32) /255 )
            teacher_list.append( teacher.astype(np.float32) /255 )
            
        shape = data_list[0].shape
        bt_data_size = ( 1, bt_size, shape[0], shape[1], shape[2] )
        teacher_data_size = ( 1, bt_size, shape[0] -int(self.offset*2), shape[1] -int(self.offset*2), shape[2] -int(self.offset*2) )

        batch = np.empty( bt_data_size )
        teacher = np.empty( teacher_data_size )
        
        for it in range( bt_size ):
            batch[0,it,:,:,:] = data_list[it]
            teacher[0,it,:,:,:] = teacher_list[it][self.offset:-self.offset,self.offset:-self.offset,self.offset:-self.offset]
        torch_batch = Variable( torch.Tensor(batch) )
        torch_teacher = Variable( torch.Tensor(teacher) )
        if self.is_cuda is not None:
            torch_batch = torch_batch.cuda(self.is_cuda)
            torch_teacher = torch_teacher.cuda(self.is_cuda)
            
        return torch_batch, torch_teacher
    
    
    def getBatch( self, bt_size ):
        r_fac_rnd = np.zeros( [bt_size], dtype=np.int32 )
        rot_rnd = np.random.randint( 3, size=bt_size )
        x_flip_rnd = np.random.randint( 2, size=bt_size )
        y_flip_rnd = np.random.randint( 2, size=bt_size )
        swap_rnd = np.random.randint( 2, size=bt_size )
        sc_id = np.random.randint( 5, size=bt_size )
        c_id = np.zeros( [bt_size], dtype=np.int32 )
        print( "Loading from batches: " )
        
        r_ct = 0
        c_ct = 0
        for it in range( bt_size ):
          r_fac_rnd[it] = r_ct
          r_ct = r_ct+1
          c_id[it] = c_ct
          if r_ct > 3:
            r_ct = 0
            c_ct = c_ct +1

        data_list = []
        teacher_list = []
        for it in range( bt_size ):
            file_str = ( self.input_path +self.r_fac_dic[r_fac_rnd[it]]
                         +self.rot_dic[rot_rnd[it]]
                         +self.x_flip_dic[x_flip_rnd[it]]
                         +self.y_flip_dic[y_flip_rnd[it]]
                         +self.swap_dic[swap_rnd[it]]
                         +"256x256x128/"
                        )
            print( str(it) +": " + file_str )
            data = np.load( file_str +getFilename( sc_id[it], self.c_dic[c_id[it]] ) )[:,0,:,:]
            teacher = np.load( file_str +"ground_truth.npy" )[:,0,:,:]
            data_list.append( data.astype(np.float32) /255 )
            teacher_list.append( teacher.astype(np.float32) /255 )
            
        shape = data_list[0].shape
        bt_data_size = ( 1, bt_size, shape[0], shape[1], shape[2] )
        teacher_data_size = ( 1, bt_size, shape[0] -int(self.offset*2), shape[1] -int(self.offset*2), shape[2] -int(self.offset*2) )

        batch = np.empty( bt_data_size )
        teacher = np.empty( teacher_data_size )
        
        for it in range( bt_size ):
            batch[0,it,:,:,:] = data_list[it]
            teacher[0,it,:,:,:] = teacher_list[it][self.offset:-self.offset,self.offset:-self.offset,self.offset:-self.offset]
        torch_batch = Variable( torch.Tensor(batch) )
        torch_teacher = Variable( torch.Tensor(teacher) )
        if self.is_cuda is not None:
            torch_batch = torch_batch.cuda(self.is_cuda)
            torch_teacher = torch_teacher.cuda(self.is_cuda)
            
        return torch_batch, torch_teacher


    def getBatchAndShuffle( self, bt_size ):
        r_fac_rnd = np.random.randint( 4, size=bt_size )
        rot_rnd = np.random.randint( 3, size=bt_size )
        x_flip_rnd = np.random.randint( 2, size=bt_size )
        y_flip_rnd = np.random.randint( 2, size=bt_size )
        swap_rnd = np.random.randint( 2, size=bt_size )
        sc_id = np.random.randint( 5, size=bt_size )
        c_id = np.random.randint( 3, size=bt_size )
        print( "Loading from batches: " )
        
        data_list = []
        teacher_list = []
        for it in range( bt_size ):
            file_str = ( self.input_path +self.r_fac_dic[r_fac_rnd[it]]
                         +self.rot_dic[rot_rnd[it]]
                         +self.x_flip_dic[x_flip_rnd[it]]
                         +self.y_flip_dic[y_flip_rnd[it]]
                         +self.swap_dic[swap_rnd[it]]
                         +"256x256x128/"
                        )
            print( str(it) +": " + file_str )
            data = np.load( file_str +getFilename( sc_id[it], self.c_dic[c_id[it]] ) )[:,0,:,:]
            teacher = np.load( file_str +"ground_truth.npy" )[:,0,:,:]
            data_list.append( data.astype( np.float32 ) /255 )
            teacher_list.append( teacher.astype( np.float32 ) /255 )
            
        shape = data_list[0].shape
        bt_data_size = ( 1, bt_size, shape[0], shape[1], shape[2] )
        teacher_data_size = ( 1, bt_size, shape[0] -int(self.offset*2), shape[1] -int(self.offset*2), shape[2] -int(self.offset*2) )

        batch = np.empty( bt_data_size )
        teacher = np.empty( teacher_data_size )
        
        for it in range( bt_size ):
            batch[0,it,:,:,:] = data_list[it]
            teacher[0,it,:,:,:] = teacher_list[it][self.offset:-self.offset,self.offset:-self.offset,self.offset:-self.offset]
        torch_batch = Variable( torch.Tensor(batch) )
        torch_teacher = Variable( torch.Tensor(teacher) )
        if self.is_cuda is not None:
            torch_batch = torch_batch.cuda(self.is_cuda)
            torch_teacher = torch_teacher.cuda(self.is_cuda)
            
        return torch_batch, torch_teacher
            
    
    
    def getTeacherNp( self, bt_nbr, bt_size, offset = 0 ):
        print( "Loading teacher " +str(bt_nbr) )
        
        r_fac_rnd = np.array( [0,1,2,3] )
        rot_rnd = np.array( [0,1,0,2] )
        x_flip_rnd = np.array( [0,0,0,1] )
        y_flip_rnd = np.array( [0,1,0,0] )
        swap_rnd = np.array( [1,1,0,0] )
        sc_id = np.array( [0,0,0,0] )
        c_id = np.array( [0,1,0,2] )
        
        teacher_list = []
        for it in range( bt_size ):
            file_str = ( self.input_path +self.r_fac_dic[r_fac_rnd[it]]
                         +self.rot_dic[rot_rnd[it]]
                         +self.x_flip_dic[x_flip_rnd[it]]
                         +self.y_flip_dic[y_flip_rnd[it]]
                         +self.swap_dic[swap_rnd[it]]
                         +"256x256x128/"
                        )
            print( str(it) +": " + file_str )
            teacher = np.load( file_str +"ground_truth.npy" )[:,0,:,:]
            teacher_list.append( teacher.astype(np.float32) /255 )
            
        shape = teacher_list[0].shape
        teacher_data_size = ( 1, bt_size, shape[0]-offset*2, shape[1]-offset*2, shape[2]-offset*2 )
        teacher = np.zeros( teacher_data_size )

        for it in range( bt_size ):
            if offset > 0:
                teacher[0,it,:,:,:] = teacher_list[it][offset:-offset,offset:-offset,offset:-offset]
            else:
                teacher[0,it,:,:,:] = teacher_list[it]
           
        return teacher
    
  
import cv2
class RealDataLoader:
    
    def __init__( self, input_path, is_cuda = -1 ):
        self.path = input_path
        self.is_cuda = is_cuda
    
    def getBatch( self, bt_nbr=0, bt_size=0 ):
        print( "Loading from: " + self.path )
        output = np.zeros( [256,256,128], np.ubyte )
        for it in range( 128 ):
            im_name = "lupi8_test" + "{0:0>4}".format(it) + ".tif"
            print( self.path +im_name )
            im = cv2.imread( self.path + im_name )
            im = cv2.cvtColor( im, cv2.COLOR_RGB2GRAY )
            output[:,:,it] = im
        output = output.reshape( [1,1,256,256,128] )
        output = output.astype( np.float64 )
        output /= 255
        print( str( np.amin(output) ) + "<" + str( np.amax(output) ) )
        output = Variable( torch.Tensor( output ) )
        if self.is_cuda is not None:
            output = output.cuda(self.is_cuda)
        return output, Variable( torch.Tensor([]) )
    
    def getRealScan( self ):
        print( "Loading from: " + self.path )
        output = np.zeros( [256,256,128], np.ubyte )
        for it in range( 128 ):
            im_name = "lupi8_test" + "{0:0>4}".format(it) + ".tif"
            print( self.path +im_name )
            im = cv2.imread( self.path + im_name )
            im = cv2.cvtColor( im, cv2.COLOR_RGB2GRAY )
            output[:,:,it] = im
        output = output.reshape( [1,1,256,256,128] )
        output = output.astype( np.float64 )
        output /= 255
        return output
 
