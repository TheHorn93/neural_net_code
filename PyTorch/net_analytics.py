# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 20:56:16 2018

@author: JHorn
"""

import sys
sys.path.insert( 0, 'networks/' )
sys.path.insert( 1, 'utils/' )
sys.path.insert( 2, 'analytics/' )

network = __import__( "2-layer_conv_net" )

import data_loader
import act_analytics
import logger
import acc_funcs as acc
import cv2

def analyseActivationMaps( net, log, load ):
    analys = act_analytics.ActivationAnalytics( net, log, load )
    analys.fillActivationMaps( "epoch_1200/" )
    #ims = analys.getDiffFromMap( analys.act_list[0][1][0][0,0,:,:,:], load.getTeacherNp(0,4)[0,0,:,:,:], 1 )
    analys.visualizeActivationMaps( "epoch_1200/act_map/" )
    
    
    
logging_path = "../../Data/logs/"
input_path = "../../Data/real_scans/Artificial/Gauss+Perlin+Uniform/"
teacher_path = "../../Data/real_scans/Artificial/Teacher/"

log_p = "2018-04-15_213814/"

kernels = [[3,3,3],[3,3,3]]
activation = acc.Sigmoid()

log = logger.Log( logging_path +log_p )
net = network.Network( kernels, 8, activation )
net.cuda()
loader = data_loader.BatchLoader( input_path, teacher_path, net.teacher_offset, 30, True )

analyseActivationMaps( net, log, loader )