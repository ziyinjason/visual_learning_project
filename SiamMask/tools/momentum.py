# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:52:33 2019

@author: Davy
"""

from scipy import spatial
import numpy as np
import pdb

def sigmoid(x):
  return 1/(1+np.exp(-x))

def lr_generate(history_speed,current_speed,history_size,current_size,a=[4,2,8,-1,-5,2]):
 
    # size score
    W_pre,H_pre=history_size[0],history_size[1]
    W_cur,H_cur=current_size[0],current_size[1]
    
#    H_pre,W_pre=300,500
#    H_cur,W_cur=450,750
    ratio_score=((H_cur+0.1)/(W_cur+0.1))/((H_pre+0.1)/(W_pre+0.1))
    size_abs_score=abs((H_cur-H_pre)/H_pre)+abs((W_cur-W_pre)/W_pre)
    
    a1,a2=4,2
    a3=-8
    size_score=(1-abs(ratio_score-1)*a1)*a2+(size_abs_score*(a3)+2)
    lr_size=1-sigmoid(size_score)
    
    # velocity score
#    history_speed=np.array([-1,20])
#    current_speed=np.array([-3,19]) 
    direction_distance = spatial.distance.cosine(history_speed,current_speed)
    magnitude_distance = np.linalg.norm(history_speed - current_speed)


    
    a4=-1              #   to 2
    a5,a6=-5,2         # a5+a6 to a6
    vel_score=(magnitude_distance*(a4)+2)+ (direction_distance*a5+a6)
    lr_v=1-sigmoid(vel_score)

            
    return lr_v,lr_size

def momentum(history_speed, current_speed, history_size,current_size):
#             direction_threshold, magnitude_threshold, lr):
    '''
        Update the object speed using momentum
        Input:
            history_speed: np.array(x,y)
            current_speed: np.array(x,y)
        Output:
            new history speed: np.array(x,y)
    '''
#    direction_distance = scipy.spatial.distance.cosine(history_speed,
#                                                       current_speed)
#    magnitude_distance = np.linalg.norm(history_speed - current_speed)
#    if direction_distance > direction_threshold or magnitude_distance > magnitude_threshold:
#        current_speed = history_speed
#    else:
#        current_speed = history_speed * lr + current_speed * (1 - lr)
    if not all(history_speed==0):
        lr_v,lr_size=lr_generate(history_speed,current_speed,history_size,current_size)
        current_speed = history_speed * lr_v + current_speed * (1 - lr_v)
        current_size = history_size * lr_v + current_size * (1 - lr_v)
    return current_speed,current_size