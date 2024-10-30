#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:53:49 2024

@author: emi
"""
# ----------- import packages --------------
import numpy as np
from stl import mesh
import pandas as pd
import os, shutil
from pathlib import Path
import sys
import logging
import scipy.io as sio
import opensim
import time


from Public_functions import *
from Private_STAPLEtools import *


#%% ----------
#  SETTINGS 
# ----------

# Set to Executables directory if running in Python directly
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# os.chdir('..')
path_file = os.getcwd()

# set output folder
output_models_folder = 'opensim_models_examples/Python'

output_model_file_name = 'Carman_auto2020_HipKnee_R_ScaledFoot.osim'

# body sides
curr_side = 'r'

# generic model to use as baseline model
Generic_osimModel_file = os.path.join(path_file, output_models_folder, 'GAIT2392_SCALED.osim')
# STAPLE generated model that we want to merge with the generic baseline
Specific_osimModel_file = os.path.join(path_file, output_models_folder, 'Carman_auto2020_HipKnee_R.osim')


# create model folder if required
dst = Path(output_models_folder)
dst.mkdir(parents=True, exist_ok=True)
dst = str(dst)

# log printout
# log_folder = Path(output_models_folder)
# logging.basicConfig(filename = str(log_folder) + '/ADDING_FOOT_' + curr_model_name + '.log', filemode = 'w', format = '%(levelname)s:%(message)s', level = logging.INFO)

# -----------------------------------------------------------------------------
# reading the models

Generic_osimModel = opensim.Model(Generic_osimModel_file)
Specific_osimModel  = opensim.Model(Specific_osimModel_file)

# Generic_osimModel.initSystem()
# Specific_osimModel.initSystem()

print('------------------------------------------')
print('     ADDING FOOT SCALED OPENSIM MODEL     ')
print('------------------------------------------')
print('Specific Model Name: ', Specific_osimModel.getName())
print('Generic Model Name: ', Generic_osimModel.getName())

# set bies from scaled model to copy into specific model
listOfBodiesFoot = ['talus_' + curr_side, 'calcn_' + curr_side, 'toes_' + curr_side]
listOfJointsFoot = ['ankle_' + curr_side, 'subtalar_' + curr_side, 'mtp_' + curr_side]

for body in listOfBodiesFoot:
    
    # get body from scaled model 
    scaled_body = Generic_osimModel.getBodySet().get(body)
    
    #  add scaled body to specific model
    Specific_osimModel.addBody(scaled_body)

Generic_MarkerSet = Generic_osimModel.getMarkerSet()
Specific_MarkerSet = Specific_osimModel.getMarkerSet()

for marker in Specific_MarkerSet.getComponentsList():
    if marker.getName() == 'RANK':
        tmp = marker.get_location()
        ank_r = np.array([tmp[0], tmp[1], tmp[2]]) 
    if marker.getName() == 'RMMA':
        tmp = marker.get_location()
        mma_r = np.array([tmp[0], tmp[1], tmp[2]])

ankle_r = 0.5*(ank_r + mma_r)

for joint in listOfJointsFoot:
    
    # get body from scaled model 
    scaled_joint = Generic_osimModel.getJointSet().get(joint)
    if joint == 'ankle_r':
        
        scaled_joint.get_frames(0).set_translation(opensim.Vec3(ankle_r[0],ankle_r[1],ankle_r[2]))
        orientation = Specific_osimModel.getJointSet().get('knee_r').get_frames(1).get_orientation()
        scaled_joint.get_frames(0).set_orientation(orientation)
        
    #  add scaled body to specific model
    Specific_osimModel.addJoint(scaled_joint)

Specific_osimModel.initSystem()

newMarkerSet = Specific_MarkerSet.clone()

for marker in Generic_MarkerSet.getComponentsList():
        
    socket_frame = marker.getParentFrameName()
        
    if socket_frame[9:] in listOfBodiesFoot:
        
        newMarkerSet.addComponent(marker)
        
        print('    * ' + marker.getName() + ' added')

Specific_osimModel.set_MarkerSet(newMarkerSet)

# finalize connections
Generic_osimModel.finalizeConnections
Specific_osimModel.finalizeConnections

# print
Specific_osimModel.printToXML(os.path.join(output_models_folder, output_model_file_name))








