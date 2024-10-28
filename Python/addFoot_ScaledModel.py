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

output_model_file_name = 'Ejemplo_OpenSimCreator_ScaledFoot.osim'

# body sides
curr_side = 'r'

# generic model to use as baseline model
Generic_osimModel_file = os.path.join(path_file, output_models_folder, 'GAIT2392_SCALED.osim')
# STAPLE generated model that we want to merge with the generic baseline
Specific_osimModel_file = os.path.join(path_file, output_models_folder, 'Ejemplo_OpenSimCreator.osim')


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

# # The ankle parent frame of the generic model will be the parent frame of 
# # the joint connecting the two models.
# specific_tibia_child_frame = Specific_osimModel.getJointSet().get('tibia_' + curr_side).get_frames(1)
# specific_tibia_child_frame.get_translation
# specific_tibia_child_frame.get_orientation

# # update PhysicalOffsetFrame socket_frame
# osimModel.getJointSet().get(cur_joint_name).getParentFrame().getSocket('parent').setConnecteePath(parent_frame.getAbsolutePathString())
# osimModel.getJointSet().get(cur_joint_name).getChildFrame().getSocket('parent').setConnecteePath(child_frame.getAbsolutePathString())

for joint in listOfJointsFoot:
    
    # get body from scaled model 
    scaled_joint = Generic_osimModel.getJointSet().get(joint)
    
    #  add scaled body to specific model
    Specific_osimModel.addJoint(scaled_joint)

Specific_osimModel.initSystem()

Generic_MarkerSet = Generic_osimModel.getMarkerSet()
Specific_MarkerSet = Specific_osimModel.getMarkerSet()

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








