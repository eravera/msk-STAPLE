#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 18:41:14 2024

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

output_model_file_name = 'Carman_auto2020_HipKnee_R_ScaledFoot_MUSCLES.osim'


# STAPLE generated model that we want to merge with the generic baseline
Specific_osimModel_file = os.path.join(path_file, output_models_folder, 'Carman_auto2020_HipKnee_R_ScaledFoot.osim')

Specific_osimModel  = opensim.Model(Specific_osimModel_file)

# Create and set the parameters for the biceps muscle
muscle1 = opensim.Millard2012EquilibriumMuscle("muscle_1",  # Muscle name
                                           100.0,  # Max isometric force
                                           0.6,  # Optimal fiber length
                                           0.55,  # Tendon slack length
                                           0.0)  # Pennation angle

# Add path points to the humerus and radius. The allows the muscle to generate
# forces on these two bodies.

origin = Specific_osimModel.getBodySet().get('pelvis')
insertion = Specific_osimModel.getBodySet().get('femur_r')

muscle1.addNewPathPoint("origin",
                       origin,
                       opensim.Vec3(0, 0.3, 0))
muscle1.addNewPathPoint("insertion",
                       insertion,
                       opensim.Vec3(0, 0.2, 0))


Specific_osimModel.addForce(muscle1)

Specific_osimModel.initSystem()


# finalize connections
Specific_osimModel.finalizeConnections

# print
Specific_osimModel.printToXML(os.path.join(output_models_folder, output_model_file_name))

