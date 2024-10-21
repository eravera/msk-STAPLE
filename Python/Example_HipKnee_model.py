#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:23:16 2024

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
# from GIBOC_core import *
# from algorithms import *
# from geometry import *
# from anthropometry import *
# from opensim_tools import *
# -----------------------------------------------------------------------
# This example demonstrates how to setup a simple STAPLE workflow to 
# automatically create a model of the hip and the knee joints from the TLEM2 dataset 
# included in the test_geometry folder.
# -----------------------------------------------------------------------

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

# folder where the various datasets (and their geometries) are located.
datasets_folder = path_file + '/bone_datasets/'

# dataset(s) that you would like to process specified as list. 
# If you add multiple datasets they will be batched processed but you will
# have to adapt the folder and file namings below.
# dataset_set = ['TLEM2_simplify']
dataset_set = ['meshes']
# mass = 45 # kg
mass = 44.4 # kg

# body sides
curr_side = 'r'

# list with the name of the bone geometries to process.
bones_list = ['pelvis', 'femur_' + curr_side, 'tibia_' + curr_side]
# bones_list = ['pelvis', 'femur_' + curr_side]

# format of visualization geometry (obj preferred - smaller files)
vis_geom_format = 'stl' # options: 'stl'/'obj'

# choose the definition of the joint coordinate systems (see documentation). 
# options: 'Modenese2018' or 'auto2020'
workflow = 'auto2020'

#%% ---------------------------------------------------------------------------
tic = time.time()
# create model folder if required
dst = Path(output_models_folder)
dst.mkdir(parents=True, exist_ok=True)
dst = str(dst)
# -------------------------------------------------

# setup for batch processing
# dataset id used to name OpenSim model and setup folders
for curr_dataset in dataset_set:
    
    # # infer body side
    # curr_side = inferBodySideFromAnatomicStruct(bones_list)
        
    # model name
    # curr_model_name = 'example_' + workflow + '_Hip_' + curr_side.upper()
    # curr_model_name = 'example_' + workflow + '_HipKnee_' + curr_side.upper()
    curr_model_name = 'Carman_' + workflow + '_HipKnee_' + curr_side.upper()
    # curr_model_name = 'Carman_' + workflow + '_Hip_' + curr_side.upper()
    
    # set output model name
    output_model_file_name = curr_model_name + '.osim'
    
    # log printout
    log_folder = Path(output_models_folder)
    logging.basicConfig(filename = str(log_folder) + '/' + curr_model_name + '.log', filemode = 'w', format = '%(levelname)s:%(message)s', level = logging.INFO)

    # foolder including the bone geometries in ('tri'/'stl') format
    tri_folder = os.path.join(datasets_folder, curr_dataset, 'stl/')
    
    # create TriGeomSet dictionary for the specified geometries
    triGeom_set = createTriGeomSet(bones_list, tri_folder)
    
    # get the body side (can also be specified by user as input to funcs)
    side = inferBodySideFromAnatomicStruct(triGeom_set)
    
    # create bone geometry folder for visualization
    geometry_folder_name = curr_model_name + '_Geometry'
    geometry_folder_path = os.path.join(output_models_folder, geometry_folder_name)
    writeModelGeometriesFolder(triGeom_set, geometry_folder_path, vis_geom_format)
    
    # process bone geometries (compute joint parameters and identify markers)
    [JCS, BL, CS] = processTriGeomBoneSet(triGeom_set, side)
    
    
    # initialize OpenSim model
    osimModel = initializeOpenSimModel(curr_model_name)
    
    # create bodies
    osimModel = addBodiesFromTriGeomBoneSet(osimModel, triGeom_set, geometry_folder_name, vis_geom_format)
    
    # create joints
    osimModel = createOpenSimModelJoints(osimModel, JCS, workflow)
    
    # update mass properties to those estimated using a scale version of
    # gait2392 with COM based on Winters's book.
    osimModel = assignMassPropsToSegments(osimModel, JCS, mass)
    
    # add markers to the bones
    addBoneLandmarksAsMarkers(osimModel, BL)
    
    # finalize connections
    osimModel.finalizeConnections
    
    # print
    osimModel.printToXML(os.path.join(output_models_folder, output_model_file_name))
    
    # inform the user about time employed to create the model
    print('-------------------------')
    print('Model generated in ' + str(np.round(time.time() - tic,1)) + ' s')
    print('Saved as ' + output_models_folder + '/' + output_model_file_name + '.')
    print('Model geometries saved in folder: ' + geometry_folder_path + '.')
    print('-------------------------')
    # logConsolePrintout('off')















