#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR(S) AND VERSION-HISTORY

Copyright (c) 2020 Modenese L.
Author:   Luca Modenese
email:    l.modenese@imperial.ac.uk
  
Pyhton Version:

Coding by Emiliano Ravera, Apr 2024. Adapted to Python 3.11
email:    emiliano.ravera@uner.edu.ar

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

from geometry import inferBodySideFromAnatomicStruct, \
                        createTriGeomSet

# -----------------------------------------------------------------------
# This example demonstrates how to setup a simple STAPLE workflow to 
# automatically create a model of the hip joint from the LHDL-CT dataset 
# included in the test_geometry folder.
# -----------------------------------------------------------------------

#%% ----------
#  SETTINGS 
# ----------

# Set to Executables directory if running in Python directly
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# set output folder
output_models_folder = 'opensim_models_examples/'

# folder where the various datasets (and their geometries) are located.
datasets_folder = 'bone_datasets/'

# dataset(s) that you would like to process specified as list. 
# If you add multiple datasets they will be batched processed but you will
# have to adapt the folder and file namings below.
dataset_set = ['LHDL_CT']
body_mass = 64 # kg

# list with the name of the bone geometries to process.
bones_list = ['pelvis_no_sacrum', 'femur_r']

# format of visualization geometry (obj preferred - smaller files)
vis_geom_format = 'obj' # options: 'stl'/'obj'

# choose the definition of the joint coordinate systems (see documentation). 
# For hip joint creation this option has no effect.
joint_defs = 'auto2020'

#%% --------------------------------------
# create model folder if required
dst = Path(output_models_folder)
dst.mkdir(parents=True, exist_ok=True)
dst = str(dst)
# -------------------------------------------------

# setup for batch processing
# dataset id used to name OpenSim model and setup folders
for curr_dataset in dataset_set:
    
    # infer body side
    curr_side = inferBodySideFromAnatomicStruct(bones_list)
        
    # model name
    curr_model_name = 'example_' + joint_defs + '_hip_' + curr_side.upper()
    
    # set output model name
    output_model_file_name = curr_model_name + '.osim'
    
    # log printout
    log_folder = Path(output_models_folder)
    logging.basicConfig(filename = str(log_folder) + '/' + curr_model_name + '.log', filemode = 'w', format = '%(levelname)s:%(message)s', level = logging.INFO)

    # foolder including the bone geometries in ('tri'/'stl') format
    tri_folder = os.path.join(datasets_folder, curr_dataset, 'tri')
    
    # create TriGeomSet dictionary for the specified geometries
    geom_set = createTriGeomSet(bones_list, tri_folder)
    
    # create bone geometry folder for visualization
    geometry_folder_name = curr_model_name, '_Geometry'
    geometry_folder_path = os.path.join(output_models_folder, geometry_folder_name)
    # writeModelGeometriesFolder(geom_set, geometry_folder_path, vis_geom_format)






