#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR(S) AND VERSION-HISTORY

Author:   Luca Modenese & Jean-Baptiste Renault. 
Copyright 2020 Luca Modenese & Jean-Baptiste Renault

Pyhton Version:

Coding by Emiliano Ravera, Apr 2024. Adapted to Python 3.11
email:    emiliano.ravera@uner.edu.ar

@author: emi
"""
# ----------- import packages --------------
import numpy as np
import pandas as pd
import os, shutil
from pathlib import Path
import sys
import time
import logging


from Public_functions import load_mesh

# -----------------------------------------------------------------------------
def inferBodySideFromAnatomicStruct(anat_struct):
    # -------------------------------------------------------------------------
    # Infer the body side that the user wants to process based on a structure 
    # containing the anatomical objects (triangulations or joint definitions) 
    # given as input. The implemented logic is trivial: the fields are checked 
    # for standard names of bones and joints used in OpenSim models.
    # 
    # 
    # Inputs: anat_struct - a list or dic containing anatomical objects, 
    # e.g. a set of bone triangulation or joint definitions.
    # 
    # Outputs: guessed_side - a body side label that can be used in all other 
    # STAPLE functions requiring such input.
    # -------------------------------------------------------------------------
    guessed_side = ''
    
    if isinstance(anat_struct, dict):
        fields_side = list(anat_struct.keys())
    elif isinstance(anat_struct, list):
        fields_side = anat_struct
    else:
        print('inferBodySideFromAnatomicStruct Input must be a list.')
    
    # check using the bone names
    body_set = ['femur', 'tibia', 'talus', 'calcn']
    guess_side_b = []
    guess_side_b = [body[-1] for body in fields_side if body[:-2] in body_set]
    
    # check using the joint names
    joint_set = ['hip', 'knee', 'ankle', 'subtalar']
    guess_side_j = []
    guess_side_j = [joint[-1] for joint in fields_side if joint[:-2] in joint_set]
    
    # composed vectors
    combined_guessed = guess_side_b + guess_side_j
        
    if 'r' in list(set(combined_guessed)) and not 'l' in list(set(combined_guessed)):
        guessed_side = 'r'
    elif 'l' in list(set(combined_guessed)) and not 'r' in list(set(combined_guessed)):
        guessed_side = 'l'
    else:
        print('inferBodySideFromAnatomicStruct Error: it was not possible to infer the body side. Please specify it manually in this occurrance.')
    
    return guessed_side

# -----------------------------------------------------------------------------
def createTriGeomSet(aTriGeomList, geom_file_folder):
    # -------------------------------------------------------------------------
    # CREATETRIGEOMSET Create a dictionary of triangulation objects from a list
    # of files and the folder where those files are located
    
    # triGeomSet = createTriGeomSet(aTriGeomList, geom_file_folder)
    
    # Inputs:
    #   aTriGeomList - a list consisting of names of triangulated geometries 
    #       that will be seeked in geom_file_folder.
    #       The names set the name of the triangulated objects and should
    #       correspond to the names of the bones to include in the OpenSim
    #       models.
    
    #   geom_file_folder - the folder where triangulated geometry files will be
    #       seeked.
    
    # Outputs:
    #   triGeomSet - dictionary with as many fields as the triangulations that
    #       could be loaded from the aTriGeomList from the geom_file_folder.
    #       Each field correspond to a triangulated geometry.
    
    # Example of use:
    # tri_folder = 'test_geometries\dataset1\tri';
    # bones_list = {'pelvis','femur_r','tibia_r','talus_r','calcn_r'};
    # geom_set = createTriGeomSet(bones_list, tri_folder);
    # -------------------------------------------------------------------------
    
    print('-----------------------------------')
    print('Creating set of triangulations')
    print('-----------------------------------')
    
    t0 = time.time() # tf = time.time() - t
    
    print('Reading geometries from folder: ' +  geom_file_folder)
    
    triGeomSet = {}
    for bone in aTriGeomList:
        
        curr_tri_geo_file = os.path.join(geom_file_folder, bone)
        curr_tri_geo = load_mesh(curr_tri_geo_file)
        if not curr_tri_geo:
            # skip building the field if this was no gemoetry
            continue
        else:
            triGeomSet[bone] = curr_tri_geo
        
        if not triGeomSet:
            logging.exception('createTriGeomSet No triangulations were read in input. \n')
        else:
            # tell user all went well
            print('Set of triangulated geometries created in ' + str(np.round(time.time() - t0, 4)))
        
    return 0







