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

# from algorithms import STAPLE_pelvis

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

# -----------------------------------------------------------------------------
def bodySide2Sign(side_raw):
    # -------------------------------------------------------------------------
    # Returns a sign and a mono-character, lower-case string corresponding to 
    # a body side. Used in several STAPLE functions for:
    # 1) having a standard side label
    # 2) adjust the reference systems to the body side.
    # 
    # Inputs:
    #   side_raw - generic string identifying a body side. 'right', 'r', 'left' 
    # and 'l' are accepted inputs, both lower and upper cases.
    # 
    #   Outputs:
    #   sign_side - sign to adjust reference systems based on body side. Value:
    #   1 for right side, Value: -1 for left side.
    # 
    #   side_low - a single character, lower case body side label that can be 
    #   used in all other STAPLE functions requiring such input.
    # -------------------------------------------------------------------------
    
    side_low = side_raw[0].lower()
    
    if side_low == 'r':
        sign_side = 1
    elif side_low == 'l':
        sign_side = -1
    else:
        print('bodySide2Sign Error: specify right "r" or left "l"')
    
    return sign_side, side_low

# -----------------------------------------------------------------------------
def processTriGeomBoneSet(triGeomBoneSet, side_raw = '', algo_pelvis = 'STAPLE', algo_femur = 'GIBOC-cylinder', algo_tibia = 'Kai2014', result_plots = 1, debug_plots = 0, in_mm = 1):
    # -------------------------------------------------------------------------
    #  Compute parameters of the lower limb joints associated with the bone 
    # geometries provided as input through a set of triangulations dictionary.
    # Note that:
    # 1) This function does not produce a complete set of joint parameters but 
    # only those available through geometrical analyses, that are:
    #       * from pelvis: ground_pelvis_child
    #       * from femur : hip_child  // knee_parent
    #       * from tibia : knee_child** 
    #       * from talus : ankle_child // subtalar parent
    # other functions then complete the information required to generate a MSK
    # model, e.g. use ankle child location and ankle axis to define an ankle
    # parent reference system etc.
    # ** note that the tibia geometry was not used to define the knee child
    # anatomical coord system in the approach of Modenese et al. JB 2018.
    # 2) Bony landmarks are identified on all bones except the talus.
    # 3) Body-fixed Cartesian coordinate system are defined but not employed in
    # the construction of the models.
    # 
    # Inputs:
    #   geom_set - a set of triangulation dictionary, normally created
    #   using the function createTriGeomSet. See that function for more
    #   details.
    # 
    #   side_raw - generic string identifying a body side. 'right', 'r', 'left' 
    #   and 'l' are accepted inputs, both lower and upper cases.
    # 
    #   algo_pelvis - the algorithm selected to process the pelvis geometry.
    # 
    #   algo_femur - the algorithm selected to process the femur geometry.
    # 
    #   algo_tibia - the algorithm selected to process the tibial geometry.
    # 
    #   result_plots - enable plots of final fittings and reference systems. 
    #   Value: 1 (default) or 0.
    # 
    #   debug_plots - enable plots used in debugging. Value: 1 or 0 (default). 
    # 
    #   in_mm - (optional) indicates if the provided geometries are given in mm
    #   (value: 1) or m (value: 0). Please note that all tests and analyses
    #   done so far were performed on geometries expressed in mm, so this
    #   option is more a placeholder for future adjustments.
    # 
    # Outputs:
    #   JCS - dictionary collecting the parameters of the joint coordinate
    #   systems computed on the bone triangulations.
    # 
    #   BL - dictionary collecting the bone landmarks identified on the
    #   three-dimensional bone surfaces.
    # 
    #   BCS - dictionary collecting the body coordinate systems of the 
    #   processed bones.
    # -------------------------------------------------------------------------
    
    # setting defaults
    
    if side_raw == '':
        side = inferBodySideFromAnatomicStruct(triGeomBoneSet)
    else:
        # get sign correspondent to body side
        _, side = bodySide2Sign(side_raw)
    
    # names of the segments
    femur_name = 'femur_' + side
    tibia_name = 'tibia_' + side
    patella_name = 'patella_' + side
    talus_name = 'talus_' + side
    calcn_name = 'calcn_' + side 
    toes_name  = 'toes_' + side
    
    BCS = {}
    JCS = {}
    BL = {}
    
    print('-----------------------------------')
    print('Processing provided bone geometries')
    print('-----------------------------------')
    
    # visualization of the algorithms that will be used (depends on available
    # segments
    print('ALGORITHMS:')
    
    if 'pelvis' in triGeomBoneSet or 'pelvis_no_sacrum' in triGeomBoneSet:
        BCS['pelvis'] = {}
        JCS['pelvis'] = {}
        BL['pelvis'] = {}
        print('  pelvis: ', algo_pelvis)
    if femur_name in triGeomBoneSet:
        BCS['femur'] = {}
        JCS['femur'] = {}
        BL['femur'] = {}
        print('  femur: ', algo_femur)
    if tibia_name in triGeomBoneSet:
        BCS['tibia'] = {}
        JCS['tibia'] = {}
        BL['tibia'] = {}
        print('  tibia: ', algo_tibia)
    if patella_name in triGeomBoneSet:
    #     print('  patella: ', algo_patella)
        print('  patella: ', 'N/A')
    if talus_name in triGeomBoneSet:
        BCS['talus'] = {}
        JCS['talus'] = {}
        BL['talus'] = {}
        print('  talus: ', 'STAPLE')
    if calcn_name in triGeomBoneSet:
        BCS['foot'] = {}
        JCS['foot'] = {}
        BL['foot'] = {}
        print('  foot: ', 'STAPLE')
    if toes_name in triGeomBoneSet:
        print('  toes: ', 'N/A')
        
    # # ---- PELVIS -----
    # if 'pelvis' in triGeomBoneSet:
    #     if algo_pelvis == 'STAPLE':
    #         BCS['pelvis'], JCS['pelvis'], BL['pelvis']  = \
    #             STAPLE_pelvis(triGeomBoneSet['pelvis'], side, result_plots, debug_plots, in_mm)
    # #     if algo_pelvis == 'Kai2014':
    # #         # BCS['pelvis'], JCS['pelvis'], BL['pelvis']  = \
    # #         #     Kai2014_pelvis(triGeomBoneSet['pelvis'], side, result_plots, debug_plots, in_mm)
    # elif 'pelvis_no_sacrum' in triGeomBoneSet:
    #     if algo_pelvis == 'STAPLE':
    #         BCS['pelvis'], JCS['pelvis'], BL['pelvis']  = \
    #             GIBOC_femur(triGeomBoneSet['pelvis_no_sacrum'], side, result_plots, debug_plots, in_mm)
    # #     if algo_pelvis == 'Kai2014':
    # #         # BCS['pelvis'], JCS['pelvis'], BL['pelvis']  = \
    # #         #     Kai2014_pelvis(triGeomBoneSet['pelvis_no_sacrum'], side, result_plots, debug_plots, in_mm)
    
    # # # ---- FEMUR -----
    # # if femur_name in triGeomBoneSet:
    # #     if 'GIBOC' in femur_name:
    # #         # BCS[femur_name], JCS[femur_name], BL[femur_name] = \
    # #         #     GIBOC_femur(triGeomBoneSet['femur_name'], side, femur_name[6:], result_plots, debug_plots, in_mm)
    
    
    
    
    return 0




