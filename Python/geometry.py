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
from sklearn import preprocessing
import matplotlib.pyplot as plt


from Public_functions import load_mesh

from GIBOC_core import TriChangeCS, \
                        plotTriangLight, \
                         quickPlotRefSystem
                        
# from algorithms import STAPLE_pelvis, \
#                         GIBOC_femur, \
#                           Kai2014_tibia

from opensim_tools import computeXYZAngleSeq

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
        
    return triGeomSet

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
def getBoneLandmarkList(bone_name):
    # -------------------------------------------------------------------------
    # Given a bone name, returns names and description of the bony landmark 
    # identifiable on its surface in a convenient cell array.
    # 
    # Inputs:
    # bone_name - a string indicating a bone of the lower limb
    # 
    # Outputs:
    # LandmarkInfo - cell array containing the name and keywords to identify 
    # the bony landmarks on each bone triangulation. This information can 
    # easily used as input to findLandmarkCoords and landmarkTriGeomBone.
    # -------------------------------------------------------------------------
    
    LandmarkInfo = {}
    
    # used notation to describe the landmarks
    # LandmarkInfo['0'] = BL name
    # LandmarkInfo['1'] = axis
    # LandmarkInfo['2'] = operator (max/min)
    # 1,2 can repeat
    # LandmarkInfo['end'] = proximal/distal (optional)
    
    if bone_name == 'pelvis':
        LandmarkInfo['0'] = ['RASI', 'x', 'max', 'z', 'max']
        LandmarkInfo['1'] = ['LASI', 'x', 'max', 'z', 'min']
        LandmarkInfo['2'] = ['RPSI', 'x', 'min', 'z', 'max']
        LandmarkInfo['3'] = ['LPSI', 'x', 'min', 'z', 'min']
    elif bone_name == 'femur_r':
        LandmarkInfo['0'] = ['RKNE', 'z', 'max', 'distal']
        LandmarkInfo['1'] = ['RMFC', 'z', 'min', 'distal']
        LandmarkInfo['2'] = ['RTRO', 'z', 'max', 'proximal']
    elif bone_name == 'femur_l':
        LandmarkInfo['0'] = ['LKNE', 'z', 'min', 'distal']
        LandmarkInfo['1'] = ['LMFC', 'z', 'max', 'distal']
        LandmarkInfo['2'] = ['LTRO', 'z', 'min', 'proximal']
    elif bone_name == 'tibia_r':
        LandmarkInfo['0'] = ['RTTB', 'x', 'max', 'proximal']
        LandmarkInfo['1'] = ['RHFB', 'z', 'max', 'proximal']
        LandmarkInfo['2'] = ['RANK', 'z', 'max', 'distal']
        LandmarkInfo['3'] = ['RMMA', 'z', 'min', 'distal']
    elif bone_name == 'tibia_l':
        LandmarkInfo['0'] = ['LTTB', 'x', 'max', 'proximal']
        LandmarkInfo['1'] = ['LHFB', 'z', 'max', 'proximal']
        LandmarkInfo['2'] = ['LANK', 'z', 'max', 'distal']
        LandmarkInfo['3'] = ['LMMA', 'z', 'min', 'distal']
    elif bone_name == 'patella_r':
        LandmarkInfo['0'] = ['RLOW', 'y', 'min', 'distal']
    elif bone_name == 'patella_l':
        LandmarkInfo['0'] = ['LLOW', 'y', 'min', 'distal']
    elif bone_name == 'calcn_r':
        LandmarkInfo['0'] = ['RHEE', 'x', 'min']
        LandmarkInfo['1'] = ['RD5M', 'z', 'max']
        LandmarkInfo['2'] = ['RD1M', 'z', 'min']
    elif bone_name == 'calcn_l':
        LandmarkInfo['0'] = ['LHEE', 'x', 'min']
        LandmarkInfo['1'] = ['LD5M', 'z', 'min']
        LandmarkInfo['2'] = ['LD1M', 'z', 'max']
    # included for advanced example on humerus
    elif bone_name == 'humerus_r':
        LandmarkInfo['0'] = ['RLE', 'z', 'max', 'distal']
        LandmarkInfo['1'] = ['RME', 'z', 'min', 'distal']
    elif bone_name == 'humerus_l':
        LandmarkInfo['0'] = ['LLE', 'z', 'min', 'distal']
        LandmarkInfo['1'] = ['LME', 'z', 'max', 'distal']
    else:
        # loggin.error('getBoneLandmarkList.m specified bone name is not supported yet.')
        print('getBoneLandmarkList.m specified bone name is not supported yet.')
    
    return LandmarkInfo
    
# -----------------------------------------------------------------------------
def findLandmarkCoords(points, axis_name, operator):
    # -------------------------------------------------------------------------
    # Pick points from a point cloud using string keywords.
    # 
    # Inputs:
    # points - indicates a point cloud, i.e. a matrix of [N_point, 3] dimensions.
    # 
    # axis_name - specifies the axis where the min/max operator will be applied.
    # Assumes values: 'x', 'y' or 'z', corresponding to first, second or third
    # column of the points input.
    # 
    # operator - specifies which point to pick in the specified axis.
    # Currently assumes values: 'min' or 'max', meaning smalles or
    # largest value in that direction.
    # 
    # Outputs:
    # BL - coordinates of the point identified in the point cloud following
    # the directions provided as input. 
    # 
    # BL_ind - index of the point identified in the point cloud following the
    # direction provided as input.
    # 
    # Example:
    # getBonyLandmark(pcloud,'max','z') asks to pick the point in the pcloud 
    # with largest z coordinate. If pcloud was a right distal femur triangulation
    # with points expressed in the reference system of the International
    # Society of Biomechanics, this could be the lateral femoral epicondyle.
    # -------------------------------------------------------------------------
    
    # interpreting direction of searchv
    if axis_name == 'x':
        dir_ind = 0
    elif axis_name == 'y':
        dir_ind = 1
    elif axis_name == 'z':
        dir_ind = 2
    else:
        print('axis_name no identified')
    
    # interpreting description of search
    if operator == 'max':
        BL_ind = np.argmax(points[:,dir_ind])
    elif operator == 'min':
        BL_ind = np.argmin(points[:,dir_ind])
    else:
        print('operator no identified')
    
    BL = points[BL_ind]
    BL = np.reshape(BL,(BL.size, 1)) # convert 1d (3,) to 2d (3,1) vector
        
    return BL, BL_ind
    
# -----------------------------------------------------------------------------
def landmarkBoneGeom(TriObj, CS, bone_name, debug_plots = 0):
    # -------------------------------------------------------------------------
    # Locate points on the surface of triangulation objects that are bone 
    # geometries.
    # 
    # Inputs:
    # TriObj - a triangulation object.
    # 
    # CS - a structure representing a coordinate systems as a structure with 
    # 'Origin' and 'V' fields, indicating the origin and the axis. 
    # In 'V', each column is the normalise direction of an axis 
    # expressed in the global reference system (CS.V = [x, y, z]).
    # 
    # bone_name - string identifying a lower limb bone. Valid values are:
    # 'pelvis', 'femur_r', 'tibia_r', 'patella_r', 'calcn_r'.
    # 
    # debug_plots - takes value 1 or 0. Plots the steps of the landmarking
    # process. Switched off by default, useful for debugging.
    # 
    # Outputs:
    # Landmarks - structure with as many fields as the landmarks defined in
    # getBoneLandmarkList.m. Each field has the name of the landmark and
    # value the 3D coordinates of the landmark point.
    # -------------------------------------------------------------------------
    # get info about the landmarks in the current bone
    LandmarkDict = getBoneLandmarkList(bone_name)

    # change reference system to bone/body reference system
    TriObj_in_CS, _, _ = TriChangeCS(TriObj, CS['V'], CS['CenterVol'])

    # visualise the bone in the bone/body ref system
    if debug_plots:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        
        # create coordinate system centred in [0, 0 ,0] and with unitary axis
        # directly along the direction of the ground ref system
        LocalCS = {}
        LocalCS['Origin'] = np.array([0,0,0])
        LocalCS['V'] = np.eye(3)
        
        plotTriangLight(TriObj_in_CS, LocalCS, ax)
        quickPlotRefSystem(LocalCS, ax)

    # get points from the triangulation
    TriPoints = TriObj_in_CS['Points']

    # define proximal or distal:
    # 1) considers Y the longitudinal direction
    # 2) dividing the bone in three parts
    ub = np.max(TriPoints[:,1])*0.3
    lb = np.max(TriPoints[:,1])*(-0.3)

    if debug_plots:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.set_title('proximal (blue) and distal (red) geom')
        # check proximal geometry
        check_proximal = TriPoints[TriPoints[:,1] > ub]
        for p in check_proximal:
            ax.scatter(p[0], p[1], p[2], color = 'blue')
        
        # check distal geometry
        check_distal = TriPoints[TriPoints[:,1] < lb]
        for p in check_distal:
            ax.scatter(p[0], p[1], p[2], color = 'red')

    Landmarks = {}
    for cur_BL_info in LandmarkDict.values():
        # extract info (see LandmarkStruct for details)
        cur_BL_name = cur_BL_info[0] # landmark name
        cur_axis = cur_BL_info[1] # x/y/z
        cur_operator = cur_BL_info[2] # min/max
        cur_bone_extremity = cur_BL_info[-1]
        
        # is the BL proximal or distal?
        if cur_bone_extremity == 'proximal':
            # identify the landmark
            local_BL, _ = findLandmarkCoords(TriPoints[TriPoints[:,1] > ub], cur_axis, cur_operator)
        elif cur_bone_extremity == 'distal':
            # identify the landmark
            local_BL, _ = findLandmarkCoords(TriPoints[TriPoints[:,1] < lb], cur_axis, cur_operator)
        else:
            # get landmark if no bone extremity is specified
            local_BL, _ = findLandmarkCoords(TriPoints, cur_axis, cur_operator)
        
        # save a landmark structure (transform back to global)
        Landmarks[cur_BL_name] = CS['CenterVol'] + np.dot(CS['V'],local_BL)
        
    return Landmarks

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
        
    # ---- PELVIS -----
    if 'pelvis' in triGeomBoneSet:
        if algo_pelvis == 'STAPLE':
            BCS['pelvis'], JCS['pelvis'], BL['pelvis']  = \
                STAPLE_pelvis(triGeomBoneSet['pelvis'], side, result_plots, debug_plots, in_mm)
    #     if algo_pelvis == 'Kai2014':
    #         # BCS['pelvis'], JCS['pelvis'], BL['pelvis']  = \
    #         #     Kai2014_pelvis(triGeomBoneSet['pelvis'], side, result_plots, debug_plots, in_mm)
    elif 'pelvis_no_sacrum' in triGeomBoneSet:
        if algo_pelvis == 'STAPLE':
            BCS['pelvis'], JCS['pelvis'], BL['pelvis']  = \
                STAPLE_pelvis(triGeomBoneSet['pelvis_no_sacrum'], side, result_plots, debug_plots, in_mm)
    #     if algo_pelvis == 'Kai2014':
    #         # BCS['pelvis'], JCS['pelvis'], BL['pelvis']  = \
    #         #     Kai2014_pelvis(triGeomBoneSet['pelvis_no_sacrum'], side, result_plots, debug_plots, in_mm)
    
    # ---- FEMUR -----
    if femur_name in triGeomBoneSet:
        if 'GIBOC' in triGeomBoneSet:
            BCS['femur'], JCS['femur'], BL['femur'] = \
                GIBOC_femur(triGeomBoneSet['femur_name'], side, triGeomBoneSet[6:], result_plots, debug_plots, in_mm)
        # if 'Miranda' in triGeomBoneSet:
        #     BCS[femur_name], JCS[femur_name], BL[femur_name] = \
        #         Miranda2010_buildfACS(triGeomBoneSet['femur_name'])
        # if 'Kai2014' in triGeomBoneSet:
        #     BCS[tibia_name], JCS[tibia_name], BL[tibia_name] = \
        #         Kai2014_femur(triGeomBoneSet['femur_name'], side)
        else:
            BCS['femur'], JCS['femur'], BL['femur'] = \
                GIBOC_femur(triGeomBoneSet['femur_name'], side, triGeomBoneSet[6:], result_plots, debug_plots, in_mm)
    
    # ---- TIBIA -----
    if tibia_name in triGeomBoneSet:
        # if 'GIBOC' in triGeomBoneSet:
        #     BCS[tibia_name], JCS[tibia_name], BL[tibia_name] = \
        #         GIBOC_tibia(triGeomBoneSet['tibia_name'], side, triGeomBoneSet[6:], result_plots, debug_plots, in_mm)
        if 'Kai2014' in triGeomBoneSet:
            BCS[tibia_name], JCS[tibia_name], BL[tibia_name] = \
                Kai2014_tibia(triGeomBoneSet['tibia_name'], side, result_plots, debug_plots, in_mm)
        else:
            BCS[tibia_name], JCS[tibia_name], BL[tibia_name] = \
                Kai2014_tibia(triGeomBoneSet['tibia_name'], side, result_plots, debug_plots, in_mm)
    

    return JCS, BL, BCS

# -----------------------------------------------------------------------------
def compileListOfJointsInJCSStruct(JCS = {}):
    # -------------------------------------------------------------------------
    # Create a list of joints that can be modelled from the structure that 
    # results from morphological analysis.
    # 
    # Inputs:
    # JCS - dictionary with the joint parameters produced by the morphological 
    # analyses of processTriGeomBoneSet. Not all listed joints are
    # actually modellable, in the sense that the parent and child
    # reference systems might not be present, the model might be incomplete etc.
    # 
    # Outputs:
    # joint_list - a list of unique elements. Each element is the name of a 
    # joint present in the JCS structure.
    # -------------------------------------------------------------------------
    joint_list = []
    
    for body in JCS:
        joint_list += [joint for joint in JCS[body] if joint not in joint_list]

    return joint_list

# -----------------------------------------------------------------------------
def jointDefinitions_auto2020(JCS = {}, jointStruct = {}):
    # -------------------------------------------------------------------------
    # Define the orientation of the parent reference system in the ankle joint 
    # using the ankle axis as Z axis and the long axis of the tibia 
    # (made perpendicular to Z) as Y axis. X defined by cross-product. The 
    # ankle is the only joint that is not defined neither in parent or child 
    # in the "default" joint definition named 'auto2020'. 
    # 
    # Inputs:
    # JCS - dictionary with the joint parameters produced by the morphological 
    # analyses of processTriGeomBoneSet. Not all listed joints are
    # actually modellable, in the sense that the parent and child
    # reference systems might not be present, the model might be incomplete etc.
    # In this function only the field `V` relative to talus and tibia will be 
    # recalled.
    # 
    # jointStruct - Dictionary including all the reference parameters that will
    # be used to generate an OpenSim JointSet.
    # 
    # Outputs:
    # jointStruct - updated dicttionary with a newly defined ankle joint parent
    # ankle V.
    # -------------------------------------------------------------------------
    
    side_low = inferBodySideFromAnatomicStruct(JCS)
    
    # bone names
    tibia_name = 'tibia_' + side_low
    talus_name = 'talus_' + side_low
    
    # joint names
    ankle_name = 'ankle_' + side_low
    knee_name = 'knee_' + side_low
    
    # joint params: JCS[bone_name] will access the geometrical information
    # from the morphological analysis
    
    # take Z from ankle joint (axis of rotation)
    if talus_name in JCS and tibia_name in JCS:
        Zpar = JCS[talus_name][ankle_name]['V'][:,2]
        Zpar = np.reshape(Zpar,(Zpar.size, 1)) # convert 1d (3,) to 2d (3,1) vector
        Ytemp = JCS[tibia_name][knee_name]['V'][:,1]
        Ytemp = np.reshape(Ytemp,(Ytemp.size, 1)) # convert 1d (3,) to 2d (3,1) vector
        
        # Y and Z orthogonal
        Ypar = preprocessing.normalize(Ytemp - Zpar*np.dot(Zpar, Ytemp), axis =0)
        Xpar = preprocessing.normalize(np.cross(Ypar, Zpar), axis =0)
        
        # assigning pose matrix and parent orientation
        jointStruct[ankle_name] = {}
        jointStruct[ankle_name]['V'] = np.zeros((3,3))
        jointStruct[ankle_name]['V'][:,0] = Xpar[:,0]
        jointStruct[ankle_name]['V'][:,1] = Ypar[:,0]
        jointStruct[ankle_name]['V'][:,2] = Zpar[:,0]
        
        jointStruct[ankle_name]['parent_orientation'] = computeXYZAngleSeq(jointStruct[ankle_name]['V'])
    else:
        return jointStruct
    
    return jointStruct

# -----------------------------------------------------------------------------
def jointDefinitions_Modenese2018(JCS = {}, jointStruct = {}):
    # -------------------------------------------------------------------------
    # Define the orientation of lower limb joints as in Modenese et al. 
    # JBiomech 2018, 17;73:108-118 https://doi.org/10.1016/j.jbiomech.2018.03.039
    # Required for the comparisons presented in Modenese and Renault, JBiomech
    # 2020. 
    # 
    # Inputs:
    # JCS - dictionary with the joint parameters produced by the morphological 
    # analyses of processTriGeomBoneSet. Not all listed joints are
    # actually modellable, in the sense that the parent and child
    # reference systems might not be present, the model might be incomplete etc.
    # 
    # jointStruct - Dictionary including all the reference parameters that will
    # be used to generate an OpenSim CustomJoints.
    # 
    # Outputs:
    # jointStruct - updated jointStruct with the joints defined as in
    # Modenese2018, rather than connected directly using the joint
    # coordinate system computed in processTriGeomBoneSet.py plus the
    # default joint definition available in jointDefinitions_auto2020.py
    # -------------------------------------------------------------------------
    
    side_low = inferBodySideFromAnatomicStruct(JCS)
    
    # joint names
    knee_name = 'knee_' + side_low
    ankle_name = 'ankle_' + side_low
    subtalar_name = 'subtalar_' + side_low
    
    # segments names
    femur_name = 'femur_' + side_low
    tibia_name = 'tibia_' + side_low
    talus_name = 'talus_' + side_low
    calcn_name = 'calcn_' + side_low
    talus_name = 'talus_' + side_low
    mtp_name = 'mtp_' + side_low
    
    if talus_name in JCS:
        TalusDict = JCS[talus_name]
        
        if tibia_name in JCS:
            TibiaDict = JCS[tibia_name]
            
            if femur_name in JCS:
                FemurDict = JCS[femur_name]
                
                # knee child orientation
                # ---------------------
                # Z aligned like the medio-lateral femoral joint, e.g. axis of cylinder
                # Y aligned with the tibial axis (v betw origin of ankle and knee)
                # X from cross product
                # ---------------------
                
                # take Z from knee joint (axis of rotation)
                Zparent  = FemurDict[knee_name]['V'][:,2]
                Zparent = np.reshape(Zparent,(Zparent.size, 1)) # convert 1d (3,) to 2d (3,1) vector
                # take line joining talus and knee centres
                TibiaDict[knee_name]['Origin'] = FemurDict[knee_name]['Origin']
                # vertical axis joining knee and ankle joint centres (same used for ankle
                # parent)
                Ytemp = (TibiaDict[knee_name]['Origin'] - FemurDict[ankle_name]['Origin'])
                Ytemp /= np.linalg.norm(TibiaDict[knee_name]['Origin'] - FemurDict[ankle_name]['Origin'], axis=0)
                # make Y and Z orthogonal
                Yparent = preprocessing.normalize(Ytemp - Zparent*np.dot(Zparent, Ytemp), axis =0)
                Xparent = preprocessing.normalize(np.cross(Ytemp, Zparent), axis =0)
                # assigning pose matrix and child orientation
                tmp_V = np.zeros((3,3))
                tmp_V[:,0] = Xparent[:,0]
                tmp_V[:,1] = Yparent[:,0]
                tmp_V[:,2] = Zparent[:,0]
                
                jointStruct[knee_name]['child_orientation'] = computeXYZAngleSeq(tmp_V)
                
            # Ankle parent orientation
            # ---------------------
            # Z aligned like the cilinder of the talar throclear
            # Y aligned with the tibial axis (v betw origin of ankle and knee)
            # X from cross product
            # ---------------------
            # take Z from ankle joint (axis of rotation)
            Zparent = TalusDict[ankle_name]['V'][:,2]
            Zparent = np.reshape(Zparent,(Zparent.size, 1)) # convert 1d (3,) to 2d (3,1) vector
            # take line joining talus and knee centres
            Ytibia = (TibiaDict[knee_name]['Origin'] - TalusDict[ankle_name]['Origin'])
            Ytibia /= np.linalg.norm(TibiaDict[knee_name]['Origin'] - TalusDict[ankle_name]['Origin'], axis=0)
            # make Y and Z orthogonal
            Yparent = preprocessing.normalize(Ytibia - Zparent*np.dot(Zparent, Ytibia), axis =0)
            Xparent = preprocessing.normalize(np.cross(Ytibia, Zparent), axis =0)
            # assigning pose matrix and child orientation
            tmp_V = np.zeros((3,3))
            tmp_V = Xparent[:,0]
            tmp_V = Yparent[:,0]
            tmp_V = Zparent[:,0]
            
            jointStruct[ankle_name]['parent_orientation'] = computeXYZAngleSeq(tmp_V)
            
        # Ankle child orientation
        # ---------------------
        # Z aligned like the cilinder of the talar throclear
        # X like calcaneus, but perpendicular to Z
        # Y from cross product
        # ---------------------
        if calcn_name in JCS:
            CalcnDict = JCS[calcn_name]
            
            # take Z from ankle joint (axis of rotation)
            Zchild = TalusDict[ankle_name]['V'][:,2]
            # take X ant-post axis of the calcaneus
            Xtemp = CalcnDict[mtp_name]['V'][:,0]
            # make X and Z orthogonal
            Xchild = preprocessing.normalize(Xtemp - Zchild*np.dot(Zchild, Xtemp), axis =0)
            Ychild = preprocessing.normalize(np.cross(Zchild, Xtemp), axis =0)
            # assigning pose matrix and child orientation
            tmp_V = np.zeros((3,3))
            tmp_V[:,0] = Xchild[:,0]
            tmp_V[:,1] = Ychild[:,0]
            tmp_V[:,2] = Zchild[:,0]
            
            jointStruct[ankle_name]['child_orientation'] = computeXYZAngleSeq(tmp_V)
            
    # Ankle child orientation
    # ---------------------
    # Z is the subtalar axis of rotation
    # Y from centre of subtalar joint points to femur joint centre
    # X from cross product
    # ---------------------
    if femur_name in JCS:
        # needs to be initialized?
        FemurDict = JCS[femur_name]
        
        # take Z from ankle joint (axis of rotation)
        Zparent = TalusDict[subtalar_name]['V'][:,2]
        # take Y pointing to the knee joint centre
        Ytemp = (FemurDict[knee_name]['parent_location'] - TalusDict[subtalar_name]['parent_location'])
        Ytemp /= np.linalg.norm(FemurDict[knee_name]['parent_location'] - TalusDict[ankle_name]['parent_location'], axis=0)
        # make Y and Z orthogonal
        Yparent = preprocessing.normalize(Ytemp - Zparent*np.dot(Zparent, Ytemp), axis =0)
        Xparent = preprocessing.normalize(np.cross(Yparent, Zparent), axis =0)
        # assigning pose matrix and child orientation
        tmp_V = np.zeros((3,3))
        tmp_V[:,0] = Xparent[:,0]
        tmp_V[:,1] = Yparent[:,0]
        tmp_V[:,2] = Zparent[:,0]
        
        jointStruct[subtalar_name]['parent_orientation'] = computeXYZAngleSeq(tmp_V)
    
    return jointStruct

# -----------------------------------------------------------------------------
def inferBodySideFromAnatomicStruct(anat_struct):
    # -------------------------------------------------------------------------   
    # Infer the body side that the user wants to process based on a structure 
    # containing the anatomical objects (triangulations or joint definitions) 
    # given as input. The implemented logic is trivial: the fields are checked 
    # for standard names of bones and joints used in OpenSim models.
    # 
    # Inputs:
    # anat_struct - a dictionary containing anatomical objects, e.g. a set of 
    # bone triangulation or joint definitions.
    # 
    # Outputs:
    # guessed_side - a body side label that can be used in all other STAPLE
    # functions requiring such input.
    # -------------------------------------------------------------------------
    guessed_side = ''
    
    if isinstance(anat_struct, dict):
        fields_side = list(anat_struct.keys())
    else:
        print('inferBodySideFromAnatomicStruct.py  Input must be dictionary.')
        # logging.error('inferBodySideFromAnatomicStruct.py  Input must be dictionary.')
        return 0 
    
    # check using the body names
    body_set = ['femur', 'tibia', 'talus', 'calcn']
    guess_side_b = [fs[-1] for b in body_set for fs in fields_side if b in fs]
    
    # check using the joint names
    joint_set = ['hip', 'knee', 'ankle', 'subtalar']
    guess_side_j = [fs[-1] for j in joint_set for fs in fields_side if j in fs]
    
    # composed list
    combined_guessed = guess_side_b + guess_side_j
    
    if all(i == 'r' for i in combined_guessed):
        guessed_side = 'r'
    elif all(i == 'l' for i in combined_guessed):
        guessed_side = 'l'
    else:
        print('guessBodySideFromAnatomicStruct.py Error: it was not possible to infer the body side. Please specify it manually in this occurrance.')
        # logging.error('guessBodySideFromAnatomicStruct.py Error: it was not possible to infer the body side. Please specify it manually in this occurrance.')
        
    return guessed_side
    
        
    
    
    
    
    
    
    
    


