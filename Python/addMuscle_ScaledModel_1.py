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

#---------------------------------------------------
# folder where the various datasets (and their geometries) are located.
datasets_folder = path_file + '/bone_datasets/'

dataset_set = 'meshes'

# body sides
curr_side = 'r'

tri_folder = os.path.join(datasets_folder, dataset_set, 'stl/')

# list with the name of the bone geometries to process.
bones_list = ['pelvis', 'femur_' + curr_side, 'tibia_' + curr_side]

# create TriGeomSet dictionary for the specified geometries
triGeom_set = createTriGeomSet(bones_list, tri_folder)

# get the body side (can also be specified by user as input to funcs)
side = inferBodySideFromAnatomicStruct(triGeom_set)

# process bone geometries (compute joint parameters and identify markers)
[JCS, BL, CS] = processTriGeomBoneSet(triGeom_set, side) 


#%%
# set output folder
output_models_folder = 'opensim_models_examples/Python'

output_model_file_name = 'Carman_auto2020_HipKnee_R_ScaledFoot_2425_MUSCLES_NEW.osim'


# STAPLE generated model that we want to merge with the generic baseline
# Specific_osimModel_file = os.path.join(path_file, output_models_folder, 'Carman_auto2020_HipKnee_R_ScaledFoot.osim')
Specific_osimModel_file = os.path.join(path_file, output_models_folder, 'Carman_auto2020_HipKnee_R_ScaledFoot_2425.osim')

Specific_osimModel  = opensim.Model(Specific_osimModel_file)

Generic_osimModel_file = '/home/emi/Documents/Codigos MATLAB_PYTHON/msk-STAPLE/Python/opensim_models_examples/Python/Model/gait2368_osim4_2_MuscleModel_Millard2012_and_MethabolicProbe_Umberger2010.osim'

Generic_osimModel  = opensim.Model(Generic_osimModel_file)

Scaled_osimModel  = opensim.Model('/home/emi/Documents/DataSets/2425~V T/2425~aa~Descalzo solo/2425~aa~Standing_scaled.osim')

# Add path points to the humerus and radius. The allows the muscle to generate
# forces on these two bodies.
generic_pelvis = np.linalg.norm(np.array([0.02, 0, 0.125]) - np.array([0.02, 0, -0.125]))*0.9
generic_femur = np.linalg.norm(np.array([-0.0095, -0.41082, 0.05]))*0.9
generic_tibia = np.linalg.norm(np.array([-0.0075, -0.415, 0.05]))*0.9


specific_pelvis = np.linalg.norm(BL['pelvis']['RASIS'] - BL['pelvis']['LASIS'])*0.001
specific_femur_r = np.linalg.norm(BL['pelvis']['RASIS'] - BL['femur_r']['RKNE'])*0.001
specific_tibia_r = np.linalg.norm(BL['femur_r']['RKNE'] - BL['tibia_r']['RANK'])*0.001

generic_currentState = Generic_osimModel.initSystem()
# scaled_currentState = Scaled_osimModel.initSystem()
currentState = Scaled_osimModel.initSystem()

for muscle in Generic_osimModel.getMuscleList():
    # print(muscle.getName())
    
    muscle_name = muscle.getName()
    
    muscle_path = [p.getName() for p in muscle.getComponentsList() if muscle_name in p.getName()]
    
    
    
    if '_r' == muscle_name[-2:] and len(muscle_path) == 2:
        # extracting the muscle parameters from reference model
        Lts = muscle.getTendonSlackLength()
        Lof = muscle.getOptimalFiberLength()
        FmaxISO = muscle.getMaxIsometricForce()
        pennAngle = muscle.getPennationAngle(generic_currentState)
        
        # Create and set the parameters for the biceps muscle
        muscle1 = opensim.Millard2012EquilibriumMuscle(muscle_name,  # Muscle name
                                                   FmaxISO,  # Max isometric force
                                                   Lof,  # Optimal fiber length
                                                   Lts,  # Tendon slack length
                                                   pennAngle)  # Pennation angle
        
        for point in muscle.getGeometryPath().getPathPointSet():
            # print(point.getName())
            
            name = point.getName()
    
            loc_x = point.getLocation(currentState).get(0)
            loc_y = point.getLocation(currentState).get(1)
            loc_z = point.getLocation(currentState).get(2)
            generic_scaled_P = np.array([loc_x, loc_y, loc_z])
            generic_scaled_P *=0.9
            
            socket = point.getSocket('parent_frame').getConnecteePath()
            
            if 'pelvis' in socket:
                P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                
                # compute centroid of neibours
                P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                # P = P1
                tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                
                # # create a triang with them
                # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                
                P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                P = np.float64(np.reshape(P,(1, P.size)))
                
                origin = Specific_osimModel.getBodySet().get('pelvis')
                
            elif 'femur_r' in socket:
                
                P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                
                # compute centroid of neibours
                P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                # P = P1
                tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                
                # # create a triang with them
                # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                
                P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                P = np.float64(np.reshape(P,(1, P.size)))
                
                origin = Specific_osimModel.getBodySet().get('femur_r')
            
            elif 'tibia_r' in socket:
                
                P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                
                # compute centroid of neibours
                P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                # P = P1
                tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                
                # # create a triang with them
                # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                
                P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                P = np.float64(np.reshape(P,(1, P.size)))
                
                origin = Specific_osimModel.getBodySet().get('tibia_r')
                
            elif 'calcn_r' in socket:
                
                # loc_x = point.getLocation(scaled_currentState).get(0)
                # loc_y = point.getLocation(scaled_currentState).get(1)
                # loc_z = point.getLocation(scaled_currentState).get(2)
                # generic_P = np.array([loc_x, loc_y, loc_z])
                
                P = generic_scaled_P
                P = np.float64(np.reshape(P,(1, P.size)))
                
                origin = Specific_osimModel.getBodySet().get('calcn_r')
            
            muscle1.addNewPathPoint(name,
                                   origin,
                                   opensim.Vec3(P[0,0], P[0,1], P[0,2]))
    
        Specific_osimModel.addForce(muscle1)
        
    # 3 MUSCLE PATH
    if '_r' == muscle_name[-2:] and len(muscle_path) == 3:
        # extracting the muscle parameters from reference model
        Lts = muscle.getTendonSlackLength()
        Lof = muscle.getOptimalFiberLength()
        FmaxISO = muscle.getMaxIsometricForce()
        pennAngle = muscle.getPennationAngle(generic_currentState)
        
        # Create and set the parameters for the biceps muscle
        muscle1 = opensim.Millard2012EquilibriumMuscle(muscle_name,  # Muscle name
                                                   FmaxISO,  # Max isometric force
                                                   Lof,  # Optimal fiber length
                                                   Lts,  # Tendon slack length
                                                   pennAngle)  # Pennation angle
        
        for point in muscle.getGeometryPath().getPathPointSet():
            # print(point.getName())
            
            name = point.getName()
            
            if name[-1] == '1' or name[-1] == '3':
    
                loc_x = point.getLocation(currentState).get(0)
                loc_y = point.getLocation(currentState).get(1)
                loc_z = point.getLocation(currentState).get(2)
                generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                generic_scaled_P *=0.9
                
                socket = point.getSocket('parent_frame').getConnecteePath()
                
                if 'pelvis' in socket:
                    P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                    ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                    
                    # compute centroid of neibours
                    P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                    # P = P1
                    tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                    
                    # # create a triang with them
                    # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                    
                    P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('pelvis')
                    
                elif 'femur_r' in socket:
                    
                    if muscle_name == 'rect_fem_r':
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                    else:
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('femur_r')
                
                elif 'tibia_r' in socket:
                    P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                    ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                    
                    # compute centroid of neibours
                    P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                    # P = P1
                    tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                    
                    # # create a triang with them
                    # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                    
                    P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('tibia_r')
                
                elif 'calcn_r' in socket:
                    
                    # loc_x = point.getLocation(scaled_currentState).get(0)
                    # loc_y = point.getLocation(scaled_currentState).get(1)
                    # loc_z = point.getLocation(scaled_currentState).get(2)
                    # generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    
                    P = generic_scaled_P
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('calcn_r')
                
                if muscle_name == 'rect_fem_r' and 'tibia_r' in socket:
                    
                    # origin = Specific_osimModel.getJointSet().get('knee_r')
                    muscle1_P3 = opensim.MovingPathPoint()
                    muscle1_P3.setName(name)
                    muscle1_P3.connectSocket_parent_frame(origin)
                                        
                    # set x coord
                    coord_x = Specific_osimModel.getCoordinateSet().get('knee_angle_r')
                    func_SimmSpline = opensim.SimmSpline()
                    x0 = np.array([-2.0944, -1.99997, -1.5708, -1.45752, -1.39626, -1.0472,\
                                   -0.698132, -0.526391, -0.349066, -0.174533, 0, 0.00017453,\
                                    0.00034907, 0.0279253, 0.0872665, 0.174533, 2.0944])
                    y0 = np.array([0.0234116, 0.0237613, 0.0251141, 0.0252795, 0.0253146,\
                                   0.0249184, 0.0242373, 0.0238447, 0.0234197, 0.0227644,\
                                   0.020984, 0.0209814, 0.0209788, 0.0205225, 0.0191754, 0.0159554, -0.0673774]) - 2*CS['tibia_r']['Origin'][0]*0.001
                    for i in range(len(x0)):
                        func_SimmSpline.addPoint(x0[i], y0[i])
                                        
                    muscle1_P3.connectSocket_x_coordinate(coord_x)
                    muscle1_P3.set_x_location(func_SimmSpline)
                    
                    # set y coord
                    coord_y = Specific_osimModel.getCoordinateSet().get('knee_angle_r')
                    func_SimmSpline = opensim.SimmSpline()
                    x0 = np.array([-2.0944, -1.99997, -1.5708, -1.45752, -1.39626, -1.0472,\
                                   -0.698132, -0.526391, -0.349066, -0.174533, 0, 0.00017453,\
                                    0.00034907, 0.0279253, 0.0872665, 0.174533, 2.0944])
                    y0 = np.array([0.0234116, 0.0237613, 0.0251141, 0.0252795, 0.0253146,\
                                   0.0249184, 0.0242373, 0.0238447, 0.0234197, 0.0227644,\
                                   0.020984, 0.0209814, 0.0209788, 0.0205225, 0.0191754, 0.0159554, -0.0673774])  + CS['tibia_r']['Origin'][1]*0.001
                    for i in range(len(x0)):
                        func_SimmSpline.addPoint(x0[i], y0[i])
                                        
                    muscle1_P3.connectSocket_y_coordinate(coord_y)
                    muscle1_P3.set_y_location(func_SimmSpline)
                    
                    # set z coord
                    coord_z = Specific_osimModel.getCoordinateSet().get('knee_angle_r')
                    func_SimmSpline = opensim.SimmSpline()
                    x0 = np.array([-2.0944, 0.1745])
                    y0 = np.array([0.0014, 0.0014]) - 2*CS['tibia_r']['Origin'][2]*0.001
                    for i in range(len(x0)):
                        func_SimmSpline.addPoint(x0[i], y0[i])
                                        
                    muscle1_P3.connectSocket_z_coordinate(coord_z)
                    muscle1_P3.set_z_location(func_SimmSpline)
                    
                    muscle1.updGeometryPath().updPathPointSet().adoptAndAppend(muscle1_P3)                   
                else:
                    muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                                
            else:
                
                loc_x = point.getLocation(currentState).get(0)
                loc_y = point.getLocation(currentState).get(1)
                loc_z = point.getLocation(currentState).get(2)
                generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                if muscle_name in ['med_gas_r', 'lat_gas_r']:
                    generic_scaled_P *=0.9
                else:
                    generic_scaled_P *=0.75
                
                socket = point.getSocket('parent_frame').getConnecteePath()
                                
                if 'pelvis' in socket:
                    P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                    # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                    
                    # # compute centroid of neibours
                    # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                    # # P = P1
                    # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                    
                    # # # create a triang with them
                    # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                    
                    # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('pelvis')
                    
                elif 'femur_r' in socket:
                    
                    if muscle_name == 'rect_fem_r':
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                                                
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # P += np.dot(np.array([0.03339, -0.403, 0.0019]), CS['femur_r']['V'])*0.001
                        # P = (np.mean(tmp_P1['Points'], axis = 0) - np.dot(np.array([-0.03339, 0.403, -0.0019]), CS['femur_r']['V']))*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                    else:
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('femur_r')
                
                elif 'tibia_r' in socket:
                    
                    if muscle_name == 'rect_fem_r':
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        # print(generic_P)
                    else:
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('tibia_r')
                
                elif 'calcn_r' in socket:
                    
                    # loc_x = point.getLocation(scaled_currentState).get(0)
                    # loc_y = point.getLocation(scaled_currentState).get(1)
                    # loc_z = point.getLocation(scaled_currentState).get(2)
                    # generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    
                    P = generic_scaled_P
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('calcn_r')
                
                if muscle_name in ['bifemlh_r', 'bifemsh_r', 'tib_ant_r', 'per_tert_r']:
                    # path point attached to bone
                    muscle1.addNewPathPoint(name,
                                           origin,
                                           opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                    
                else:
                    tmp_coor = point.getSocket('coordinate').getConnecteePath()
                    coord = Specific_osimModel.getCoordinateSet().get(os.path.basename(tmp_coor))
                    
                    muscle1_P2 = opensim.ConditionalPathPoint()
                    muscle1_P2.setName(name)
                    muscle1_P2.connectSocket_parent_frame(origin)
                    muscle1_P2.setLocation(opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                    muscle1_P2.connectSocket_coordinate(coord)
                    muscle1_P2.setRangeMin(coord.getRangeMin())
                    muscle1_P2.setRangeMax(coord.getRangeMax())
                    muscle1.updGeometryPath().updPathPointSet().adoptAndAppend(muscle1_P2)
        
        Specific_osimModel.addForce(muscle1)
    
    # 4 MUSCLE PATH
    if '_r' == muscle_name[-2:] and len(muscle_path) == 4:
        # extracting the muscle parameters from reference model
        Lts = muscle.getTendonSlackLength()
        Lof = muscle.getOptimalFiberLength()
        FmaxISO = muscle.getMaxIsometricForce()
        pennAngle = muscle.getPennationAngle(generic_currentState)
        
        # Create and set the parameters for the biceps muscle
        muscle1 = opensim.Millard2012EquilibriumMuscle(muscle_name,  # Muscle name
                                                   FmaxISO,  # Max isometric force
                                                   Lof,  # Optimal fiber length
                                                   Lts,  # Tendon slack length
                                                   pennAngle)  # Pennation angle
        
        if muscle_name == 'grac_r':
            for point in muscle.getGeometryPath().getPathPointSet():
                # print(point.getName())
                
                name = point.getName()
                
                if name[-1] == '1' or name[-1] == '4':
        
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.9
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                    muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                elif name[-1] == '3':
        
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.8
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                    muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                else:
                    
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.8
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                   
                    tmp_coor = point.getSocket('coordinate').getConnecteePath()
                    coord = Specific_osimModel.getCoordinateSet().get(os.path.basename(tmp_coor))
                    
                    muscle1_P2 = opensim.ConditionalPathPoint()
                    muscle1_P2.setName(name)
                    muscle1_P2.connectSocket_parent_frame(origin)
                    muscle1_P2.setLocation(opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                    muscle1_P2.connectSocket_coordinate(coord)
                    muscle1_P2.setRangeMin(coord.getRangeMin())
                    muscle1_P2.setRangeMax(coord.getRangeMax())
                    muscle1.updGeometryPath().updPathPointSet().adoptAndAppend(muscle1_P2)
            
            Specific_osimModel.addForce(muscle1)    
                
            
        elif muscle_name == 'vas_int_r':
            
            for point in muscle.getGeometryPath().getPathPointSet():
                # print(point.getName())
                
                name = point.getName()
                
                if name[-1] == '1':
        
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.9
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                    muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                elif name[-1] == '2':
        
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.9
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                    muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                elif name[-1] == '3':
                    
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.8
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                   
                    tmp_coor = point.getSocket('coordinate').getConnecteePath()
                    coord = Specific_osimModel.getCoordinateSet().get(os.path.basename(tmp_coor))
                    
                    muscle1_P3 = opensim.ConditionalPathPoint()
                    muscle1_P3.setName(name)
                    muscle1_P3.connectSocket_parent_frame(origin)
                    muscle1_P3.setLocation(opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                    muscle1_P3.connectSocket_coordinate(coord)
                    muscle1_P3.setRangeMin(coord.getRangeMin())
                    muscle1_P3.setRangeMax(coord.getRangeMax())
                    muscle1.updGeometryPath().updPathPointSet().adoptAndAppend(muscle1_P3)
                
                elif name[-1] == '4':
                    # origin = Specific_osimModel.getJointSet().get('knee_r')
                    muscle1_P4 = opensim.MovingPathPoint()
                    muscle1_P4.setName(name)
                    muscle1_P4.connectSocket_parent_frame(origin)
                                        
                    # set x coord
                    coord_x = Specific_osimModel.getCoordinateSet().get('knee_angle_r')
                    func_SimmSpline = opensim.SimmSpline()
                    x0 = np.array([-2.0944, -1.99997, -1.5708, -1.45752, -1.39626, -1.0472,\
                                   -0.698132, -0.526391, -0.349066, -0.174533, 0, 0.00017453,\
                                    0.00034907, 0.0279253, 0.0872665, 0.174533, 2.0944])
                    y0 = np.array([0.0234116, 0.0237613, 0.0251141, 0.0252795, 0.0253146,\
                                   0.0249184, 0.0242373, 0.0238447, 0.0234197, 0.0227644,\
                                   0.020984, 0.0209814, 0.0209788, 0.0205225, 0.0191754, 0.0159554, -0.0673774]) - 2*CS['tibia_r']['Origin'][0]*0.001
                    for i in range(len(x0)):
                        func_SimmSpline.addPoint(x0[i], y0[i])
                                        
                    muscle1_P4.connectSocket_x_coordinate(coord_x)
                    muscle1_P4.set_x_location(func_SimmSpline)
                    
                    # set y coord
                    coord_y = Specific_osimModel.getCoordinateSet().get('knee_angle_r')
                    func_SimmSpline = opensim.SimmSpline()
                    x0 = np.array([-2.0944, -1.99997, -1.5708, -1.45752, -1.39626,\
                                   -1.0472, -0.698132, -0.526391, -0.349066, -0.174533,\
                                   0, 0.00017453, 0.00034907, 0.0279253, 0.0872665, 0.174533, 2.0944])
                    y0 = np.array([0.025599, 0.0259487, 0.0273124, 0.0274796, 0.0275151,\
                                   0.0271363, 0.0265737, 0.0263073, 0.0261187, 0.0260129,\
                                   0.0252923, 0.0252911, 0.0252898, 0.0250526, 0.0242191,\
                                   0.0218288, -0.0685706])  - 1.5*CS['tibia_r']['Origin'][1]*0.001
                    for i in range(len(x0)):
                        func_SimmSpline.addPoint(x0[i], y0[i])
                                        
                    muscle1_P4.connectSocket_y_coordinate(coord_y)
                    muscle1_P4.set_y_location(func_SimmSpline)
                    
                    # set z coord
                    coord_z = Specific_osimModel.getCoordinateSet().get('knee_angle_r')
                    func_SimmSpline = opensim.SimmSpline()
                    x0 = np.array([-2.0944, 0.1745])
                    y0 = np.array([0.0018, 0.0018]) - 2*CS['tibia_r']['Origin'][2]*0.001
                    for i in range(len(x0)):
                        func_SimmSpline.addPoint(x0[i], y0[i])
                                        
                    muscle1_P4.connectSocket_z_coordinate(coord_z)
                    muscle1_P4.set_z_location(func_SimmSpline)
                    
                    muscle1.updGeometryPath().updPathPointSet().adoptAndAppend(muscle1_P4)
            
            Specific_osimModel.addForce(muscle1)
            
        else:
            for point in muscle.getGeometryPath().getPathPointSet():
                # print(point.getName())
                
                name = point.getName()
        
                loc_x = point.getLocation(currentState).get(0)
                loc_y = point.getLocation(currentState).get(1)
                loc_z = point.getLocation(currentState).get(2)
                generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                generic_scaled_P *=0.8
                
                socket = point.getSocket('parent_frame').getConnecteePath()
                
                if 'pelvis' in socket:
                    P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                    # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                    
                    # # compute centroid of neibours
                    # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                    # # P = P1
                    # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                    
                    # # # create a triang with them
                    # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                    
                    # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('pelvis')
                    
                elif 'femur_r' in socket:
                    
                    P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                    # ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                    
                    # # compute centroid of neibours
                    # P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                    # # P = P1
                    # tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                    
                    # # # create a triang with them
                    # # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                    
                    # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('femur_r')
                
                elif 'tibia_r' in socket:
                    
                    P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                    # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                    
                    # # compute centroid of neibours
                    # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                    # # P = P1
                    # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                    
                    # # # create a triang with them
                    # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                    
                    # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('tibia_r')
                
                muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
        
            Specific_osimModel.addForce(muscle1)
        
    
    # 5 MUSCLE PATH
    if '_r' == muscle_name[-2:] and len(muscle_path) == 5:
        # extracting the muscle parameters from reference model
        Lts = muscle.getTendonSlackLength()
        Lof = muscle.getOptimalFiberLength()
        FmaxISO = muscle.getMaxIsometricForce()
        pennAngle = muscle.getPennationAngle(generic_currentState)
        
        # Create and set the parameters for the biceps muscle
        muscle1 = opensim.Millard2012EquilibriumMuscle(muscle_name,  # Muscle name
                                                   FmaxISO,  # Max isometric force
                                                   Lof,  # Optimal fiber length
                                                   Lts,  # Tendon slack length
                                                   pennAngle)  # Pennation angle
        
        if muscle_name == 'semiten_r':
            
            for point in muscle.getGeometryPath().getPathPointSet():
                # print(point.getName())
                
                name = point.getName()
                
                if name[-1] == '1' or name[-1] == '5':
        
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.9
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                    
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                    muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                elif name[-1] == '3' or name[-1] == '4':
        
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.8
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                    
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                    muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                else:
                    
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.8
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                                        
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                   
                    tmp_coor = point.getSocket('coordinate').getConnecteePath()
                    coord = Specific_osimModel.getCoordinateSet().get(os.path.basename(tmp_coor))
                    
                    muscle1_P2 = opensim.ConditionalPathPoint()
                    muscle1_P2.setName(name)
                    muscle1_P2.connectSocket_parent_frame(origin)
                    muscle1_P2.setLocation(opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                    muscle1_P2.connectSocket_coordinate(coord)
                    muscle1_P2.setRangeMin(coord.getRangeMin())
                    muscle1_P2.setRangeMax(coord.getRangeMax())
                    muscle1.updGeometryPath().updPathPointSet().adoptAndAppend(muscle1_P2)
            
            Specific_osimModel.addForce(muscle1) 
        
        elif muscle_name == 'psoas_r' or muscle_name == 'iliacus_r':
            for point in muscle.getGeometryPath().getPathPointSet():
                # print(point.getName())
                
                name = point.getName()
                
                if name[-1] == '1' or name[-1] == '5':
        
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.9
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                    muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                    
                elif name[-1] == '2' or name[-1] == '4':
        
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.8
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                    muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                else:
                    
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.8
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                   
                    tmp_coor = point.getSocket('coordinate').getConnecteePath()
                    coord = Specific_osimModel.getCoordinateSet().get(os.path.basename(tmp_coor))
                    
                    muscle1_P2 = opensim.ConditionalPathPoint()
                    muscle1_P2.setName(name)
                    muscle1_P2.connectSocket_parent_frame(origin)
                    muscle1_P2.setLocation(opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                    muscle1_P2.connectSocket_coordinate(coord)
                    muscle1_P2.setRangeMin(coord.getRangeMin())
                    muscle1_P2.setRangeMax(coord.getRangeMax())
                    muscle1.updGeometryPath().updPathPointSet().adoptAndAppend(muscle1_P2)
            
            Specific_osimModel.addForce(muscle1)
        
        elif muscle_name == 'vas_med_r' or muscle_name == 'vas_lat_r':
            
            for point in muscle.getGeometryPath().getPathPointSet():
                # print(point.getName())
                
                name = point.getName()
                
                if name[-1] == '1':
        
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.9
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                    muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                elif name[-1] == '2':
        
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.8
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                    muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                    
                elif name[-1] == '3' or name[-1] == '4':
                    
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.7
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                   
                    tmp_coor = point.getSocket('coordinate').getConnecteePath()
                    coord = Specific_osimModel.getCoordinateSet().get(os.path.basename(tmp_coor))
                    
                    muscle1_P3 = opensim.ConditionalPathPoint()
                    muscle1_P3.setName(name)
                    muscle1_P3.connectSocket_parent_frame(origin)
                    muscle1_P3.setLocation(opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                    muscle1_P3.connectSocket_coordinate(coord)
                    muscle1_P3.setRangeMin(coord.getRangeMin())
                    muscle1_P3.setRangeMax(coord.getRangeMax())
                    muscle1.updGeometryPath().updPathPointSet().adoptAndAppend(muscle1_P3)
                
                elif name[-1] == '5':
                    # origin = Specific_osimModel.getJointSet().get('knee_r')
                    muscle1_P4 = opensim.MovingPathPoint()
                    muscle1_P4.setName(name)
                    muscle1_P4.connectSocket_parent_frame(origin)
                                        
                    # set x coord
                    coord_x = Specific_osimModel.getCoordinateSet().get('knee_angle_r')
                    func_SimmSpline = opensim.SimmSpline()
                    x0 = np.array([-2.0944, -1.99997, -1.5708, -1.45752, -1.39626, -1.0472,\
                                   -0.698132, -0.526391, -0.349066, -0.174533, 0, 0.00017453,\
                                    0.00034907, 0.0279253, 0.0872665, 0.174533, 2.0944])
                    y0 = np.array([0.0234116, 0.0237613, 0.0251141, 0.0252795, 0.0253146,\
                                   0.0249184, 0.0242373, 0.0238447, 0.0234197, 0.0227644,\
                                   0.020984, 0.0209814, 0.0209788, 0.0205225, 0.0191754, 0.0159554, -0.0673774]) - 2*CS['tibia_r']['Origin'][0]*0.001
                    for i in range(len(x0)):
                        func_SimmSpline.addPoint(x0[i], y0[i])
                                        
                    muscle1_P4.connectSocket_x_coordinate(coord_x)
                    muscle1_P4.set_x_location(func_SimmSpline)
                    
                    # set y coord
                    coord_y = Specific_osimModel.getCoordinateSet().get('knee_angle_r')
                    func_SimmSpline = opensim.SimmSpline()
                    x0 = np.array([-2.0944, -1.99997, -1.5708, -1.45752, -1.39626,\
                                   -1.0472, -0.698132, -0.526391, -0.349066, -0.174533,\
                                   0, 0.00017453, 0.00034907, 0.0279253, 0.0872665, 0.174533, 2.0944])
                    y0 = np.array([0.025599, 0.0259487, 0.0273124, 0.0274796, 0.0275151,\
                                   0.0271363, 0.0265737, 0.0263073, 0.0261187, 0.0260129,\
                                   0.0252923, 0.0252911, 0.0252898, 0.0250526, 0.0242191,\
                                   0.0218288, -0.0685706])  - 1.5*CS['tibia_r']['Origin'][1]*0.001
                    for i in range(len(x0)):
                        func_SimmSpline.addPoint(x0[i], y0[i])
                                        
                    muscle1_P4.connectSocket_y_coordinate(coord_y)
                    muscle1_P4.set_y_location(func_SimmSpline)
                    
                    # set z coord
                    coord_z = Specific_osimModel.getCoordinateSet().get('knee_angle_r')
                    func_SimmSpline = opensim.SimmSpline()
                    x0 = np.array([-2.0944, 0.1745])
                    y0 = np.array([0.0018, 0.0018]) - 1*CS['tibia_r']['Origin'][2]*0.001
                    for i in range(len(x0)):
                        func_SimmSpline.addPoint(x0[i], y0[i])
                                        
                    muscle1_P4.connectSocket_z_coordinate(coord_z)
                    muscle1_P4.set_z_location(func_SimmSpline)
                    
                    muscle1.updGeometryPath().updPathPointSet().adoptAndAppend(muscle1_P4)
            
            Specific_osimModel.addForce(muscle1)    
        
        else:
            for point in muscle.getGeometryPath().getPathPointSet():
                # print(point.getName())
                
                name = point.getName()
                
                if name[-1] == '1' or name[-1] == '5':
        
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.9
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # compute centroid of neibours
                        P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # P = P1
                        tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # create a triang with them
                        # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                    muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
                    
                else:
                        
                    loc_x = point.getLocation(currentState).get(0)
                    loc_y = point.getLocation(currentState).get(1)
                    loc_z = point.getLocation(currentState).get(2)
                    generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    generic_scaled_P *=0.8
                    
                    socket = point.getSocket('parent_frame').getConnecteePath()
                    
                    if 'pelvis' in socket:
                        P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('pelvis')
                        
                    elif 'femur_r' in socket:
                        
                        P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('femur_r')
                    
                    elif 'tibia_r' in socket:
                        
                        P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                        # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                        
                        # # compute centroid of neibours
                        # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                        # # P = P1
                        # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                        
                        # # # create a triang with them
                        # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                        
                        # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                        P = np.float64(np.reshape(P,(1, P.size)))
                        
                        origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                    muscle1.addNewPathPoint(name,
                                           origin,
                                           opensim.Vec3(P[0,0], P[0,1], P[0,2]))
        
            Specific_osimModel.addForce(muscle1)
    
    # 7 MUSCLE PATH
    if '_r' == muscle_name[-2:] and len(muscle_path) == 7:
        # extracting the muscle parameters from reference model
        Lts = muscle.getTendonSlackLength()
        Lof = muscle.getOptimalFiberLength()
        FmaxISO = muscle.getMaxIsometricForce()
        pennAngle = muscle.getPennationAngle(generic_currentState)
        
        # Create and set the parameters for the biceps muscle
        muscle1 = opensim.Millard2012EquilibriumMuscle(muscle_name,  # Muscle name
                                                   FmaxISO,  # Max isometric force
                                                   Lof,  # Optimal fiber length
                                                   Lts,  # Tendon slack length
                                                   pennAngle)  # Pennation angle
        
        for point in muscle.getGeometryPath().getPathPointSet():
            # print(point.getName())
            
            name = point.getName()
            
            if name[-1] == '1' or name[-1] == '7':
    
                loc_x = point.getLocation(currentState).get(0)
                loc_y = point.getLocation(currentState).get(1)
                loc_z = point.getLocation(currentState).get(2)
                generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                generic_scaled_P *=0.9
                
                socket = point.getSocket('parent_frame').getConnecteePath()
                
                if 'pelvis' in socket:
                    P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                    ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                    
                    # compute centroid of neibours
                    P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                    # P = P1
                    tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                    
                    # # create a triang with them
                    # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                    
                    P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('pelvis')
                    
                elif 'femur_r' in socket:
                    
                    P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                    ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                    
                    # compute centroid of neibours
                    P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                    # P = P1
                    tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                    
                    # # create a triang with them
                    # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                    
                    P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('femur_r')
                
                elif 'tibia_r' in socket:
                    
                    P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                    ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                    
                    # compute centroid of neibours
                    P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                    # P = P1
                    tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                    
                    # # create a triang with them
                    # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                    
                    P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('tibia_r')
                
                muscle1.addNewPathPoint(name,
                                   origin,
                                   opensim.Vec3(P[0,0], P[0,1], P[0,2]))
            
            else:
                
                loc_x = point.getLocation(currentState).get(0)
                loc_y = point.getLocation(currentState).get(1)
                loc_z = point.getLocation(currentState).get(2)
                generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                generic_scaled_P *=0.8
                
                socket = point.getSocket('parent_frame').getConnecteePath()
                
                if 'pelvis' in socket:
                    P = generic_scaled_P + CS['pelvis']['Origin'].T*0.001
                    # ind_P = np.argmin(np.linalg.norm((triGeom_set['pelvis']['Points'] - P*1000), axis = 1))
                    
                    # # compute centroid of neibours
                    # P1 = np.where(triGeom_set['pelvis']['ConnectivityList'] == ind_P)[0]
                    # # P = P1
                    # tmp_P1 = TriReduceMesh(triGeom_set['pelvis'], P1)
                    
                    # # # create a triang with them
                    # # tmp_P1 = TriDilateMesh(triGeom_set['pelvis'], tmp_P1, 1)
                    
                    # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('pelvis')
                    
                elif 'femur_r' in socket:
                    
                    P = generic_scaled_P + CS['femur_r']['Origin'].T*0.001
                    # ind_P = np.argmin(np.linalg.norm((triGeom_set['femur_r']['Points'] - P*1000), axis = 1))
                    
                    # # compute centroid of neibours
                    # P1 = np.where(triGeom_set['femur_r']['ConnectivityList'] == ind_P)[0]
                    # # P = P1
                    # tmp_P1 = TriReduceMesh(triGeom_set['femur_r'], P1)
                    
                    # # # create a triang with them
                    # # tmp_P1 = TriDilateMesh(triGeom_set['femur_r'], tmp_P1, 1)
                    
                    # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('femur_r')
                
                elif 'tibia_r' in socket:
                    
                    P = generic_scaled_P + CS['tibia_r']['Origin'].T*0.001
                    # ind_P = np.argmin(np.linalg.norm((triGeom_set['tibia_r']['Points'] - P*1000), axis = 1))
                    
                    # # compute centroid of neibours
                    # P1 = np.where(triGeom_set['tibia_r']['ConnectivityList'] == ind_P)[0]
                    # # P = P1
                    # tmp_P1 = TriReduceMesh(triGeom_set['tibia_r'], P1)
                    
                    # # # create a triang with them
                    # # tmp_P1 = TriDilateMesh(triGeom_set['tibia_r'], tmp_P1, 1)
                    
                    # P = np.mean(tmp_P1['Points'], axis = 0)*0.001
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('tibia_r')
                    
                elif 'calcn_r' in socket:
                    
                    # loc_x = point.getLocation(scaled_currentState).get(0)
                    # loc_y = point.getLocation(scaled_currentState).get(1)
                    # loc_z = point.getLocation(scaled_currentState).get(2)
                    # generic_scaled_P = np.array([loc_x, loc_y, loc_z])
                    
                    P = generic_scaled_P
                    P = np.float64(np.reshape(P,(1, P.size)))
                    
                    origin = Specific_osimModel.getBodySet().get('calcn_r')
                
                muscle1.addNewPathPoint(name,
                                       origin,
                                       opensim.Vec3(P[0,0], P[0,1], P[0,2]))
    
        Specific_osimModel.addForce(muscle1)


Specific_osimModel.initSystem()


# finalize connections
Specific_osimModel.finalizeConnections

# print
Specific_osimModel.printToXML(os.path.join(output_models_folder, output_model_file_name))

