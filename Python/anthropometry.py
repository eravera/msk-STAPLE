#!/usr/bin/env python3
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
import opensim

# from geometry import bodySide2Sign, \
#                       inferBodySideFromAnatomicStruct

# from Public_functions import *
# from GIBOC_core import *
# from algorithms import *
from geometry import *
# from anthropometry import *
# from opensim_tools import *
# -----------------------------------------------------------------------------
def gait2392MassProps(segment_name = ''):
    # -------------------------------------------------------------------------
    # Return the mass and inertia of the body segments of model gait2392, 
    # which for the lower limb are the same as Rajagopal's model. The script 
    # saves from reading the OpenSim model directly. Note that side does not 
    # need to be specified as mass properties are the same on both.
    # 
    # Inputs:
    # segment_name - a string with the name of an OpenSim body included in
    # the gait2392 model. i.e. 'pelvis', 'femur', 'tibia', 'talus',
    # 'calcn', 'toes', 'patella' and 'torso'.
    # 
    # Outputs:
    # MP - a dictionary with keys 'mass', 'mass_center', 'inertia_xx',
    # 'inertia_yy', 'inertia_zz'.
    # 
    # -------------------------------------------------------------------------
    # 
    # Note that the Rajagopal model has the same lower limb inertial 
    # properties, which is why I added the patella from that model.
    # torso differs because that model has arms.
    
    MP = {}
    
    if segment_name != 'pelvis' or segment_name != 'torso' or segment_name != 'full_body':
        segment_name = segment_name[:-2]
        
    if segment_name == 'full_body':
        # This is: pelvis + 2*(fem+tibia+talus+foot+toes+patella)+torso 
        MP['mass'] = 11.777 + 2*(9.3014+3.7075+0.1+1.25+0.2166+0.0862)+34.2366
        
    elif segment_name == 'pelvis':
        MP['mass'] = 11.777
        MP['mass_center'] = np.array([-0.0707, 0, 0])
        MP['inertia_xx'] = 0.1028
        MP['inertia_yy'] = 0.0871
        MP['inertia_zz'] = 0.0579
        
    elif segment_name == 'femur':
        MP['mass'] = 9.3014
        MP['mass_center'] = np.array([0, -0.17, 0])
        MP['inertia_xx'] = 0.1339
        MP['inertia_yy'] = 0.0351
        MP['inertia_zz'] = 0.1412
        
    elif segment_name == 'tibia':
        MP['mass'] = 3.7075
        MP['mass_center'] = np.array([0, -0.1867, 0])
        MP['inertia_xx'] = 0.0504
        MP['inertia_yy'] = 0.0051
        MP['inertia_zz'] = 0.0511
    
    elif segment_name == 'talus':
        MP['mass'] = 0.1
        MP['mass_center'] = np.array([0, 0, 0])
        MP['inertia_xx'] = 0.001
        MP['inertia_yy'] = 0.001
        MP['inertia_zz'] = 0.001
    
    elif segment_name == 'calcn':
        MP['mass'] = 1.25
        MP['mass_center'] = np.array([0.1, 0.03, 0])
        MP['inertia_xx'] = 0.0014
        MP['inertia_yy'] = 0.0039
        MP['inertia_zz'] = 0.0041
    
    elif segment_name == 'toes':
        MP['mass'] = 0.2166
        MP['mass_center'] = np.array([0.0346, 0.006, -0.0175])
        MP['inertia_xx'] = 0.0001
        MP['inertia_yy'] = 0.0002
        MP['inertia_zz'] = 0.0001
        
    elif segment_name == 'torso':
        # different from Rajagopal that has arms
        MP['mass'] = 34.2366
        MP['mass_center'] = np.array([-0.03, 0.32, 0])
        MP['inertia_xx'] = 1.4745
        MP['inertia_yy'] = 0.7555
        MP['inertia_zz'] = 1.4314
    
    elif segment_name == 'patella':
        MP['mass'] = 0.0862
        MP['mass_center'] = np.array([0.0018, 0.0264, 0])
        MP['inertia_xx'] = 2.87e-006
        MP['inertia_yy'] = 1.311e-005
        MP['inertia_zz'] = 1.311e-005
        
    else:
        # loggin.error('Please specify a segment name among those included in the gait2392 model')
        print('Please specify a segment name among those included in the gait2392 model')
    
    return MP

# -----------------------------------------------------------------------------
def mapGait2392MassPropToModel(osimModel):
    # -------------------------------------------------------------------------
    # Map the mass properties of the gait2392 model to the equivalent segments 
    # of the model specified as input.
    # 
    # Inputs:
    # osimModel - the OpenSim model for which the mass properties of the 
    # segments will be updated using the gait2392 values.
    # 
    # Outputs:
    # osimModel - the OpenSim model with the updated inertial properties.
    # 
    # -------------------------------------------------------------------------
    
    # loop through the bodies of the model
    N_bodies = osimModel.getBodySet.getSize()
    
    for n_b in range(N_bodies):
        curr_body = osimModel.getBodySet.get(n_b)
        curr_body_name = str(curr_body.getName())
        
        # retried mass properties of gait2392
        massProp = gait2392MassProps(curr_body_name)
        
        # retrieve segment to update
        curr_body = osimModel.getBodySet.get(curr_body_name)
        
        # assign mass
        curr_body.setMass(massProp.mass)
        
        # build a matrix of inertia with the gait2392 values
        xx = massProp['inertia_xx']
        yy = massProp['inertia_yy']
        zz = massProp['inertia_zz']
        xy = 0.0
        xz = 0.0
        yz = 0.0
        upd_inertia = opensim.Inertia(xx, yy, zz, xy, xz, yz)
        
        # set inertia
        curr_body.setInertia(upd_inertia)
        print('Mapped on body: ' + curr_body_name)
        
    return osimModel

# -----------------------------------------------------------------------------
def scaleMassProps(osimModel, coeff):
    # -------------------------------------------------------------------------
    # Scale mass and inertia of the bodies of an OpenSim model assuming that 
    # the geometry stays constant and only the mass changes proportionally to 
    # a coefficient assigned in input.
    # 
    # Inputs:
    # osimModel - the OpenSim model for which the mass properties of the
    # segments will be scaled.
    # 
    # coeff - ratio of mass new_mass/curr_model_mass. This is used to scale
    # the inertial properties of the gait2392 model to the mass of a
    # specific individual.
    # 
    # Outputs:
    # osimModel - the OpenSim model with the scaled inertial properties.
    # 
    # -------------------------------------------------------------------------
    
    # get bodyset
    subjspec_bodyset = osimModel.getBodySet
    for n_b in range(subjspec_bodyset.getSize()):
        
        curr_body = subjspec_bodyset.get(n_b)
        
        # updating the mass
        curr_body.setMass(coeff*curr_body.getMass)
        
        # updating the inertia matrix for the change in mass
        m = curr_body.get_inertia()
    
        # components of inertia
        xx = m.get(0)*coeff
        yy = m.get(1)*coeff
        zz = m.get(2)*coeff
        xy = m.get(3)*coeff
        xz = m.get(4)*coeff
        yz = m.get(5)*coeff
        upd_inertia = opensim.Inertia(xx, yy, zz, xy, xz, yz)
        
        # updating Inertia
        curr_body.setInertia(upd_inertia)
        
        del m
        
    return osimModel

# -----------------------------------------------------------------------------
def assignMassPropsToSegments(osimModel, JCS = {}, subj_mass = 0, side_raw = ''):
    # -------------------------------------------------------------------------
    # Assign mass properties to the segments of an OpenSim model created 
    # automatically. Mass and inertia are scaled from the values used in the 
    # gait2392 model.
    # NOTE: this function will be rewritten (prototype).
    # 
    # Inputs:
    # osimModel - an OpenSim model generated automatically for which the mass
    # properties needs to be personalised.
    # 
    # JCS - a dictionary including the joint coordinate system computed from
    # the bone geometries. Required for computing the segment lengths and
    # identifying the COM positions.
    # 
    # subj_mass - the total mass of the individual of which we are building a
    # model, in Kg. Required for scaling the mass properties of the
    # generic gati2392 model.
    # 
    # side_raw - generic string identifying a body side. 'right', 'r', 'left'
    # and 'l' are accepted inputs, both lower and upper cases.
    # 
    # Outputs:
    # osimModel - the OpenSim model with the personalised mass properties.
    # 
    # -------------------------------------------------------------------------
    
    if side_raw == '':
        side = inferBodySideFromAnatomicStruct(JCS)
    else:
        # get sign correspondent to body side
        _, side = bodySide2Sign(side_raw)

    femur_name = 'femur_' + side
    tibia_name = 'tibia_' + side
    talus_name = 'talus_' + side
    calcn_name = 'calcn_' + side
    hip_name = 'hip_' + side
    knee_name = 'knee_' + side
    ankle_name = 'ankle_' + side
    toes_name = 'mtp_' + side

    print('------------------------')
    print('  UPDATING MASS PROPS   ')
    print('------------------------')

    # compute lengths of segments from the bones and COM positions using 
    # coefficients from Winter 2015 (book)

    # Keep in mind that all Origin fields have [3x1] dimensions
    print('Updating centre of mass position (Winter 2015)...')
    if femur_name in JCS:
        # compute thigh length
        thigh_axis = JCS[femur_name][hip_name]['Origin'] - JCS[femur_name][knee_name]['Origin']
        thigh_L = np.linalg.norm(thigh_axis)
        thigh_COM = thigh_L*0.567 * (thigh_axis/thigh_L) + JCS[femur_name][knee_name]['Origin']
        # assign  thigh COM
        osimModel.getBodySet().get(femur_name).setMassCenter(opensim.ArrayDouble.createVec3(thigh_COM/1000))
        
        # shank
        if talus_name in JCS:
            # compute thigh length
            shank_axis = JCS[talus_name][knee_name]['Origin'] - JCS[talus_name][ankle_name]['Origin']
            shank_L = np.linalg.norm(shank_axis)
            shank_COM = shank_L*0.567 * (shank_axis/shank_L) + JCS[talus_name][ankle_name]['Origin']
            # assign  thigh COM
            osimModel.getBodySet().get(tibia_name).setMassCenter(opensim.ArrayDouble.createVec3(shank_COM/1000))
            
            # foot
            if calcn_name in JCS:
                # compute thigh length
                foot_axis = JCS[talus_name][knee_name]['Origin'] - JCS[calcn_name][toes_name]['Origin']
                foot_L = np.linalg.norm(foot_axis)
                calcn_COM = shank_L*0.5 * (foot_axis/foot_L) + JCS[calcn_name][toes_name]['Origin']
                # assign  thigh COM
                osimModel.getBodySet().get(calcn_name).setMassCenter(opensim.ArrayDouble.createVec3(calcn_COM/1000))
                
    # -----------------------------------------------------------------------------
    # map gait2392 properties to the model segments as an initial value
    print('Mapping segment masses and inertias from gait2392 model.')
    osimModel = mapGait2392MassPropToModel(osimModel)

    # opensim model total mass (consistent in gait2392 and Rajagopal)
    MP = gait2392MassProps('full_body')
    gait2392_tot_mass = MP['mass']

    # calculate mass ratio of subject mass and gait2392 mass
    coeff = subj_mass/gait2392_tot_mass

    # scale gait2392 mass properties to the individual subject
    print('Scaling inertial properties to assigned body weight...')
    scaleMassProps(osimModel, coeff)

    print('Done.')














