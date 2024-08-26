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
from stl import mesh
import fast_simplification
import opensim

from Public_functions import load_mesh

from GIBOC_core import computeMassProperties_Mirtich1996

# -----------------------------------------------------------------------------
def writeModelGeometriesFolder(aTriGeomBoneSet, aGeomFolder = '.', aFileFormat = 'obj', coeffFaceReduc = 0.3):
    # -------------------------------------------------------------------------
    # Write bone geometry for an automated model in the specified geometry 
    # folder using a user-defined file format.
    #
    # Inputs:
    #    aTriGeomBoneSet - a set of triangulation dict generated using the
    #        createTriGeomSet function.
    #
    #    aGeomFolder - the folder where to save the geometries of the automatic
    #        OpenSim model.
    #
    #    aFileFormat - the format to use when writing the bone geometry files.
    #       Currently 'stl' files and Waterfront 'obj' files are supported.
    #       Note that both formats are ASCII, as required by the OpenSim
    #       visualiser.
    #
    #   coeffFaceReduc - number between 0 and 1 indicating the ratio of faces
    #       that the final geometry will have. Default value 0.3, meaning that
    #       the geometry files will have 30% of the faces of the original bone
    #       geometries. This is required for faster visualization in OpenSim.
    # 
    # Outputs:
    #   none - the bone geometries are saved in the specified folder using the
    #       specified file format.
    # -------------------------------------------------------------------------
        
    # create geometry file if not existing
    dst = Path(aGeomFolder)
    dst.mkdir(parents=True, exist_ok=True)
    dst = str(dst)
        
    # ensure lower case fileformat
    aFileFormat = aFileFormat.lower()
    
    # how many bones to print?
    bone_names = list(aTriGeomBoneSet.keys())
    
    print('-------------------------------------')
    print('Writing geometries for visualization ')
    print('-------------------------------------')
    
    for curr_bone_name in bone_names:
        
        curr_tri = aTriGeomBoneSet[curr_bone_name]
    
        # reduce number of faces in geometry
        points_out, faces_out = fast_simplification.simplify(curr_tri['Points'], curr_tri['Vertex'], 1-coeffFaceReduc) 
        
        if aFileFormat == 'obj':
            
            logging.exception('writeModelGeometriesFolder Error: Please write model geometries as .STL files (preferred).')
            
        elif aFileFormat == 'stl':
            
            new_mesh = mesh.Mesh(np.zeros(faces_out.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces_out):
                for j in range(3):
                    new_mesh.vectors[i][j] = points_out[f[j],:]

            # Write the mesh to file 
            new_mesh.save(os.path.join(aGeomFolder, curr_bone_name + '.stl'))
            
        else:
            
            logging.exception('writeModelGeometriesFolder Error: Please specify a file format to write the model geometries between ''stl'' and ''obj''.')
    
    # inform the user
    print('Stored ' + aFileFormat + ' files in folder ' + aGeomFolder)
    
    return 0

# -----------------------------------------------------------------------------
def computeXYZAngleSeq(aRotMat):
    # -------------------------------------------------------------------------
    # Convert a rotation matrix in the orientation vector used in OpenSim 
    # (X-Y-Z axes rotation order).
    # 
    # Inputs:
    # aRotMat - a rotation matrix, normally obtained writing as columns the
    # axes of the body reference system, expressed in global reference
    # system.
    # 
    # Outputs:
    # orientation - the sequence of angles used in OpenSim to define the
    # joint orientation. Sequence of rotation is X-Y-Z.
    # -------------------------------------------------------------------------    
    orientation = np.zeros((1,3))
    
    # fixed body sequence of angles from rot mat usable for orientation in OpenSim
    beta = np.arctan2(aRotMat[0,2], np.sqrt(aRotMat[0,0]**2 + aRotMat[0,1]**2))
    alpha = np.arctan2(-aRotMat[1,2]/np.cos(beta), aRotMat[2,2]/np.cos(beta))
    gamma = np.arctan2(-aRotMat[0,1]/np.cos(beta), aRotMat[0,0]/np.cos(beta))
    
    # build a vector
    orientation[0,0] = beta
    orientation[0,1] = alpha
    orientation[0,2] = gamma
    
    return orientation




#%% OpenSim functions

# -----------------------------------------------------------------------------
def initializeOpenSimModel(aModelNameString):
    # -------------------------------------------------------------------------
    # Create barebone of OpenSim model adding name, gravity and credits for the
    # automatic models.
    # 
    # Inputs:
    # aModelNameString - a string that will be set as name of the automatic model.
    # 
    # Outputs:
    # osimModel - an OpenSim model to use as basis for the automatic modelling.
    # It has only a name, includes gravity and has credits.
    # -------------------------------------------------------------------------
    
    print('-------------------------------------')
    print('Initializing automatic OpenSim model:')
    
    # create the model
    osimModel = opensim.Model()
    
    # set gravity
    osimModel.setGravity(opensim.Vec3(0, -9.8081, 0))
    
    # set model name
    osimModel.setName(aModelNameString)
    
    # set credits
    osimModel.set_credits('Luca Modenese, Jean-Baptiste Renault 2020. Model created using the STAPLE (Shared Tools for Automatic Personalised Lower Extremity) modelling toolbox. GitHub page: https://github.com/modenaxe/msk-STAPLE.')
    
    return osimModel

# -----------------------------------------------------------------------------
def addBodiesFromTriGeomBoneSet(osimModel = '', geom_set = {}, vis_geom_folder = '', vis_geom_format = 'obj', body_density = 1420*0.001**3, in_mm = 1):
    # -------------------------------------------------------------------------
    # Create a body for each triangulation object in the provided geom_set 
    # structure. These bodies are added to the specified OpenSim model. A 
    # density for computing the mass properties and a details of the 
    # visualization folder can also be provided.
    # NOTE: the added bodies are not yet connected by appropriate joints, so
    # unless this function is used in a workflow including 
    # createLowerLimbJoints or joint definition, the resulting OpenSim model
    # will consist of bodies connected to ground.
    # 
    # Inputs:
    # osimModel - an OpenSim model to which the bodies created from
    # triangulation objects will be added.
    # 
    # geom_set - a set of Dict triangulation objects, normally created
    # using the function createTriGeomSet. See that function for more details.
    # 
    # vis_geom_folder - the folder where the geometry files used to
    # visualised the OpenSim model will be stored.
    # 
    # vis_geom_format - the format used to write the geometry files employed
    # by the OpenSim model for visualization.
    # 
    # body_density - (optional) the density assigned to the triangulation
    # objects when computing the mass properties. Default value is 1420
    # Kg/m^3, which is the density assigned to bone in Dumas et al. 
    # IEEE Transactions on Biomedical engineering (2005). In the
    # generation of automatic lower extremity models this value is
    # overwritten by mass properties estimated through regression
    # equations. The purpose of computing them is to provide a reasonable
    # first estimate
    # 
    # in_mm - (optional) indicates if the provided geometries are given in mm
    # (value: 1) or m (value: 0). Please note that all tests and analyses
    # done so far were performed on geometries expressed in mm, so this
    # option is more a placeholder for future adjustments.
    # 
    # Outputs:
    # osimModel - the OpenSim model provided as input updated to include the
    # bodies defined from the triangulation objects.
    # -------------------------------------------------------------------------
    
    if in_mm == 1:
        dim_fact = 0.001
    else:
        dim_fact = 1
    
    # bone density by default (Dumas 2005)
    body_density = 1420*dim_fact**3
    
    # add the individual bodies to the model
    print('-------------------------------------')
    print('Adding ' + str(len(geom_set)) + ' bodies to the OpenSim model')
    
    for cur_body_name, cur_geom in geom_set.items():
        # geometry file used for visualisation
        cur_vis_geom_file = os.path.join(vis_geom_folder, cur_body_name + '.' + vis_geom_format)
        
        if cur_body_name == 'pelvis_no_sacrum':
            cur_body_name == 'pelvis'
        
        print('     ' + 'i) ' + cur_body_name)
        
        # creating the body and adding it to the OpenSim model
        # addBodyFromTriGeomObj(osimModel, cur_geom, cur_body_name, cur_vis_geom_file, body_density, in_mm)
        print('      ADDED')
        print('      -----')

    return osimModel

# -----------------------------------------------------------------------------
def addBodyFromTriGeomObj(osimModel = '', triGeom = {}, body_name = '', vis_mesh_file = '', body_density = 1420*0.001**3, in_mm = 1):
    # -------------------------------------------------------------------------
    # Create an OpenSim Body from a MATLAB triangulation object. 
    # Requires a name.
    # 
    # Inputs:
    # osimModel - an OpenSim model to which the bodies created from
    # triangulation objects will be added.
    # 
    # triGeom - a set of Dict triangulation objects.
    # 
    # body_name - a string indicating the name to assign to the OpenSim body.
    # 
    # vis_mesh_file - the path of the visualisation file that will be assigned 
    # to the OpenSim body. For automatic models, this will be an object file 
    # generated by writeModelGeometriesFolder.
    # 
    # body_density - (optional) the density assigned to the triangulation
    # objects when computing the mass properties. Default value is 1420
    # Kg/m^3, which is the density assigned to bone in Dumas et al. 
    # IEEE Transactions on Biomedical engineering (2005). In the
    # generation of automatic lower extremity models this value is
    # overwritten by mass properties estimated through regression
    # equations. The purpose of computing them is to provide a reasonable
    # first estimate
    # 
    # in_mm - (optional) indicates if the provided geometries are given in mm
    # (value: 1) or m (value: 0). Please note that all tests and analyses
    # done so far were performed on geometries expressed in mm, so this
    # option is more a placeholder for future adjustments.
    # 
    # Outputs:
    # osimModel - the OpenSim model provided as input updated to include the
    # bodies defined from the triangulation objects.
    # -------------------------------------------------------------------------
    
    if in_mm == 1:
        dim_fact = 0.001
    else:
        dim_fact = 1
    
    # bone density by default (Dumas 2005)
    body_density = 1420*dim_fact**3
    
    # compute mass properties
    # NOTE: when creating automated models this initial mass properties are
    # overwritten by mapping proper segment inertial properties and scaling
    # them.
    boneMassProps = computeMassProperties_Mirtich1996(triGeom)
    bone_mass = boneMassProps['mass']*body_density # vol [mm^3] * density [kg/mm^3] 
    bone_COP = boneMassProps['COM']*dim_fact 
    bone_inertia = boneMassProps['Ivec']*body_density*dim_fact**2 # Ivec [mm^3 * mm^2] * density [kg/mm^3] 
    
    # create opensim body
    osim_body = opensim.Body(body_name, bone_mass, \
                             opensim.ArrayDouble.createVec3(bone_COP), \
                             opensim.Inertia(bone_inertia[0], bone_inertia[1], bone_inertia[2], \
                                             bone_inertia[3], bone_inertia[4], bone_inertia[5]))
    # add body to model
    osimModel.addBody(osim_body)
    
    # add visualization mesh
    if vis_mesh_file != '':
        vis_geom = opensim.Mesh(vis_mesh_file)
        vis_geom.set_scale_factors(opensim.Vec3(dim_fact))
        osimModel.attachGeometry(vis_geom)
       
    return osimModel





















