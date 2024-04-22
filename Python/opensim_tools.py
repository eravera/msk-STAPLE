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


from Public_functions import load_mesh

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