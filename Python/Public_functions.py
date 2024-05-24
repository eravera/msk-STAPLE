#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:39:06 2024

@author: emi
"""
# ----------- import packages --------------
import numpy as np
from stl import mesh
import pandas as pd
import os, shutil
from pathlib import Path
import sys
import time
import logging
import scipy.io as sio
import gmshparser

# def triangulation(T,P):
#     # -------------------------------------------------------------------------
#     # Equivalent to MatLab function. A matrix format to represent triangulations.
#     # This format has two parts:
#     # - The vertices, represented as a matrix in which each row contains the
#     # coordinates of a point in the triangulation.
#     # - The triangulation connectivity, represented as a matrix in which each 
#     # row defines a triangle or tetrahedron.
#     # 
#     # Inputs: 
#     #    T = list that define the connectivity 
#     #    P = list of points
#     #
#     # Output:
#     #     TR = dictionary of the triangulation from this data
#     # -------------------------------------------------------------------------
    

def load_mesh(a_tri_mesh_file):
    # -------------------------------------------------------------------------
    # LOAD_MESH Read a file, with specified or unspecified extension, as a
    # three-dimensional surface mesh file. The script guesses the triangulation
    # format when it is not specified, attempting to open the file as a
    # STL or MATLAB file.
    
    # Inputs:
    #   a_tri_mesh_file - a file path to a surface mesh, with extension .STL,
    #       .MAT, .MSH or no extension.
    
    # Outputs:
    #     tri_geom - a triangulation object.
    # -------------------------------------------------------------------------
    
    print('Attending to read mesh file: ' + a_tri_mesh_file)
    
    tri_geom = {}
    
    # check if there is a file that could be opened adding extension
    files = [f for f in os.listdir() if os.path.isfile(f)]
    
    if not (any('' in ext for ext in files) or any('.mat' in ext for ext in files) or any('.stl' in ext for ext in files) or any('.picle' in ext for ext in files)):
        print(a_tri_mesh_file + ' geometry not available.')
        return tri_geom
    
    if isinstance(a_tri_mesh_file, str):
        # get extention
        ext = Path(a_tri_mesh_file).suffix
        # if stl file just open it
        if ext == '.stl':
            tmp_tri_geom = mesh.Mesh.from_file(a_tri_mesh_file)
            
            Faces = tmp_tri_geom.vectors
            P = Faces.reshape(-1, 3)
            Vertex = np.zeros(np.shape(tmp_tri_geom.v0), dtype=np.int64)

            _, idx = np.unique(P, axis=0, return_index=True)
            Points = P[np.sort(idx)]

            for pos, elem in enumerate(tmp_tri_geom.v0):
                tmp = np.where(Points == elem)[0]
                if len(tmp) > 3:
                    l0 = []
                    l0 = list(tmp)
                    tmp1 = [x for x in l0 if l0.count(x) > 1]
                    Vertex[pos,0] = tmp1[0]
                else:
                    Vertex[pos,0] = tmp[0]
                    
            for pos, elem in enumerate(tmp_tri_geom.v1):
                tmp = np.where(Points == elem)[0]
                if len(tmp) > 3:
                    l0 = []
                    l0 = list(tmp)
                    tmp1 = [x for x in l0 if l0.count(x) > 1]
                    Vertex[pos,1] = tmp1[0]
                else:
                    Vertex[pos,1] = tmp[0]

            for pos, elem in enumerate(tmp_tri_geom.v2):
                tmp = np.where(Points == elem)[0]
                if len(tmp) > 3:
                    l0 = []
                    l0 = list(tmp)
                    tmp1 = [x for x in l0 if l0.count(x) > 1]
                    Vertex[pos,2] = tmp1[0]
                else:
                    Vertex[pos,2] = tmp[0]

            tri_geom = {'Points': Points, 'ConnectivityList': Vertex}
                        
            kwd = 'STL'
            
        # if matlab file just open it
        elif ext == '.mat':
            # tri_geom = sio.loadmat(a_tri_mesh_file, squeeze_me=True)
            # kwd = 'MATLAB'
            tri_geom = {}
        # if gmsh file just open it
        elif ext == '.msh':
            # tmp_mesh = gmshparser.parse(a_tri_mesh_file)
            # # Nodes
            # nid = tmp_mesh.get_node_entities().get_nodes().get_tag()
            # ncoord = tmp_mesh.get_node_entities().get_nodes().get_coordinates()
            # # Elements
            # elid = tmp_mesh.get_element_entities().get_elements().get_tag()
            # elcon = tmp_mesh.get_element_entities().get_elements().get_connectivity()
            
            # # VER ACA COMO GENERAR EL TIPO MESH o triangulation
            
            # kwd = 'GMSH'
            tri_geom = {}
        elif ext == '':
            try:
                # tri_geom = sio.loadmat(a_tri_mesh_file, squeeze_me=True)
                # kwd = 'MATLAB'
                tri_geom = {}
            except:
                # if does not have extension try to open stl file
                tmp_tri_geom = mesh.Mesh.from_file(a_tri_mesh_file)
                
                Faces = tmp_tri_geom.vectors
                P = Faces.reshape(-1, 3)
                Vertex = np.zeros(np.shape(tmp_tri_geom.v0), dtype=np.int64)

                _, idx = np.unique(P, axis=0, return_index=True)
                Points = P[np.sort(idx)]

                for pos, elem in enumerate(tri_geom.v0):
                    tmp = np.where(Points == elem)[0]
                    if len(tmp) > 3:
                        l0 = []
                        l0 = list(tmp)
                        tmp1 = [x for x in l0 if l0.count(x) > 1]
                        Vertex[pos,0] = tmp1[0]
                    else:
                        Vertex[pos,0] = tmp[0]
                        
                for pos, elem in enumerate(tri_geom.v1):
                    tmp = np.where(Points == elem)[0]
                    if len(tmp) > 3:
                        l0 = []
                        l0 = list(tmp)
                        tmp1 = [x for x in l0 if l0.count(x) > 1]
                        Vertex[pos,1] = tmp1[0]
                    else:
                        Vertex[pos,1] = tmp[0]

                for pos, elem in enumerate(tri_geom.v2):
                    tmp = np.where(Points == elem)[0]
                    if len(tmp) > 3:
                        l0 = []
                        l0 = list(tmp)
                        tmp1 = [x for x in l0 if l0.count(x) > 1]
                        Vertex[pos,2] = tmp1[0]
                    else:
                        Vertex[pos,2] = tmp[0]

                tri_geom = {'Points': Points, 'ConnectivityList': Vertex}
                
                kwd = 'STL'
                
    if not tri_geom:
        logging.exception(a_tri_mesh_file + ' could not be read. Please doucle chack inputs. \n')
    else:
        # use GIBOC function to fix normals
        # tri_geom = TriFixNormals(tri_geom)
        # file is read
        print('...read as ' + kwd + ' file.')
    
    return tri_geom

# -----------------------------------------------------------------------------
def freeBoundary(Tr = {}):
    # -------------------------------------------------------------------------
    # returns the free boundary points of the triangles or tetrahedra in Tr. 
    # A point in TR is on the free boundary if it is referenced by only one triangle or tetrahedron.
    
    FreeB = {}
    FreeB['ID'] = []
    FreeB['Coord'] = []

    for IDpoint, Point in enumerate(Tr['Points']):
        vertex_triangle = []
        CloudPoint = []
        tmp_norm = []
        
        # identify the triangles whit this vertex
        vertex_triangle = list(np.where(Tr['ConnectivityList'] == IDpoint)[0])
        
        # identify the neighborhood of points (all point from triangles that inlcude Point)
        for neighbor in vertex_triangle:
            
            v0 = Tr['Points'][Tr['ConnectivityList'][neighbor, 0]]
            v1 = Tr['Points'][Tr['ConnectivityList'][neighbor, 1]]
            v2 = Tr['Points'][Tr['ConnectivityList'][neighbor, 2]]
            
            if np.linalg.norm(v0 - Point) != 0:
                CloudPoint.append(v0)
            if np.linalg.norm(v1 - Point) != 0:
                CloudPoint.append(v1)
            if np.linalg.norm(v2 - Point) != 0:
                CloudPoint.append(v2)
        
        # for each neighborhood compute the norm with the another neighborhood. 
        # If this norm is zero in minus of two times, this point is in the bounder
        for neig in CloudPoint:
            tmp_norm = [np.linalg.norm(neig - val) for val in CloudPoint]
            if tmp_norm.count(0) < 2:
                # duplicate points
                FreeB['ID'].append(IDpoint)
                FreeB['Coord'].append(Point)

    # remove duplicate points
    FreeB['ID'] = FreeB['ID'][::2]
    FreeB['Coord'] = FreeB['Coord'][::2]
    
    return FreeB 